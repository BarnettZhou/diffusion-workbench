# FastAPI 图片放大适配说明

Core 已支持任务级图片放大，但当前 `diffusion_workbench_api` 尚未把这些字段暴露为
HTTP 接口。本文件说明 FastAPI 与前端应如何映射现有 Core 能力；实现时不要在 API
层重新执行 ComfyUI workflow，也不要允许客户端提交服务器文件路径。

## 1. Core 调用契约

提交任务时构造 `UpscaleSettings`，并放入 `GenerationSettings.upscale`：

```python
from diffusion_workbench_core.domain import UpscaleMethod, UpscaleSettings

upscale = UpscaleSettings(
    enabled=True,
    method=UpscaleMethod.LATENT_HIRES,
    scale=2.0,
    interpolation="bislerp",
    steps=9,
    start_step=4,
    cfg=1.0,
    sampler="dpmpp_2m_sde",
    scheduler="sgm_uniform",
    seed=None,
)
```

`seed=None`、`cfg=None`、`sampler=None`、`scheduler=None` 表示继承首次采样的实际值。
`steps=9`、`start_step=4` 表示二次采样跳过前 4 步，从第 5 步开始，实际执行
`9 - 4 = 5` 步。Core 不使用 `denoise + redraw_steps` 换算。

启用放大后，任务记录包含两条输出路径：

```text
krea2-00001.png
krea2-00001-upscale.png
```

Worker 先保存原图，再执行放大。放大阶段失败或被取消时，任务最终可能是 `failed` 或
`cancelled`，但已经保存的原图仍会保留。

## 2. 请求 Schema

建议在 `CreateJobsRequest` 中增加可选的嵌套对象 `upscale`。省略时等价于
`{"enabled": false}`。

```json
{
  "mode": "krea2",
  "model_index": 1,
  "vae_index": 1,
  "prompt": "portrait",
  "width": 576,
  "height": 576,
  "steps": 8,
  "cfg": 1.0,
  "sampler": "euler",
  "scheduler": "simple",
  "upscale": {
    "enabled": true,
    "method": "latent_hires",
    "scale": 2.0,
    "interpolation": "bislerp",
    "steps": 9,
    "start_step": 4,
    "cfg": 1.0,
    "sampler": "dpmpp_2m_sde",
    "scheduler": "sgm_uniform",
    "seed": null
  }
}
```

建议的 Pydantic 字段：

| 字段 | 类型/默认值 | 含义 |
|---|---|---|
| `enabled` | `bool = false` | 是否生成第二张放大图 |
| `method` | `resize \| upscale_model \| latent_hires` | 放大方法 |
| `scale` | `float = 2.0` | 大于 1 且不超过 4 |
| `interpolation` | `str` | 插值方式，取值依 method 而定 |
| `model_index` | `int \| null` | 仅 `upscale_model` 使用；不能传路径 |
| `tile` | `int = 512` | 仅模型超分使用，128～1024 且为 32 的倍数 |
| `overlap` | `int = 32` | 仅模型超分使用，`0 <= overlap < tile/2` |
| `steps` | `int = 9` | latent 二次采样的总步数 |
| `start_step` | `int = 4` | 跳过的前置步数，`0 <= start_step < steps` |
| `cfg` | `float \| null` | `null` 继承首次采样 CFG |
| `sampler` | `str \| null` | `null` 继承首次采样 sampler |
| `scheduler` | `str \| null` | `null` 继承首次采样 scheduler |
| `seed` | `int \| null` | `null` 继承任务实际 seed |

Pydantic 完成结构校验后，仍应构造 `UpscaleSettings` 并调用 `validate()`，让 Core
domain 保持合法值的唯一事实来源。

## 3. 三种方法示例

### 普通插值 `resize`

```json
{
  "enabled": true,
  "method": "resize",
  "scale": 2.0,
  "interpolation": "lanczos"
}
```

不加载额外模型，也不二次采样，资源占用最低。可用插值：`nearest-exact`、
`bilinear`、`area`、`bicubic`、`lanczos`。

### 模型超分 `upscale_model`

```json
{
  "enabled": true,
  "method": "upscale_model",
  "model_index": 1,
  "scale": 2.0,
  "interpolation": "lanczos",
  "tile": 512,
  "overlap": 32
}
```

模型来自服务端 `workbench.yaml` 的全局 `upscaling.models` 目录。API 应先通过
`core.list_upscale_models()` 按 index 解析为 `ResourceItem`，再传给
`UpscaleSettings.model`。禁止接收 `model_path`。Worker 使用 tiled scale；OOM 时会自动
把 tile 逐次减半，最低到 128。模型执行后移回 CPU，并可在 Worker 内复用。

模型原生倍率与请求倍率不一致时，Worker 会在模型超分后用所选 interpolation 调整到
目标倍率。

### Latent 重绘 `latent_hires`

```json
{
  "enabled": true,
  "method": "latent_hires",
  "scale": 2.0,
  "interpolation": "bislerp",
  "steps": 9,
  "start_step": 4,
  "cfg": null,
  "sampler": null,
  "scheduler": null,
  "seed": null
}
```

可用插值：`nearest-exact`、`bilinear`、`area`、`bicubic`、`bislerp`。该方法复用当前
diffusion、text encoder 和 VAE，不加载额外超分模型，但会进行二次扩散采样，耗时和
显存通常高于普通 resize。

## 4. 资源与选项端点

建议增加：

```text
GET /api/v1/upscale-options
GET /api/v1/upscale-models
```

`/upscale-options` 返回方法、两组 interpolation、sampler、scheduler 和默认值；静态
取值应直接导入 Core domain 常量。`/upscale-models` 调用
`core.list_upscale_models()`，响应只公开 index、name、display_name，不公开绝对路径。

```json
{
  "models": [
    {"index": 1, "name": "4x-UltraSharp.pth", "display_name": "4x-UltraSharp.pth"}
  ]
}
```

## 5. Service 层映射

`submit_jobs()` 保持现有 model/VAE 解析逻辑，再增加：

1. `enabled=false`：构造默认 `UpscaleSettings()`。
2. `upscale_model`：用 `model_index` 在 `core.list_upscale_models()` 中查找
   `ResourceItem`；不存在返回 404。
3. 其余字段构造 `UpscaleSettings`；校验错误返回 422。
4. 把结果传给 `GenerationSettings(upscale=upscale)`。

不要在 API 层计算输出文件名、执行 resize 或调用 ComfyUI。双路径分配、串行执行、
原图先保存和失败状态都由 Core 负责。

## 6. Job 响应与下载

建议 `JobResponse` 增加：

```json
{
  "upscale": {
    "enabled": true,
    "method": "latent_hires",
    "scale": 2.0,
    "steps": 9,
    "start_step": 4
  },
  "upscaled_output_name": "krea2-00001-upscale.png",
  "upscaled_image_url": "/api/v1/images/{job_id}/upscaled"
}
```

保留现有原图端点，并增加：

```text
GET /api/v1/images/{job_id}              # 原图
GET /api/v1/images/{job_id}/upscaled     # 放大图
GET /api/v1/images/{job_id}/metadata
GET /api/v1/images/{job_id}/upscaled/metadata
```

文件路由必须分别使用 `job.output_path` 与 `job.upscaled_output_path`，并继续校验解析后的
路径位于 `core.config.output_dir` 内。由于放大失败或取消时原图可能已经存在，原图下载
不应再只以 `status == completed` 为条件；以受控路径和文件实际存在为准。放大图路径为
`None` 表示任务未启用放大，返回 404；有记录但文件丢失可返回 410。

元数据 schema v3 的 `parameters.width/height` 仍表示首次生成尺寸；当前文件的真实尺寸与
产物类型位于 `artifact.width/height/kind`。放大图 metadata 响应应公开 `artifact` 和
`parameters.upscale`，不要继续只返回首次生成尺寸。

公共事件转换可在 `job_finished` 或 `upscale_saved` 时增加 `upscaled_image_url`。不要把
`output_path`、`upscaled_output_path` 的服务器绝对路径发送给浏览器。

## 7. 前端映射

前端可参考现有 ZIT/Krea2/ZIB 生成表单，增加一个“图片放大”区域或 tab：

- 总开关默认关闭；关闭时保留用户刚才的参数。
- 方法切换后只显示该方法相关字段。
- `upscale_model` 通过 `/upscale-models` 下拉选择，不允许自由输入路径。
- `latent_hires` 直接展示“总步数”和“开始步数”，并实时显示
  `实际执行步数 = steps - start_step`。
- 提交时把整个表单复制为任务快照；任务开始后修改 UI 不影响已排队任务。
- 完成后同时展示原图和放大图；放大失败时若原图 URL 可用，仍展示原图并标明放大失败。

首版不需要在前端实现 `denoise`、`redraw_steps` 或 tiled diffusion redraw。
