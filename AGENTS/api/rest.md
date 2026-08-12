# REST API 参考

所有端点挂载在 `/api/v1` 下。响应 JSON 不包含服务器绝对路径、PID 或完整
traceback;客户端也不允许提交任何服务器文件路径。

## 健康与状态

### `GET /api/v1/health`

进程存活检查。Worker `stopped` 属于正常的惰性状态，不影响健康。

```json
{ "status": "ok" }
```

### `GET /api/v1/status`

队列与运行时状态。`pid` 和模型绝对路径不会出现在公开响应中。

```json
{
  "queue": 1,
  "running": "job-uuid-or-null",
  "worker": "ready",
  "gpu": "10.42/15.92 GiB",
  "memory": {"used_gib": 12.3, "total_gib": 31.8, "percent": 38.7},
  "gpu_memory_used_gib": 10.42,
  "gpu_memory_total_gib": 15.92,
  "gpu_memory_percent": 65.5,
  "gpu_utilization_percent": 92.0,
  "loaded_resources": {
    "workload": "wan2.2-ti2v-5b",
    "model": "wan2.2-ti2v-5b.safetensors",
    "vae": "wan2.2_vae.safetensors",
    "text_encoder": "umt5_xxl.safetensors",
    "clip_type": "wan"
  }
}
```

`worker` 为 `ready` 或 `stopped`(尚未执行首个任务或 stop 之后)。资源探测最多约 2 秒
陈旧，也可能为 `查询中` / `不可用`；数值字段在不可用时为 `null`。`memory` 是系统
RAM，`gpu_memory_*` 是显存容量，`gpu_utilization_percent` 是 GPU 核心利用率。
`loaded_resources` 是 Worker 当前加载的资源
(workload/model/vae/text_encoder/clip_type),路径字段只公开文件名,未加载任何
资源时为 `null`。

## 资源

### `GET /api/v1/modes`

返回服务端实际配置的模式及加载能力。前端应以此列表渲染模式选项，并根据
`requires_vae` 决定是否请求用户选择外置 VAE，不应根据 mode 名称推断。

```json
{
  "modes": [
    {"mode": "zit", "model_loader": "components", "requires_vae": true},
    {"mode": "sdxl", "model_loader": "checkpoint", "requires_vae": false}
  ]
}
```

### `GET /api/v1/resources/{mode}/{kind}`

`mode` ∈ `zit|krea2|zib|sdxl`,`kind` ∈ `diffusion|vae`。SDXL 的 `diffusion`
列表实际是完整 checkpoint，`vae` 列表为空。

```json
{
  "resources": [
    {
      "index": 1,
      "name": "model.safetensors",
      "display_name": "portrait (model.safetensors)",
      "alias": "portrait"
    }
  ]
}
```

`index` 按文件名排序从 1 开始，目录内容变化后会重新排序，**不是永久资源 ID**。
客户端应先拉取列表再提交任务。响应不含 `path`。

### `PUT /api/v1/resources/{mode}/{kind}/{index}/alias`

请求体 `{"alias": "portrait"}`。alias 在同一 `(mode, kind)` 内唯一。

- 200 `{"ok": true}`
- 404 index 不存在
- 422 alias 为空或与其他资源冲突

## 任务

### `POST /api/v1/jobs`

提交一张或一批任务。text encoder 与 clip type 由服务器按 mode 固定，不接受客户端
提交模型路径或覆盖；负面提示词、steps、CFG、sampler、scheduler 是任务级参数，
客户端可按任务提交，省略时取默认值。

请求体：

```json
{
  "mode": "krea2",
  "model_index": 1,
  "vae_index": 1,
  "prompt": "an adult studio portrait",
  "negative_prompt": "blurry, watermark",
  "width": 576,
  "height": 576,
  "steps": 8,
  "seed": -1,
  "count": 1,
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
    "cfg": null,
    "sampler": null,
    "scheduler": null,
    "seed": null
  }
}
```

约束:`mode` ∈ `zit|krea2|zib|sdxl`;`model_index` ≥ 1；ZIT/Krea2/ZIB 的
`vae_index` 必填且 ≥ 1，SDXL 必须省略 `vae_index`；`prompt` 1–16000
字符；`negative_prompt` 可为空、最长 16000 字符；宽高必须是 16 的倍数；
`steps` 1–100（默认 8);`seed` ≥ -1(`-1` 表示随机）;`count` 1–32(HTTP
admission limit);`cfg` 为大于 0 的有限浮点数（默认 1.0);`sampler`/`scheduler`
的合法值以 core domain 为唯一事实来源（当前 44 个采样器、9 个调度器），完整
列表见 `GET /api/v1/sampling-options`。

`upscale` 为可选的图片放大参数，省略或 `enabled=false` 时只输出原始尺寸图片。
启用后任务会额外生成 `<name>-upscale.png`；字段约束（`method` ∈
`resize|upscale_model|latent_hires`、`scale` 大于 1 且不超过 4、`start_step` 必须
小于 `steps`、模型超分的 `tile`/`overlap` 等）以 core domain 校验为准，
`model_index` 引用 `GET /api/v1/upscale-models` 的 index，不接受任何路径。
`cfg`/`sampler`/`scheduler`/`seed` 为 `null` 表示继承首次采样的实际值。

### `GET /api/v1/sampling-options`

返回全部可用采样器/调度器及默认值，供客户端渲染选项：

```json
{
  "samplers": ["euler", "dpmpp_2m_sde", "..."],
  "schedulers": ["simple", "sgm_uniform", "beta", "..."],
  "defaults": { "sampler": "euler", "scheduler": "simple" }
}
```

响应 202（已持久化并入队，不代表图片完成）:

```json
{
  "jobs": [
    {
      "id": "job-uuid",
      "batch_id": null,
      "status": "queued",
      "mode": "krea2",
      "seed": -1,
      "width": 576,
      "height": 576,
      "steps": 8,
      "sampler": "euler",
      "scheduler": "simple",
      "cfg": 1.0,
      "model_name": "model.safetensors",
      "vae_name": "vae.safetensors",
      "prompt": "an adult studio portrait",
      "negative_prompt": "blurry, watermark",
      "submitted_at": "2026-07-27T13:00:00+08:00",
      "started_at": null,
      "completed_at": null,
      "duration_seconds": null,
      "output_name": "krea2-00009.png",
      "image_url": null,
      "error": null,
      "upscale": {
        "enabled": true,
        "method": "latent_hires",
        "scale": 2.0,
        "interpolation": "bislerp",
        "model_name": null,
        "tile": 512,
        "overlap": 32,
        "steps": 9,
        "start_step": 4,
        "cfg": null,
        "sampler": null,
        "scheduler": null,
        "seed": null
      },
      "upscaled_output_name": "krea2-00009-upscale.png",
      "upscaled_image_url": null
    }
  ]
}
```

- `count > 1` 时所有任务共享同一个 `batch_id`,seed 各自独立。
- 随机 seed 的任务此刻仍是 `-1`，实际 seed 由 `job_started` 事件公布并写回 SQLite,
  之后可从 `GET /api/v1/jobs/{job_id}` 读取。
- `image_url` 在原图文件实际存在时为 `/api/v1/images/{job_id}`(放大失败或取消时
  原图可能已保存，不以 `status == "completed"` 为条件）;`upscaled_image_url` 同理，
  指向 `/api/v1/images/{job_id}/upscaled`;`upscaled_output_name` 仅启用放大时存在。
- `error` 只含首行摘要，完整 traceback 只保留在服务端 SQLite。

错误:404 资源 index 不存在；422 参数校验失败。

### `GET /api/v1/upscale-options`

图片放大的方法、两组插值、采样器/调度器与默认值，静态取值直接来自 core domain:

```json
{
  "methods": ["resize", "upscale_model", "latent_hires"],
  "image_interpolations": ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
  "latent_interpolations": ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],
  "samplers": ["euler", "..."],
  "schedulers": ["simple", "..."],
  "defaults": {
    "method": "latent_hires",
    "scale": 2.0,
    "interpolation": "bislerp",
    "tile": 512,
    "overlap": 32,
    "steps": 9,
    "start_step": 4
  }
}
```

`image_interpolations` 用于 `resize`/`upscale_model`,`latent_interpolations` 用于
`latent_hires`。

### `GET /api/v1/upscale-models`

服务端 `workbench.yaml` 全局 `upscaling.models` 目录中的放大模型，只公开
index/name/display_name，不含绝对路径：

```json
{
  "models": [
    { "index": 1, "name": "4x-UltraSharp.pth", "display_name": "4x-UltraSharp.pth", "alias": null }
  ]
}
```

index 同样按文件名排序，目录变化后会重排，不是永久 ID。

### `GET /api/v1/jobs/{job_id}`

读取单个任务的持久化状态，响应字段同上（单个 JobResponse 对象）。客户端断线重连后
用它恢复任务结果。404 表示 job 不存在。

### `GET /api/v1/jobs`

历史任务，按提交时间倒序 keyset 分页。

查询参数:`status`(queued|running|completed|failed|cancelled)、`mode`(zit|krea2|zib|sdxl)、
`limit`(1–200，默认 50)、`cursor`（上一页返回的 `next_cursor`)。

```json
{
  "jobs": [ /* JobResponse[] */ ],
  "next_cursor": "2026-07-27T13:00:00+08:00|job-uuid"
}
```

`next_cursor` 为 `null` 表示没有更多记录。422 表示 cursor 无效。

## 控制

### `POST /api/v1/control/stop`

**全局停止**：取消正在运行的任务并清空整个等待队列。图片 Worker 和已加载的模型
资源保留；被取消的视频任务会释放其视频模型资源，下一张任务按需重新加载。本服务没有单任务取消端点。

```json
{ "accepted": true, "scope": "running-and-entire-queue" }
```

可重复调用（幂等）。

### `POST /api/v1/control/skip`

**跳过当前任务**：只取消正在运行的那一张，队列其余任务继续执行。图片任务保留 Worker
与已加载资源；被取消的视频任务会释放视频模型资源。没有正在运行或可取消的任务时不算错误。

有任务被跳过：

```json
{ "accepted": true, "skipped_job_id": "job-uuid", "scope": "current-job-only" }
```

空闲（或当前任务尚不可取消）:

```json
{ "accepted": false, "skipped_job_id": null, "reason": "no-cancellable-job" }
```

### `POST /api/v1/control/release`

**释放资源**:让 Worker 卸载已加载的模型/VAE/text encoder 等资源(Worker 进程保留,
下一个任务按需重新加载)。只有在没有运行任务且队列为空时才允许,否则 409;
释放超时 504。前端资源面板轮询 `/status` 的 `loaded_resources` 展示当前占用。

```json
{ "released": true }
```

Worker 取消失败返回 500。

## 图片

### `GET /api/v1/images/{job_id}`

受控下载任务的原始 PNG(`image/png`)。服务端校验存储路径必须位于配置的
`output_dir` 内、文件确实存在。放大阶段失败或取消时原图可能已保存，因此不再以
`status == completed` 为条件，以受控路径和文件实际存在为准。

- 200 PNG 字节流
- 404 job 不存在，或任务未完成且文件不存在
- 410 任务已到终态但文件已丢失
- 500 存储路径越出 output_dir（数据异常）

没有也不应有按路径读取任意文件的端点。

### `GET /api/v1/images/{job_id}/upscaled`

受控下载任务的放大图 PNG(`<name>-upscale.png`)。`upscaled_output_path` 为 `None`
表示任务未启用放大。

- 200 PNG 字节流
- 404 job 不存在、任务未启用放大，或任务仍在队列/执行中
- 410 任务已到终态但放大文件已丢失
- 500 存储路径越出 output_dir（数据异常）

### `GET /api/v1/images/{job_id}/upscaled/metadata`

同 `/images/{job_id}/metadata`，读取放大图的内嵌元数据；`artifact` 反映放大后的
真实尺寸与 `kind: "upscaled"`。

### `GET /api/v1/images/{job_id}/metadata`

读取 PNG 内嵌的生成参数。**图片是独立产物：展示层应以图片元数据为准，SQLite 只做
日志和队列状态。**响应已脱敏，只含生成参数和资源文件名（不含绝对路径、SHA-256):

```json
{
  "mode": "zit",
  "prompt": "...",
  "negative_prompt": "...",
  "negative_conditioning": "positive_reused",
  "width": 576,
  "height": 576,
  "steps": 8,
  "seed": 42,
  "cfg": 1.0,
  "sampler": "euler",
  "scheduler": "simple",
  "model_name": "model.safetensors",
  "vae_name": "vae.safetensors",
  "artifact": { "kind": "original", "width": 576, "height": 576 },
  "upscale": {
    "enabled": true,
    "method": "upscale_model",
    "scale": 2.0,
    "interpolation": "lanczos",
    "model_name": "4x-UltraSharp.pth",
    "tile": 512,
    "overlap": 32,
    "steps": 9,
    "start_step": 4,
    "cfg": null,
    "sampler": null,
    "scheduler": null,
    "seed": null
  }
}
```

`negative_conditioning` 为 `encoded_negative_prompt`(ZIB 或 CFG ≠ 1 时 Core 实际
编码了负面条件）或 `positive_reused`。读取 metadata schema v1 的旧 PNG 时，
`negative_prompt` 回退为空字符串、`negative_conditioning` 为 `null`。

`artifact` 是当前文件的真实尺寸与产物类型（`original` / `upscaled`);
`width`/`height` 仍表示首次生成尺寸。`upscale` 来自任务级放大参数，
`model_path` 等绝对路径已脱敏为 `model_name`。
SDXL 的 `vae_name` 为 `null`，因为 VAE 来自同一 checkpoint。

- 200 元数据对象
- 404 job 不存在、任务未完成，或 PNG 不含生成元数据
- 410 文件丢失；422 元数据损坏或 schema 版本不支持；500 路径越界

## 相册

相册**不依赖 jobs 表**（图片可能被删除/移动），每次请求重新扫描目录下的
`*.png` 与 `*.mp4` 并按修改时间倒序返回；PNG 尺寸解析结果按 `(mtime_ns, size)` 做内存缓存（mp4 不解析尺寸，宽高记 0),
文件变化或消失自动失效。

相册支持多目录：内置目录 `output`（配置中的输出目录，不可改名/删除）加用户
添加的目录。用户目录持久化在 cache 目录的 `album_dirs.json`，删除目录只移出
列表，不删除磁盘文件。

### `GET /api/v1/album/dirs`

```json
{
  "dirs": [
    { "id": "output", "name": "默认", "builtin": true },
    { "id": "a1b2c3d4", "name": "示例", "path": "E:/images", "builtin": false }
  ]
}
```

### `POST /api/v1/album/dirs`

请求体 `{"name": "...", "path": "..."}`。`path` 必须是本机已存在的目录，
422 表示名称为空或目录不存在；成功返回 201 与新目录对象。

### `PUT /api/v1/album/dirs/{dir_id}`

仅可改名：请求体 `{"name": "..."}`。内置目录返回 400，不存在返回 404。

### `DELETE /api/v1/album/dirs/{dir_id}`

从列表移除目录（不删除磁盘文件）。内置目录 400，不存在 404。

### `GET /api/v1/album`

查询参数:`limit`(1–200，默认 60)、`cursor`（上一页的 `next_cursor`)、
`dir`（目录 id，默认 `output`)、`subdir`（可选，只看某个一级子目录；子目录内
为深度查找，含更深层级，值须为单级目录名，否则 422）。

```json
{
  "images": [
    {
      "id": "2026-07-27/zit-00005.png",
      "name": "zit-00005.png",
      "mtime_ns": 1780000000000000000,
      "size_bytes": 845312,
      "kind": "image",
      "width": 576,
      "height": 576
    }
  ],
  "next_cursor": "1780000000000000000|2026-07-27/zit-00005.png"
}
```

`id` 是相对于所在目录的 POSIX 相对路径，URL 中使用前需编码;`kind` 为 `image` 或
`video`(mp4 的 `width`/`height` 为 0)。422 表示 cursor
无效；`dir` 不存在返回 404，目录不可访问返回 410。

### `GET /api/v1/album/subdirs`

列出 `dir`（目录 id，默认 `output`）下的第一级子目录名（通常按日期分目录），
按名称倒序（新的在前），不再往下深入。配合 `/album` 的 `subdir` 参数使用。

```json
{"subdirs": ["2026-07-28", "2026-07-27"]}
```

### `GET /api/v1/album/image/{path}`

读取图片或视频（按后缀返回 `image/png` 或 `video/mp4`)，查询参数 `dir` 同上。路径越出目录返回 400，文件不存在返回 404。

### `GET /api/v1/album/image/{path}/poster`

视频封面：ffmpeg 抽取首帧的 JPEG，查询参数 `dir` 同上。封面按
（目录, 路径, mtime, 大小）缓存到 cache 目录的 `album_posters/`，文件被替换后
自动重新生成。iOS(WebKit) 不会为 `preload="metadata"` 的 `<video>` 渲染首帧，
相册网格的视频封面必须用本端点的静态图。仅支持 mp4，其他文件返回 404；
服务器没有 ffmpeg 返回 503。

### `DELETE /api/v1/album/image/{path}`

从本机删除该图片，查询参数 `dir` 同上，返回 `{"deleted": "<path>"}`。路径校验与
读取一致：越界 400，文件不存在 404。

### `POST /api/v1/album/batch-delete`

批量删除图片/视频。请求体：`{"dir": "output", "relpaths": ["2026-07-27/a.png", "..."]}`，
`relpaths` 为非空字符串列表（≤200 个）。逐个解析并删除，单个失败不影响其余；
已不存在的文件视为删除成功（幂等）。返回：

```json
{"deleted": ["a.png"], "failed": [{"relpath": "b.png", "reason": "..."}]}
```

`relpaths` 为空、非列表或含非字符串项时返回 422。

### `GET /api/v1/album/image/{path}/metadata`

PNG 同 `/images/{job_id}/metadata`，返回脱敏的内嵌生成参数，查询参数 `dir` 同上。
若 PNG 不含本应用的 iTXt 块，回退解析 ComfyUI 原生 `prompt` tEXt 块（API prompt
节点图），返回 `source: "comfyui"` 的脱敏参数：`model_name`/`vae_name`/
`text_encoder_name` 为对应加载器的文件名（仅 basename），`mode` 按 unet 文件名
推断（krea/zib/sdxl/zit，匹配不到为 null），采样器/调度器/步数/cfg/seed 取文档序
第一个 KSampler 或 KSamplerAdvanced，正/负提示词取其 positive/negative 链接的
CLIPTextEncode 文本（经 ConditioningZeroOut 透传一层）；取不到的字段为 null。
两者都没有时返回 404。
mp4 没有内嵌元数据,改为按输出文件(日期目录名 + 文件名)反查 jobs 表的视频
任务记录,返回脱敏的视频生成参数:

```json
{
  "video_model": "wan2.2-ti2v-5b",
  "generation_type": "t2v",
  "prompt": "...",
  "negative_prompt": "",
  "width": 704,
  "height": 960,
  "duration_seconds": 5,
  "fps": 24,
  "length": 121,
  "steps": 20,
  "seed": 12345,
  "cfg": 5.0,
  "shift": 8.0,
  "denoise": 1.0,
  "sampler": "uni_pc",
  "scheduler": "simple",
  "model_name": "wan2.2-ti2v-5b.safetensors",
  "vae_name": "wan2.2_vae.safetensors"
}
```

找不到对应生成记录(如外部拷贝进来的 mp4)返回 404。

## 设置

设置项集中保存在 cache 目录的 `settings.json`（与数据库同目录），后续新设置项在
同一文件扩展。

### `GET /api/v1/settings`

返回全部设置项：

```json
{
  "size_presets": [[576, 576], [768, 768], [1024, 1024], [960, 1280]],
  "wan_video_size_presets": [[704, 960], [960, 704], [1280, 720], [720, 1280]],
  "minimax_video_size_presets": [],
  "prompt_presets": [
    {"id": "a1b2", "title": "通用质量词", "kind": "positive", "text": "masterpiece, best quality"}
  ],
  "sampling_defaults": {
    "zit": {"steps": 8, "sampler": "euler", "scheduler": "simple", "cfg": 1},
    "krea2": {"steps": 8, "sampler": "euler", "scheduler": "simple", "cfg": 1},
    "zib": {"steps": 8, "sampler": "euler", "scheduler": "simple", "cfg": 1},
    "sdxl": {"steps": 8, "sampler": "euler", "scheduler": "simple", "cfg": 1}
  },
  "llm": {
    "interface": "ollama",
    "base_url": "http://127.0.0.1:11434",
    "api_key": "",
    "model": "",
    "system_prompt": "",
    "sd_system_prompt": "适合 Stable Diffusion / SDXL 的提示词要求",
    "format_prompt": "输出 Positive/Negative 的格式要求",
    "language_prompt": "输出语言要求,包含 {language} 占位符",
    "think": false,
    "think_effort": ""
  }
}
```

`llm.system_prompt` 是原有的 FLUX 风格用户自定义提示词，保留该字段以兼容已有设置；
`llm.sd_system_prompt` 是独立的 SD/SDXL 风格用户自定义提示词。两者只在对应风格的
提示词生成请求中使用。

### `PUT /api/v1/settings`

部分更新：请求体只需包含要修改的设置项，校验通过后原子写入并返回完整设置。
`size_presets` 与视频表单尺寸标签的每项必须是 `[宽, 高]`，正整数且自动去重；宽高倍数
按系列区分：`size_presets` 与 `wan_video_size_presets`（wan 系列视频尺寸标签）要求 16 的倍数，
`minimax_video_size_presets`（MiniMax 系列视频尺寸标签）要求 32 的倍数。
旧版统一的 `video_size_presets` 在加载时自动迁移为 `wan_video_size_presets`。
未知设置项或非法值返回 422。

`sampling_defaults` 是工作台表单「重置」按钮使用的各模式默认采样参数，键为模式名
（`zit` / `krea2` / `zib` / `sdxl`）。允许只提交部分模式或部分字段，服务端用默认值补齐；
`steps` 为 1-100 的整数，`cfg` 为大于 0 的数值，`sampler` / `scheduler` 必须是
`GET /api/v1/sampling-options` 返回的合法值。

`llm` 可以局部提交，服务端会用默认值补齐未提交字段。`interface` 只允许 `ollama`
或 `openai`；`think` 必须是布尔值，其余大模型设置字段必须是字符串。

`prompt_presets` 是设置页「提示词」tab 管理的常用提示词预设，整表替换式提交。每项为
`{"id", "title", "kind", "text"}`：`id` 是客户端生成的非空字符串，`title` 不能为空，
`kind` 只允许 `positive`（正面）或 `negative`（负面），`text` 为提示词内容字符串。

## 大模型提示词生成

### `POST /api/v1/prompt-assist`

一次性生成提示词。请求体：

```json
{
  "instruction": "雨夜东京街头的人像摄影",
  "language": "en",
  "prompt_style": "sd"
}
```

`language` 只允许 `en` 或 `zh`；`prompt_style` 只允许 `flux` 或 `sd`，省略时默认为
`flux`，兼容旧客户端。服务端依次拼接对应风格的用户自定义提示词、输出格式规范和
语言要求。响应包含 `positive`、`negative`、`raw`、`parsed` 和独立的 `session_id`。

### `POST /api/v1/prompt-assist/chat`

请求字段与非流式接口相同，另可提交 `session_id`。响应为 SSE，事件顺序为
`session`，若干 `thinking` / `content`，最后是 `done` 或 `error`。只有语言和
`prompt_style` 均一致时才复用会话历史；任一项变化都会创建新 session，避免不同
提示词上下文相互污染。

## 模型信息

模型列表来自 Core 资源目录；别名存 SQLite aliases 表（与工作台表单共用），备注与
量化类型存 cache 目录 `model_info.json`，封面存 cache 目录 `covers/{mode}/`。
量化类型不按文件名猜测，初始为 `null`，由前端按需调用 quant 端点扫描文件后写入。

### `GET /api/v1/models/{mode}`

```json
{
  "models": [
    {
      "index": 1,
      "name": "model.safetensors",
      "alias": "人像",
      "mode": "zit",
      "size_bytes": 123456789,
      "quant": "bf16",
      "note": "常用",
      "has_cover": true,
      "cover_url": "/api/v1/models/zit/model.safetensors/cover"
    }
  ]
}
```

### `POST /api/v1/models/{mode}/{name}/quant`

扫描该模型 safetensors 头部，按权重实际 dtype 判断量化方式（fp8/int8 为量化，
混合时以 `+` 连接，如 `fp8+int8`；纯 bf16/fp16 全精度文件如实标注），写入
`model_info.json` 后返回 `{"ok": true, "quant": "fp8"}`。扫描结果按 size/mtime
缓存；name 不存在 404，无法从文件识别（非 safetensors 或头部异常）422。

### `GET /api/v1/models/{mode}/{name}/cover`

返回封面图片；无封面 404。

### `PUT /api/v1/models/{mode}/{name}/cover`

请求体为图片字节，`Content-Type` 限 `image/png`、`image/jpeg`、`image/webp`（其他 415)，
最大 10MB。同一模型重复上传会替换旧封面（含不同扩展名）。

### `PUT /api/v1/models/{mode}/{name}/info`

请求体 `{"alias": "...", "note": "..."}`，字段均可选。alias 校验同资源 alias 规则
（同 mode/kind 内唯一，冲突 422);name 不存在 404。
`output_path` 是生成时的路径快照；图片移动或删除不影响 SQLite 任务历史，`GET /jobs`
和 `GET /jobs/{job_id}` 仍会返回完整记录。

## 视频

视频生成（Wan TI2V-5B/I2V-14B、MiniMax H3 FL2VA/Ref2VA）经独立端点暴露；任务仍走同一条串行队列，全局 stop/skip 对
视频任务同样生效。不带输入图片为 T2V(文生视频),带 `input_image_id` 为 I2V
(图生视频);text encoder 由服务端 `video_resources` 配置固定注入,客户端不能指定；H3
的音频 VAE 同样由服务端固定注入，不出现在资源选择响应中。

### `GET /api/v1/video/models`

返回服务端实际配置的视频模型及生成能力。前端应以 `requires_input_image` 控制输入图片
是否必选，并用 `generation_types` 决定可展示的生成类型：

```json
{ "video_models": [
  {
    "video_model": "wan2.2-ti2v-5b",
    "label": "Wan 2.2 TI2V-5B",
    "generation_types": ["t2v", "i2v"],
    "requires_input_image": false
  },
  {
    "video_model": "wan2.2-i2v-14b",
    "label": "Wan 2.2 I2V-14B",
    "generation_types": ["i2v"],
    "requires_input_image": true
  },
  {
    "video_model": "minimax-h3",
    "label": "MiniMax H3",
    "generation_types": ["t2v", "i2v", "r2v"],
    "requires_input_image": false
  }
] }
```

### `GET /api/v1/video/models/{video_model}/resources`

指定视频模型可选的 diffusion/VAE 资源,字段与 `/resources/{mode}/{kind}` 一致
(index/name/display_name,视频资源没有 alias 机制,`alias` 固定为 `null`);
`video_model` 未配置返回 404。

Wan 2.2 I2V-14B 的 diffusion 列表可以同时包含 safetensors 和 `.gguf`。前端应将
包含 `_high_noise_` 的文件作为 high-noise 选项，将包含 `_low_noise_` 的文件作为
low-noise 选项；两者必须是文件名互换 `_high_noise_`/`_low_noise_` 后的精确配对，
不能混用 FP8 与 GGUF。选择 GGUF 时服务端要求 ComfyUI 安装 `ComfyUI-GGUF`；VAE
列表只包含 safetensors。

```json
{
  "models": [
    { "index": 1, "name": "wan2.2-ti2v-5b.safetensors", "display_name": "wan2.2-ti2v-5b.safetensors", "alias": null }
  ],
  "vaes": [
    { "index": 1, "name": "wan2.2_vae.safetensors", "display_name": "wan2.2_vae.safetensors", "alias": null }
  ]
}
```

### `POST /api/v1/video/input-images`

受控上传 I2V 输入图片:请求体为图片二进制,`Content-Type` 只接受
`image/png`、`image/jpeg`、`image/webp`,大小 ≤ 10MB,服务端用 PIL 校验确实是
有效图片。图片落盘到 cache 下的 `video_inputs/` 目录,文件名是服务端生成的
`<uuid4 hex><扩展名>`,客户端只拿到这个 id,之后不能提交任何服务器路径。

响应 201:`{ "id": "....png", "url": "/api/v1/video/input-images/....png" }`。
422 表示类型不支持、内容为空、超限或不是有效图片。

### `GET /api/v1/video/input-images/{image_id}`

按 id 读取受控上传的输入图片。id 先过白名单正则再做路径解析,越界、非法
id 或文件不存在一律 404。

### `POST /api/v1/video/input-images/from-album`

把相册里已存在的图片直接导入为 I2V 输入图片(服务端本地复制,不经客户端
上传)。请求体 `{"id": "<相册 relpath>", "dir": "output"}`,`id`/`dir` 语义与
相册端点一致,路径解析复用相册的受控逻辑;图片校验(类型/大小/PIL 可解码)
与上传接口相同。响应 201 与上传接口一致(`{id, url}`);400 路径越界、404
目录或文件不存在、422 不是支持的图片。

### `POST /api/v1/video/jobs`

提交一个或一批视频任务。请求体:

```json
{
  "video_model": "wan2.2-ti2v-5b",
  "model_index": 1,
  "high_model_index": null,
  "low_model_index": null,
  "vae_index": 1,
  "prompt": "一只猫在雪地里奔跑",
  "negative_prompt": "",
  "width": 704,
  "height": 960,
  "duration_seconds": 5,
  "fps": 24,
  "steps": 20,
  "seed": -1,
  "count": 1,
  "cfg": 5.0,
  "shift": 8.0,
  "latent_multiplier": 1.0,
  "sampler": "uni_pc",
  "scheduler": "simple",
  "input_image_id": null
}
```

14B 请求必须同时提供 `high_model_index` 和 `low_model_index`；它们分别引用上述
high-noise/low-noise 列表中的 index，且必须是同格式的精确配对。`model_index` 对
14B 仍保留，用于兼容旧客户端，服务端会根据文件名自动解析配对；新客户端应优先
使用两个显式字段。TI2V-5B 等其他视频模型继续使用 `model_index`，并将
`high_model_index`/`low_model_index` 省略。`input_image_id` 对 TI2V-5B 可选:引用 `POST /video/input-images` 返回的 id,设置后为
I2V,不设置为 T2V。I2V-14B 必须设置 `input_image_id`，否则 API 返回 422；Core 会自动
配对 high-noise 与 low-noise diffusion 模型并执行双阶段采样。

约束:`video_model` 必填且必须是已配置的视频模型;`model_index`、`high_model_index`、
`low_model_index`、`vae_index`（适用时）均 ≥ 1;
`prompt` 1–16000 字符;宽高必须是 16 的倍数(默认 704×960);`duration_seconds`
为 ≥ 1 的整数秒(默认 5);`fps` 1–120(默认 24);总帧数
length = duration_seconds × fps + 1 且必须满足 4n + 1;`steps` 1–100(默认 20);
`seed` ≥ -1(`-1` 表示随机);`count` 1–8(HTTP admission limit);`cfg` 为大于 0
的有限浮点数(默认 5.0);`shift` 0–100(默认 8.0,ComfyUI `ModelSamplingSD3`,
前端表单同样开放 0–100);`sampler`/`scheduler` 的合法值与图片任务同源(core
domain 的 SAMPLERS/SCHEDULERS);`latent_multiplier` 为大于 0 的有限浮点数(默认
1.0),在采样前乘到 Wan latent 的 samples 上。Turbo 模型可按模型说明设置为
0.8 以缓解过饱和。请求未提供 `sampler` 时，I2V-14B 默认使用官方工作流的
`euler`，其他视频模型默认使用 `uni_pc`。

MiniMax H3 的 FL2VA 不带图片为 T2V，单张 `input_image_id` 为首帧 I2V；Ref2VA 模型使用
单张 `reference_image_id` 生成 R2V。第一阶段不支持多图、参考视频或参考音频，且
`input_image_id` 与 `reference_image_id` 不能同时提供。H3 宽高必须为 32 的倍数，单边为 32–1344，且总面积不超过 768×1344；FPS 固定 24，CFG 固定 1；`length` 从
`duration_seconds * 24` 向上对齐到 `17n+5`（5 秒对应 124 帧），因此实际媒体时长可
略长于请求秒数。H3 请求未提供 sampler 时默认 `res_multistep`，shift 默认 12；内部
audio shift 固定为 3。输出 MP4 含 H.264 视频和 32 kHz 双声道 AAC 音频。

响应 202(已持久化并入队,不代表视频完成):

```json
{
  "jobs": [
    {
      "id": "job-uuid",
      "batch_id": null,
      "status": "queued",
      "video_model": "wan2.2-ti2v-5b",
      "generation_type": "t2v",
      "seed": -1,
      "width": 704,
      "height": 960,
      "duration_seconds": 5,
      "fps": 24,
      "length": 121,
      "steps": 20,
      "sampler": "uni_pc",
      "scheduler": "simple",
      "cfg": 5.0,
      "shift": 8.0,
      "denoise": 1.0,
      "latent_multiplier": 1.0,
      "model_name": "wan2.2-ti2v-5b.safetensors",
      "vae_name": "wan2.2_vae.safetensors",
      "prompt": "一只猫在雪地里奔跑",
      "negative_prompt": "",
      "submitted_at": "2026-08-09T13:00:00+08:00",
      "started_at": null,
      "completed_at": null,
      "elapsed_seconds": null,
      "output_name": "wan2.2-ti2v-5b-00001.mp4",
      "video_url": null,
      "input_image_url": null,
      "error": null
    }
  ]
}
```

- `generation_type` 为 `t2v`、`i2v` 或 `r2v`(取决于输入图片/参考图片);
  `duration_seconds` 是视频时长秒,`elapsed_seconds` 是实际耗时。
- `video_url` 在 mp4 文件实际存在时为 `/api/v1/videos/{job_id}`,否则为 `null`。
- `input_image_url` 仅 I2V 任务有值,为 `/api/v1/video/input-images/{id}`；`reference_image_url` 仅 R2V 任务有值，路径格式相同。
- 错误:404 资源 index 或 input_image_id 不存在;422 参数校验失败(含未配置的
  video_model 以外的非法枚举值)。

### `GET /api/v1/video/jobs/{job_id}`

读取单个视频任务的持久化状态(单个 VideoJobResponse 对象)。404 表示 job 不存在。

### `GET /api/v1/videos/{job_id}`

受控下载视频任务的 mp4(`video/mp4`)。服务端校验存储路径必须位于配置的
`output_dir` 内、文件确实存在,规则与图片端点一致。

- 200 mp4 字节流
- 404 job 不存在,或任务未完成且文件不存在
- 410 任务已到终态但文件已丢失
- 500 存储路径越出 output_dir(数据异常)

### 视频模型卡片

`/api/v1/video-models/{video_model}` 下镜像图片 models 端点:list、cover
GET/PUT、quant POST、info PUT,响应字段与 `/models/{mode}` 一致(`mode` 为
video_model 值,`alias` 固定为 `null`)。备注/量化/封面复用同一个
`model_info.json` 与 `covers/` 目录,以 video_model 字符串作 key,与图片 mode
不会冲突。视频目录没有 alias 机制,info PUT 只支持 `note`,payload 带 `alias`
返回 422;`video_model` 未配置返回 404。

## 错误码汇总

| 场景 | HTTP |
|---|---:|
| Pydantic / 生成参数校验失败 | 422 |
| 资源 index / job 不存在 | 404 |
| alias 冲突 | 422 |
| 分页 cursor 无效 | 422 |
| 图片任务未完成 / 未启用放大 | 404 |
| 图片或放大文件丢失 | 410 |
| 存储路径越界 | 500 |
| Worker 运行时失败 | POST 已返回 202；经 job 状态和 `job_error` 事件报告 |
