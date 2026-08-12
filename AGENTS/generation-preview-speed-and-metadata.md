# 生成预览、速度与 PNG 元数据

本文定义采样期间的预览/性能事件，以及最终 PNG 内嵌生成参数的稳定契约。调用端只应
依赖这里列出的公共字段，不应读取 Worker 内部对象或解析普通 stdout。

## 1. 启用预览

预览默认关闭。FastAPI 等能够消费图片的调用端应在 Core 初始化后、提交任务前启用：

```python
core = WorkbenchCore.from_config("configs/workbench.yaml")
core.set_event_sink(event_hub.publish)
core.set_preview_enabled(True)
```

TUI 不调用 `set_preview_enabled(True)`，因此既不接收预览，也不承担 latent 转 JPEG 和
IPC 的开销。开关在任务开始时写入 Worker 命令；切换只影响之后开始执行的任务。

## 2. 采样性能事件

每个采样步骤发送一个 `step_progress`：

```json
{
  "type": "step_progress",
  "job_id": "job-uuid",
  "step": 3,
  "total": 8,
  "elapsed_seconds": 6.0,
  "step_seconds": 2.1,
  "seconds_per_step": 2.0,
  "steps_per_second": 0.5,
  "eta_seconds": 10.0
}
```

| 字段 | 含义 |
|---|---|
| `elapsed_seconds` | 当前采样已用墙钟时间 |
| `step_seconds` | 最近一步与上一步 callback 之间的时间 |
| `seconds_per_step` | 从采样开始计算的平均每步耗时 |
| `steps_per_second` | 平均采样速度；无有效耗时时可能为 `null` |
| `eta_seconds` | 按当前平均速度估算的剩余采样时间 |

这些指标只描述 sampling，不包含首次模型加载、提示词编码、VAE 解码、PNG/MP4 保存和首次
资源 SHA-256 计算。启用预览后，已经发生的预览编码/传输开销会体现在后续步骤的速度中。

Worker 的 `prompt` 阶段包含文本 tokenizer/conditioning 构建。连续任务在文本编码器、模式
和提示词均未变化时会复用纯文本 conditioning，避免重复执行完整文本编码；因此该阶段事件
仍会发送，但阶段耗时可能接近于零。带首帧或参考图的视频 conditioning 依赖图像内容，
必须重新构建，不与纯文生视频缓存混用。

## 3. 预览事件

启用后，每个 `step_progress` 后发送同一步的 `preview_image`：

```json
{
  "type": "preview_image",
  "job_id": "job-uuid",
  "step": 3,
  "total": 8,
  "mime_type": "image/jpeg",
  "encoding": "base64",
  "width": 72,
  "height": 72,
  "data": "/9j/4AAQ..."
}
```

预览使用 ComfyUI 模型 latent format 的 Latent2RGB 映射，不运行正式 VAE。它适合观察
构图和颜色趋势，但不代表最终清晰度。FastAPI 可以直接转发 JSON；若带宽或 CPU 成为
瓶颈，应将 `data` 解码后通过 WebSocket 二进制帧发送，并在伴随消息中保留 job/step。

预览属于瞬时状态，不写 SQLite、不重放，也不要记录 base64 正文。

## 4. PNG 元数据

新生成的 PNG 在 `diffusion_workbench` iTXt 块中保存 UTF-8 JSON。顶层结构为：

```json
{
  "schema_version": 4,
  "generator": {"name": "diffusion-workbench", "version": "0.1.0"},
  "job": {"id": "job-uuid", "batch_id": null},
  "parameters": {
    "mode": "krea2",
    "prompt": "portrait",
    "negative_prompt": "",
    "width": 576,
    "height": 576,
    "steps": 8,
    "seed": 42,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "denoise": 1.0,
    "negative_conditioning": "positive_reused"
  },
  "resources": {
    "diffusion_model": {
      "filename": "model.safetensors",
      "path": "D:\\models\\model.safetensors",
      "size_bytes": 123456,
      "modified_ns": 1780000000000000000,
      "sha256": "full-64-character-sha256"
    },
    "vae": {},
    "text_encoder": {},
    "clip_type": "krea2",
    "model_loader": "components"
  },
  "runtime": {
    "comfyui": "0.28.0",
    "python": "3.12.11",
    "pytorch": "2.x",
    "cuda": "12.8"
  },
  "performance": {
    "load_seconds": 10.0,
    "sampling_seconds": 6.0,
    "vae_seconds": 1.0,
    "generation_seconds": 7.0,
    "cuda_peak_allocated_gib": 12.0,
    "cuda_peak_reserved_gib": 14.0
  }
}
```

随机 seed 在任务开始时解析，因此 PNG 保存的是实际 seed，不是 `-1`。外置资源都包含
完整 SHA-256；Worker 按绝对路径、大小和纳秒修改时间缓存哈希。同一 Worker 内未变化的
资源不会重复哈希，首次使用或文件被替换后会完整读取一次。SDXL 的
`model_loader` 为 `checkpoint`，`vae`/`text_encoder` 为 `null`，checkpoint 的完整哈希
记录在 `diffusion_model`。

`negative_conditioning` 取值为 `encoded_negative_prompt`(mode 为 `zib`，或 CFG ≠ 1
时 Core 实际编码了负面提示词）或 `positive_reused`（复用正面条件）。读取仍兼容
schema v1-v3 的旧 PNG 仍可读取；v1 没有 `negative_prompt`/`negative_conditioning` 字段，消费方
应按缺失处理（空字符串 / `null`)。

旧 PNG 不会自动补写元数据。图片编辑器、聊天软件或图床可能删除 PNG 文本块，SQLite
仍是本机历史记录的最终事实来源。

## 5. 视频任务

视频任务不发送 latent 图片预览。Worker 会发送 `video_encoding` 与
`video_saved` 阶段事件，完成事件 `job_finished` 带 `artifact_type: "video"`，输出为
H.264 MP4，路径格式为 `output/YYYY-MM-DD/<video_model>-NNNNN.mp4`。MP4 使用 ComfyUI
Video API 写入 `diffusion_workbench` 容器 metadata，值为 JSON，包含模型、提示词、尺寸、
时长、帧率、length、采样参数、latent multiplier、输入图片/参考图片路径、资源指纹、运行时和性能字段。输入图片和参考图片的
服务端路径不能由 HTTP 客户端直接指定，应由受控资产引用解析。

MiniMax H3 的 metadata 还包含 `generation_type`（`t2v`/`i2v`/`r2v`）和固定内部参数 `audio_shift`；输出在 H.264 视频流之外包含
32 kHz 双声道 AAC 音轨。H3 的 `duration_seconds` 保留用户请求值，实际媒体时长由向上
对齐后的 `length / 24` 决定，因此 5 秒请求会生成 124 帧，约 5.17 秒。

`runtime_status()` 的 `loaded_resources` 显示当前 workload、model、vae、audio_vae（H3）、
text_encoder 和 clip_type；`release_resources()` 在无运行任务且队列为空时卸载 Worker 资源。图片与视频
workload 切换时 Worker 会自动执行同样的释放。

## 6. 读取与恢复

代码读取：

```python
from diffusion_workbench_core import read_generation_metadata

metadata = read_generation_metadata("output.png")
if metadata is None:
    raise ValueError("图片不包含 diffusion-workbench 元数据")
```

技术验证 demo：

```powershell
uv run python -m demo.demo_png_metadata output.png
```

恢复任务时必须用资源 SHA-256 在服务端资源目录中匹配 checkpoint，再构造
`GenerationSettings`。PNG 来自不可信输入时，不能直接打开其中记录的绝对路径，也不能
绕过 `ResourceCatalog` 的目录边界。`schema_version` 不受支持、JSON 损坏或资源哈希无法
匹配时，应明确拒绝自动恢复并让用户重新选择资源。

## 7. ComfyUI 生成 PNG 的参数展示

相册对不含本应用 iTXt 块的 PNG 回退解析 ComfyUI 原生 `prompt` tEXt 块（API prompt
节点图），由 `diffusion_workbench_core.png_metadata.read_comfyui_metadata(path)`
实现，只用于展示，不参与任务恢复：

- `model_name`/`vae_name`/`text_encoder_name`：文档序第一个 UNETLoader/VAELoader/
  CLIPLoader 的文件名（仅 basename，去掉子目录）。
- `mode`：按 unet 文件名小写 best-effort 推断，匹配顺序 krea2 → zib → sdxl → zit，
  匹配不到为 null。
- 采样器/调度器/步数/cfg/seed：文档序第一个 KSampler 或 KSamplerAdvanced
  （后者的种子字段为 `noise_seed`）；多轮采样也只取第一个。输入为链接而非字面量时
  取不到，记 null。
- 正/负提示词：该采样器 positive/negative 链接指向的 CLIPTextEncode 的 `text`；
  链接到 ConditioningZeroOut 时透传一层其 `conditioning` 输入。

返回 dict 带 `source: "comfyui"`；非 ComfyUI PNG（无 `prompt` 块或格式不符）返回
None。API 形状见 `AGENTS/api/rest.md` 相册 metadata 小节，前端在文件抽屉顶部展示
「由 ComfyUI 生成」。
