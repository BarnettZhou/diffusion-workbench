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

这些指标只描述 sampling，不包含首次模型加载、提示词编码、VAE 解码、PNG 保存和首次
资源 SHA-256 计算。启用预览后，已经发生的预览编码/传输开销会体现在后续步骤的速度中。

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
  "schema_version": 1,
  "generator": {"name": "diffusion-workbench", "version": "0.1.0"},
  "job": {"id": "job-uuid", "batch_id": null},
  "parameters": {
    "mode": "krea2",
    "prompt": "portrait",
    "width": 576,
    "height": 576,
    "steps": 8,
    "seed": 42,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "denoise": 1.0,
    "negative_conditioning": "zeroed_positive"
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
    "clip_type": "krea2"
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

随机 seed 在任务开始时解析，因此 PNG 保存的是实际 seed，不是 `-1`。三类资源都包含
完整 SHA-256；Worker 按绝对路径、大小和纳秒修改时间缓存哈希。同一 Worker 内未变化的
资源不会重复哈希，首次使用或文件被替换后会完整读取一次。

旧 PNG 不会自动补写元数据。图片编辑器、聊天软件或图床可能删除 PNG 文本块，SQLite
仍是本机历史记录的最终事实来源。

## 5. 读取与恢复

代码读取：

```python
from diffusion_workbench_core import read_generation_metadata

metadata = read_generation_metadata("output.png")
if metadata is None:
    raise ValueError("图片不包含 diffusion-workbench 元数据")
```

技术验证 demo：

```powershell
uv run python demo_png_metadata.py output.png
```

恢复任务时必须用资源 SHA-256 在服务端资源目录中匹配 checkpoint，再构造
`GenerationSettings`。PNG 来自不可信输入时，不能直接打开其中记录的绝对路径，也不能
绕过 `ResourceCatalog` 的目录边界。`schema_version` 不受支持、JSON 损坏或资源哈希无法
匹配时，应明确拒绝自动恢复并让用户重新选择资源。
