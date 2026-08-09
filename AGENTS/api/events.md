# WebSocket 事件流

`WS /api/v1/events` 推送 Core 的实时公共事件。服务端只有一个 Core event sink，由
EventHub 向所有连接扇出；每个事件追加单调递增的 `event_id`，客户端可据此丢弃乱序
事件。

## 连接语义

- 事件**不持久化、不重放**。正确姿势:先连接订阅，再拉取 `GET /api/v1/status` 和
  `GET /api/v1/jobs/{id}` 快照;断线重连后重新拉快照，不能仅凭事件重建事实。
- 服务端事件队列有上限；慢客户端会丢失最旧事件（含预览），但绝不影响生成。
- 与 Core 原始事件相比,`output_path` 被改写为 `image_url`,`error` 只保留首行摘要,
  绝不包含绝对路径或 traceback。

## 事件类型

### `queue_progress`

队列快照。`sequence` 只对队列快照单调递增，用于丢弃旧快照。

```json
{ "type": "queue_progress", "queued": 2, "running": "job-uuid-or-null", "sequence": 17, "event_id": 42 }
```

### `job_started`

任务开始执行。随机 seed 在此公布实际值。

```json
{ "type": "job_started", "job_id": "job-uuid", "seed": 6339795016295235492, "event_id": 43 }
```

### `stage_progress`

执行阶段。stage 顺序:`starting_worker → loading_model → prompt → latent →
sampling → vae → saving → saved`。

```json
{ "type": "stage_progress", "job_id": "job-uuid", "stage": "vae", "total": 8, "event_id": 44 }
```

### `step_progress`

采样步进度,step 从 1 开始，并携带采样性能指标（只描述 sampling，不含模型加载和
VAE 解码):

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
  "eta_seconds": 10.0,
  "event_id": 45
}
```

`steps_per_second` 在无有效耗时时可能为 `null`。进入 sampling 但尚无 step 事件时，
客户端可显示 `0/n`。

### `preview_image`

服务端已启用预览(`set_preview_enabled(True)`)，每个 `step_progress` 后跟随同一步
的 latent 预览(base64 JPEG)。预览只反映构图和颜色趋势，不代表最终清晰度。

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
  "data": "/9j/4AAQ...",
  "event_id": 46
}
```

预览是瞬时状态：不入库、不重放，慢客户端会被丢弃。带宽成为瓶颈时可在后续版本改
为二进制帧，契约见
[../generation-preview-speed-and-metadata.md](../generation-preview-speed-and-metadata.md)。

### `job_error`

任务失败。`error` 只有首行摘要；完整 traceback 在服务端 SQLite 的 job 记录中。
`job_id` 可能为 `null`（与具体任务无关的错误，如 Worker 关闭失败）。

```json
{ "type": "job_error", "job_id": "job-uuid", "error": "RuntimeError: ...", "event_id": 47 }
```

### `job_finished`

任务到达终态。`status` ∈ `completed|failed|cancelled`;`image_url` 指向原图下载
端点；任务启用了图片放大且最终完成时，额外携带 `upscaled_image_url` 指向放大图
端点。放大阶段失败或取消时不带 `upscaled_image_url`，但原图可能已保存，
以 `GET /api/v1/jobs/{job_id}` 的持久化状态为准。

```json
{
  "type": "job_finished",
  "job_id": "job-uuid",
  "status": "completed",
  "image_url": "/api/v1/images/job-uuid",
  "upscaled_image_url": "/api/v1/images/job-uuid/upscaled",
  "event_id": 48
}
```

注意:stop 清空的 queued 任务不一定都有逐条事件，以 SQLite 状态为准。
