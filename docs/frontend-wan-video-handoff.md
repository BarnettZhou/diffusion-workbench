# Wan 视频前端交接说明

Core、TUI、HTTP API 和 frontend 已接入 Wan 2.2 TI2V-5B 与 I2V-14B FP8。所有视频任务
复用现有 `WorkbenchCore`，不会自行启动 ComfyUI Worker 或绕过 Core 队列。

## 配置与模型

`WorkbenchConfig.video_resources` 按 `VideoModel` 分类。当前支持 `wan2.2-ti2v-5b` 和
`wan2.2-i2v-14b`。每个类型包含一个或多个 `diffusion` 目录、一个或多个 `vae` 目录、
固定 `text_encoder` 文件和固定 `clip_type: wan`。本次 14B 冒烟使用 safetensors 版 UMT5；
GGUF 文本编码器是否可用取决于当前 ComfyUI 的 loader 支持，frontend 不应自行切换 loader。
14B I2V 的 diffusion 目录必须同时
包含名称带 `_high_noise_` 与 `_low_noise_` 的两个模型；Core 会根据所选任一文件自动
解析配对模型。客户端只能使用 Core 返回的资源 index/受控资源引用，不应提交模型或 VAE 路径。

`GET /api/v1/video/models` 会为每个已配置模型返回 `label`、`generation_types` 和
`requires_input_image`。frontend 用这些字段渲染模型名称和输入图片约束；旧 API 响应
缺少能力字段时，仍会按 `wan2.2-i2v-14b` 必须提供图片的规则保守兜底。14B 未上传输入
图片时前端阻止提交，API 也会返回 422 作为最终校验。

## Core 调用

```python
core.list_video_models(VideoModel.WAN22_TI2V_5B)
core.list_video_vaes(VideoModel.WAN22_TI2V_5B)
core.submit_video(VideoGenerationSettings(...), count)
core.get_video_job(job_id)
core.runtime_status()
core.release_resources()
```

`VideoGenerationSettings` 默认值为 704x960、5 秒、24 FPS、20 步、CFG 5、shift 8、`uni_pc` + `simple`、seed -1、denoise 1。`shift` 范围为 0 到 100，用于 ComfyUI 的 `ModelSamplingSD3`。`length` 是 `duration_seconds * fps + 1`，宽高必须是 16 的倍数，Wan latent 还要求 `length = 4n + 1`。5B 设置 `input_image` 时为 I2V，不设置时为 T2V；14B 类型是 I2V，必须设置 `input_image`。14B 使用 `WanImageToVideo` 节点和 Wan 2.1 VAE，采样时先用 high-noise 模型、再用 low-noise 模型完成两个阶段。输入图片应先通过受控上传/资产接口落盘，再将服务端生成的资产 ID 映射为路径；不能让 HTTP 客户端直接提交任意服务器文件路径。

## 任务与输出

视频任务是独立的 `VideoJobRecord`，包含 `video_model`、`generation_type`、输入图片、时长、FPS、帧数和 denoise。输出为 H.264 MP4，文件名前缀与模型类型一致，例如 `output/YYYY-MM-DD/wan2.2-ti2v-5b-00001.mp4` 或 `output/YYYY-MM-DD/wan2.2-i2v-14b-00001.mp4`。任务事件 `job_finished` 带 `artifact_type: "video"`。视频阶段名为 `video_encoding` 和 `video_saved`；当前没有 latent 预览。

## 资源状态与释放

`runtime_status()` 返回 `loaded_resources`（workload、model、vae、text_encoder、clip_type）以及资源占用信息：

```python
{
    "gpu": "12.34/15.92 GiB",  # 兼容字段，显存已用/总量
    "memory": {
        "used_gib": 18.42,
        "total_gib": 31.85,
        "percent": 57.8,
    },
    "gpu_memory_used_gib": 12.34,
    "gpu_memory_total_gib": 15.92,
    "gpu_memory_percent": 77.5,
    "gpu_utilization_percent": 92.0,
}
```

`memory` 是系统 RAM；`gpu_memory_*` 是显存容量占用；`gpu_utilization_percent` 是 GPU
核心利用率，不是显存占用。探测由后台线程执行，最多约 2 秒刷新一次，不阻塞状态调用。
探测尚未完成或系统没有可用 `nvidia-smi` 时，旧的 `gpu` 字段为 `查询中`/`不可用`，数值
字段为 `None`。HTTP 对应 `GET /api/v1/status`，TUI 可通过 `/status` 或 `/resources status`
查看这些指标。

`release_resources()` 只能在没有运行任务且队列为空时调用；切换图片/视频工作负载时 Worker
也会自动释放旧资源。HTTP 层应把繁忙时的 `RuntimeError` 转成明确的冲突响应。

## TUI 参考

视频命令组包括 `/video type`、`/video model`、`/video vae`、`/video prompt`、`/video image set|clear`、`/video size`、`/video duration`、`/video fps`、`/video steps`、`/video seed`、`/video cfg`、`/video shift`、`/video sampler`、`/video scheduler`、`/video status`、`/video start`。资源诊断命令为 `/resources status` 和 `/resources release`。
