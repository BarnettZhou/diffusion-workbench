# Wan 视频前端交接说明

本次变更已在 Core 和 TUI 中加入 Wan 2.2 TI2V-5B 的文生视频（T2V）与图生视频（I2V）能力。HTTP API 和 frontend 尚未修改，后续实现应复用现有 `WorkbenchCore`，不要自行启动 ComfyUI Worker 或绕过 Core 队列。

## 配置与模型

`WorkbenchConfig.video_resources` 按 `VideoModel` 分类。当前唯一支持值为 `wan2.2-ti2v-5b`，每个类型包含一个或多个 `diffusion` 目录、一个或多个 `vae` 目录、固定 `text_encoder` 文件和固定 `clip_type: wan`。客户端只能使用 Core 返回的资源 index/受控资源引用，不应提交模型或 VAE 路径。

## Core 调用

```python
core.list_video_models(VideoModel.WAN22_TI2V_5B)
core.list_video_vaes(VideoModel.WAN22_TI2V_5B)
core.submit_video(VideoGenerationSettings(...), count)
core.get_video_job(job_id)
core.runtime_status()
core.release_resources()
```

`VideoGenerationSettings` 默认值为 704x960、5 秒、24 FPS、20 步、CFG 5、shift 8、`uni_pc` + `simple`、seed -1、denoise 1。`shift` 范围为 0 到 100，用于 ComfyUI 的 `ModelSamplingSD3`。`length` 是 `duration_seconds * fps + 1`，宽高必须是 16 的倍数，Wan latent 还要求 `length = 4n + 1`。设置 `input_image` 时为 I2V，不设置时为 T2V。输入图片应先通过受控上传/资产接口落盘，再将服务端生成的资产 ID 映射为路径；不能让 HTTP 客户端直接提交任意服务器文件路径。

## 任务与输出

视频任务是独立的 `VideoJobRecord`，包含 `video_model`、`generation_type`、输入图片、时长、FPS、帧数和 denoise。输出为 H.264 MP4：`output/YYYY-MM-DD/wan2.2-ti2v-5b-00001.mp4`。任务事件 `job_finished` 带 `artifact_type: "video"`。视频阶段名为 `video_encoding` 和 `video_saved`；当前没有 latent 预览。

## 资源状态与释放

`runtime_status()` 返回 `loaded_resources`（workload、model、vae、text_encoder、clip_type）和 GPU 占用字符串。`release_resources()` 只能在没有运行任务且队列为空时调用；切换图片/视频工作负载时 Worker 也会自动释放旧资源。HTTP 层应把繁忙时的 `RuntimeError` 转成明确的冲突响应。

## TUI 参考

视频命令组包括 `/video type`、`/video model`、`/video vae`、`/video prompt`、`/video image set|clear`、`/video size`、`/video duration`、`/video fps`、`/video steps`、`/video seed`、`/video cfg`、`/video shift`、`/video sampler`、`/video scheduler`、`/video status`、`/video start`。资源诊断命令为 `/resources status` 和 `/resources release`。
