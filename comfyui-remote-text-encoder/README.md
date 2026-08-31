# ComfyUI Remote Text Encoder

在 Mac 上使用 ComfyUI 源码版运行的轻量 HTTP 服务。默认 ComfyUI 根目录为
`~/comfyui`，只负责 CLIPTextEncode 和 Krea2EditGroundedEncode，不执行采样。

## 安装与启动

```bash
cd ~/diffusion-workbench/comfyui-remote-text-encoder
~/comfyui/.venv/bin/python server.py \
  --comfy-root ~/comfyui \
  --encoder-id qwen3vl4b.safetensors \
  --clip-path ~/models/qwen3vl4b.safetensors \
  --host 0.0.0.0 --port 50051
```

Windows 端 `remote_encoder.enabled=true` 后会按 encoder 文件名调用此服务。
