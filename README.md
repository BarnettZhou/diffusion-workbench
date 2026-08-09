# diffusion-workbench

基于 ComfyUI Core 的本地 ZIT / Krea2 / ZIB / SDXL 文生图工作台。项目将可复用的推理、资源、
串行队列和 SQLite 持久化放在 `diffusion_workbench_core`，Textual TUI 位于
`diffusion_workbench`，FastAPI HTTP 服务位于 `diffusion_workbench_api`。

```powershell
uv sync
copy .\configs\workbench.example.yaml .\configs\workbench.yaml  # 首次使用,改成本机路径
uv run diffusion-workbench --config .\configs\workbench.yaml
```

HTTP API 服务（单进程单 worker，与 TUI 互斥）:

```powershell
uv run uvicorn diffusion_workbench_api.app:app --host 127.0.0.1 --port 8188 --workers 1
```

或用一键脚本（自动按需构建前端、单端口托管页面与 API):

```powershell
python start.py [--port 8188]
```

读取生成 PNG 内嵌的完整参数：

```powershell
uv run python -m demo.demo_png_metadata .\output\2026-07-27\krea2-00001.png
```

命令、配置和资源生命周期说明见
[AGENTS/workbench-cli.md](AGENTS/workbench-cli.md)。

开发与接入文档：

- [HTTP API 文档](AGENTS/api/README.md)(REST 参考、WebSocket 事件流)
- [Core 技术参考](AGENTS/core-technical-reference.md)
- [FastAPI 后端接入指南](AGENTS/backend-integration.md)
- [生成预览、速度与 PNG 元数据](AGENTS/generation-preview-speed-and-metadata.md)
