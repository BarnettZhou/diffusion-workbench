# diffusion-workbench

基于 ComfyUI Core 的本地 ZIT / Krea2 文生图工作台。项目将可复用的推理、资源、
串行队列和 SQLite 持久化放在 `diffusion_workbench_core`，Textual TUI 位于
`diffusion_workbench`。

```powershell
uv sync
uv run diffusion-workbench --config .\configs\workbench.yaml
```

读取生成 PNG 内嵌的完整参数：

```powershell
uv run python -m demo.demo_png_metadata .\output\2026-07-27\krea2-00001.png
```

命令、配置和资源生命周期说明见
[docs/workbench-cli.md](docs/workbench-cli.md)。

开发与接入文档：

- [Core 技术参考](docs/core-technical-reference.md)
- [FastAPI 后端接入指南](docs/backend-integration.md)
- [Krea2 非 ComfyUI 后端研究](docs/krea2-alternative-backends.md)
