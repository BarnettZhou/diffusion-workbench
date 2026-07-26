# diffusion-workbench

基于 ComfyUI Core 的本地 ZIT / Krea2 文生图工作台。项目将可复用的推理、资源、
串行队列和 SQLite 持久化放在 `diffusion_workbench_core`，Textual TUI 位于
`diffusion_workbench`。

```powershell
uv sync
uv run diffusion-workbench --config .\configs\workbench.yaml
```

命令、配置和资源生命周期说明见
[docs/workbench-cli.md](docs/workbench-cli.md)。
