# diffusion-workbench CLI

`diffusion-workbench` 是 Textual TUI 调用端，推理、任务队列、资源目录和 SQLite
持久化位于独立的 `diffusion_workbench_core` 包中。后续 HTTP 服务应直接复用
`WorkbenchCore`，不要绕过 core 启动第二套 GPU Worker。

预览事件、速度字段和 PNG 内嵌参数的完整契约见
[生成预览、速度与 PNG 元数据](generation-preview-speed-and-metadata.md)。

## 启动

```powershell
uv sync
uv run diffusion-workbench --config .\configs\workbench.yaml
```

激活 `.venv` 后也可以直接运行 `diffusion-workbench`。

默认配置使用本机 ComfyUI 的 Python。模型目录、VAE 目录和每种模式固定的 text
encoder 都在 `configs/workbench.yaml` 中设置；`diffusion` 与 `vae` 都接受多个目录。

## 命令

```text
/mode [zit|krea2|zib]
/model list
/model set <index>
/model set-alias <index> <alias-name>
/vae list
/vae set <index>
/vae set-alias <index> <alias-name>
/prompt <prompt>
/negative <negative-prompt>
/size <width>*<height>
/steps <1-100>
/seed <-1|非负整数>
/cfg <positive-number>
/sampler list
/sampler set <name|index>
/scheduler list
/scheduler set <name|index>
/start [num]
/status
/skip
/stop
/exit
```

默认使用 Euler + simple、CFG 1。`/sampler list` 和 `/scheduler list` 显示 Core
开放的 ComfyUI 选项；Worker 执行任务前还会确认当前安装的 ComfyUI 是否支持所选名称。
默认模式为 ZIT，默认尺寸为 576×576、8 步、随机 seed。模型和 VAE 默认不选择。
`/start` 省略 `num` 时默认提交 1 个任务。

TUI 底部状态栏按真实执行事件显示“启动推理 Worker”“加载模型资源”“编码提示词”
“准备 latent”“采样中”“VAE 解码”“保存图片”“图片已保存”等阶段。采样步数始终
单独显示：进入采样前为“未开始/n”，采样时实时更新为“1/n”，采样完成后保持
“n/n”，采样时还会显示实时速度和预计剩余时间。状态栏同时显示队列、Worker 和 GPU
状态。生成失败时保持命令输入可用；
详细错误同时写入对应任务的 SQLite 记录。

任务按提交时的设置快照依次执行。Worker 在 TUI 存活期间保留已加载资源；同模式
切换 diffusion 时复用 text encoder 和 VAE，跨模式时先释放旧模式资源。`/skip`
只终止当前任务，等待队列不变；若仍有任务，Core 会重启 Worker 并继续下一项，若没有
等待任务则进入空闲状态。任务尚在准备或已经进入完成落库阶段时不会误报跳过；Worker
取消失败会直接显示错误。`/stop` 会终止当前 Worker 并清空队列，下一次 `/start` 会创建
干净的 Worker；`/exit` 释放全部资源。

输出写入 `output/YYYY-MM-DD/<mode>-NNNNN.png`。每张 PNG 的
`diffusion_workbench` iTXt 块包含实际 seed、prompt、尺寸、采样参数、资源路径/SHA-256
和运行时版本；可通过 `uv run python -m demo.demo_png_metadata <image.png>` 验证读取。任务和别名仍会
存入 `.cache/diffusion_workbench.sqlite3`。

同一个 SQLite 数据库同一时间只允许一个 core 实例持有；TUI 与未来 HTTP 接口应共享
这个实例和它的串行队列，避免两个 Worker 同时占用 GPU。
