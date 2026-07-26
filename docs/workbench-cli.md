# diffusion-workbench CLI

`diffusion-workbench` 是 Textual TUI 调用端，推理、任务队列、资源目录和 SQLite
持久化位于独立的 `diffusion_workbench_core` 包中。后续 HTTP 服务应直接复用
`WorkbenchCore`，不要绕过 core 启动第二套 GPU Worker。

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
/mode [zit|krea2]
/model list
/model set <index>
/model set-alias <index> <alias-name>
/vae list
/vae set <index>
/vae set-alias <index> <alias-name>
/prompt <prompt>
/size <width>*<height>
/steps <8-20>
/seed <-1|非负整数>
/start [num]
/status
/stop
/exit
```

当前固定为 Euler + simple、CFG 1。默认模式为 ZIT，默认尺寸为 576×576、8 步、
随机 seed。模型和 VAE 默认不选择。`/start` 省略 `num` 时默认提交 1 个任务。

TUI 底部状态栏按 Worker 的真实执行事件依次显示“加载模型”“处理 prompt”
“采样步数 1/n”“VAE 处理”“图片已保存”。生成失败时会显示错误状态并保持命令
输入可用；详细错误同时写入对应任务的 SQLite 记录。

任务按提交时的设置快照依次执行。Worker 在 TUI 存活期间保留已加载资源；同模式
切换 diffusion 时复用 text encoder 和 VAE，跨模式时先释放旧模式资源。`/stop`
会终止当前 Worker 并清空队列，下一次 `/start` 会创建干净的 Worker；`/exit` 释放
全部资源。

输出写入 `output/YYYY-MM-DD/<mode>-NNNNN.png`。任务和别名存入
`.cache/diffusion_workbench.sqlite3`。

同一个 SQLite 数据库同一时间只允许一个 core 实例持有；TUI 与未来 HTTP 接口应共享
这个实例和它的串行队列，避免两个 Worker 同时占用 GPU。
