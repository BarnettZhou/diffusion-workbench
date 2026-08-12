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

默认配置使用本机 ComfyUI 的 Python。拆分模型的模型目录、VAE 目录和固定 text
encoder 都在 `configs/workbench.yaml` 中设置；SDXL 则配置完整 checkpoint 文件。

## 命令

```text
/mode [zit|krea2|zib|sdxl]
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
/upscale on|off
/upscale method list
/upscale method set <resize|upscale_model|latent_hires|index>
/upscale scale <factor>
/upscale interpolation list
/upscale interpolation set <name|index>
/upscale model list
/upscale model set <name|index>
/upscale tile <size>
/upscale overlap <size>
/upscale steps <1-100>
/upscale start-step <0..steps-1>
/upscale cfg <positive-number|inherit>
/upscale sampler list
/upscale sampler set <name|index|inherit>
/upscale scheduler list
/upscale scheduler set <name|index|inherit>
/upscale seed <inherit|非负整数>
/upscale status
/upscale reset
/start [num]
/status
/skip
/stop
/exit
```

默认使用 Euler + simple、CFG 1。`/sampler list` 和 `/scheduler list` 显示 Core
开放的 ComfyUI 选项；Worker 执行任务前还会确认当前安装的 ComfyUI 是否支持所选名称。
默认模式为 ZIT，默认尺寸为 576×576、8 步、随机 seed。模型和 VAE 默认不选择；
SDXL 的 VAE 内嵌在 checkpoint 中，只需选择模型。
`/start` 省略 `num` 时默认提交 1 个任务。

## 视频生成

视频功能与图片参数独立，当前支持 `wan2.2-ti2v-5b`、`wan2.2-i2v-14b` 和
`minimax-h3`：

```text
/video type list|set <video-model>
/video model list|set <index>
/video vae list|set <index>
/video prompt <prompt>
/video negative <negative-prompt>
/video size <width>*<height>
/video duration <seconds>
/video fps <fps>
/video steps <1-100>
/video seed <-1|非负整数>
/video cfg <positive-number>
/video shift <0-100>
/video latent-multiplier <positive-number>
/video sampler list|set <name|index>
/video scheduler list|set <name|index>
/video image set <path>|clear
/video status
/video start [num]
/resources status|release
```

默认视频为 704x960、5 秒、24 FPS（121 帧）、20 步、CFG 5、shift 8、latent multiplier 1、`simple` 调度器，denoise 固定为 1。TI2V-5B 默认 `uni_pc`，切换到 I2V-14B 时默认使用官方工作流的 `euler`。shift 可设置为 0 到 100，用于 ComfyUI 的 `ModelSamplingSD3`；latent multiplier 为采样前的 latent 缩放系数，Turbo 模型可设为 0.8。模型和 VAE 均从配置目录扫描并需要选择。TI2V-5B 设置图片是 I2V，不设置图片是 T2V；I2V-14B 必须设置图片，并自动配对 diffusion 目录中的同格式 high/low 两个模型。视频 diffusion 也可选择 `.gguf`，但需要 ComfyUI 安装 `ComfyUI-GGUF`；视频 VAE 仍使用 safetensors。视频写入对应模型前缀的 `output/YYYY-MM-DD/` 目录。`/resources release` 只允许在队列空闲时执行。

切换到 `minimax-h3` 时自动采用 608x352、5 秒（向上对齐为 124 帧）、24 FPS、8 步、
CFG 1、shift 12 和 `res_multistep + simple`。H3 宽高必须为 32 的倍数，FPS 固定 24，
CFG 固定 1；H3 宽高单边为 32–1344，面积不超过 768×1344；FL2VA 无图片时为 T2V，单张图片作为首帧时为 I2V；选择文件名含 `ref2va`
的模型后，`/video reference set <path>` 设置一张参考图即为 R2V。第一阶段不支持多图、
参考视频或参考音频，参考图不能与首帧同时设置。
音频 VAE 从配置固定注入，输出 MP4 包含 H.264 视频与 32 kHz 双声道 AAC 音轨。

`/resources status` 和 `/status` 会显示系统 RAM、显存容量占用及 GPU 核心利用率；资源探测尚未完成或不可用时显示 `不可用`。

## 图片放大

放大默认关闭。`/upscale on` 开启后，每个任务先保存原图，再保存文件名带
`-upscale` 后缀的放大图。`/upscale off` 只关闭开关，不清除已经设置的参数；
`/upscale reset` 恢复默认值并关闭放大。

可用方法：

- `resize`：普通图片插值，资源占用最低，不加载额外模型。
- `upscale_model`：Real-ESRGAN、ESRGAN、SwinIR 等像素超分模型；模型从
  `workbench.yaml` 的全局 `upscaling.models` 目录读取。
- `latent_hires`：放大首次采样得到的 latent，再用当前 diffusion 模型二次采样。

`resize` 和 `upscale_model` 可用 interpolation 为 `nearest-exact`、`bilinear`、
`area`、`bicubic`、`lanczos`；`latent_hires` 可用 `nearest-exact`、`bilinear`、
`area`、`bicubic`、`bislerp`。

模型超分的 `tile` 默认 512，范围 128～1024 且必须是 32 的倍数；`overlap` 默认 32，
必须小于 tile 的一半。执行时如果显存不足，Worker 会自动把 tile 逐次减半，最低 128。
超分模型执行后移回 CPU，后续任务可复用。

Latent 二次采样直接使用 `steps` 与 `start-step`。例如：

```text
/upscale steps 9
/upscale start-step 4
```

表示总采样时间表为 9 步，跳过前 4 步，从人类计数的第 5 步开始，实际执行 5 步。
CFG、sampler、scheduler 和 seed 使用 `inherit` 时继承首次采样的实际值。

`/upscale status` 会显示开关、方法、预计尺寸，以及 latent 方法的实际执行步数。
`/start` 提交的是完整不可变快照；提交后继续修改放大参数不会改变已排队任务。

TUI 底部状态栏按真实执行事件显示“启动推理 Worker”“加载模型资源”“编码提示词”
“准备 latent”“采样中”“VAE 解码”“保存图片”“图片已保存”等阶段。采样步数始终
单独显示：进入采样前为“未开始/n”，采样时实时更新为“1/n”，采样完成后保持
“n/n”，采样时还会显示实时速度和预计剩余时间。状态栏同时显示队列、Worker 和 GPU
状态。生成失败时保持命令输入可用；
详细错误同时写入对应任务的 SQLite 记录。

任务按提交时的设置快照依次执行。Worker 在 TUI 存活期间保留已加载资源；同模式
切换 diffusion 时复用 text encoder 和 VAE；同一 SDXL checkpoint 会整体复用；跨模式时
先释放旧模式资源。`/skip`
只终止当前任务，等待队列不变；若仍有任务，Core 会重启 Worker 并继续下一项，若没有
等待任务则进入空闲状态。任务尚在准备或已经进入完成落库阶段时不会误报跳过；Worker
取消失败会直接显示错误。`/stop` 会终止当前 Worker 并清空队列，下一次 `/start` 会创建
干净的 Worker；`/exit` 释放全部资源。

输出写入 `output/YYYY-MM-DD/<mode>-NNNNN.png`。开启放大时还会写入
`output/YYYY-MM-DD/<mode>-NNNNN-upscale.png`。每张 PNG 的
`diffusion_workbench` iTXt 块包含实际 seed、prompt、尺寸、采样参数、资源路径/SHA-256
和运行时版本；可通过 `uv run python -m demo.demo_png_metadata <image.png>` 验证读取。任务和别名仍会
存入 `.cache/diffusion_workbench.sqlite3`。

同一个 SQLite 数据库同一时间只允许一个 core 实例持有；TUI 与未来 HTTP 接口应共享
这个实例和它的串行队列，避免两个 Worker 同时占用 GPU。
