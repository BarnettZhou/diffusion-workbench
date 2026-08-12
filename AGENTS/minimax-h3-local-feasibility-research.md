# MiniMax H3 量化版本本机可行性研究

> 调研时间：2026-08-12
> 目标机器：NVIDIA GeForce RTX 5070 Ti（16,303 MiB，compute capability 12.0）、系统内存 31.84 GiB。
> 初次模型实测环境为 ComfyUI-aki-v1.6（ComfyUI commit `dec5d9450a5290bcf63430409ea41018e67f41c3`，v0.30.2）、PyTorch 2.7.0+cu128、comfy-kitchen 0.2.26；后续只读核对时 Core 已升级到 v0.31.1、comfy-kitchen 0.2.28，PyTorch 仍为 2.7.0+cu128。

## 结论先行

**MiniMax H3 在这台电脑上有条件可运行，建议先验证 FL2VA/T2V 或 I2V 的 INT4 ConvRot 扩散模型 + NVFP4/AWQ 文本编码器，使用约 5 秒、低于 768p 的短视频。** 本次已在本机完成模型对象加载、文本编码前向、VAE 识别和主模型完整搬入 GPU 验证，不只是根据文件大小推算。ComfyUI 原生支持 H3，官方教程要求 ComfyUI 0.30.0 或更新版本；当前 v0.31.1 已满足版本门槛。[ComfyUI H3 教程](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)

但这不是“16GB 显存可无条件跑满规格”的结论。H3 是 33B dense Omni-Transformer；文本编码器是完整 Qwen3-VL-32B，仅在推理中可省去约 13B AdaLN 分支。官方开源首版使用 full attention，稀疏 attention 尚未发布，因此长视频、较高分辨率和 R2V 参考输入都会显著增加显存、系统内存和耗时压力。[MiniMax 官方模型卡：架构与部署](https://huggingface.co/MiniMaxAI/MiniMax-H3#model-architecture)

本机最稳的候选是：

1. `minimax_h3_fl2va_pruned_int4_convrot.safetensors` + `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`：T2V/I2V/首尾帧 FL2VA。
2. `minimax_h3_ref2va_pruned_int4_convrot.safetensors` + 同一文本编码器：R2V，但参考图/视频/音频会扩大上下文和 VAE 开销。

两个 `minimaxH3INT8INT4_*` 文件在初次 v0.30.2 实测中均以 `KeyError: 'asym_w4a8_int8'` 失败；升级到 v0.31.1 后，stock loader 已补上该格式，两个文件均已重新实测加载成功。无需为此安装社区 W4A8 loader，其 README 仍标注 Windows 不支持。[ComfyUI-W4A8-Loader README](https://github.com/starsFriday/ComfyUI-W4A8-Loader/blob/main/README.zh-CN.md)

## 官方模型与工作流事实

MiniMax 官方说明 H3 是统一多模态系统：输入可组合文字、图像、视频、音频，输出视频和原生立体声；Base 模型默认 768p、24 FPS、4--15 秒，2K 由未开源的 Regenerate-2K 模块完成。H3 Base 有两套任务权重：FL2VA（无图/一张/两张首尾帧）和 Ref2VA（多图、视频、音频参考）。[MiniMax H3 README](https://huggingface.co/MiniMaxAI/MiniMax-H3#system-overview)

架构关键点：H3-Encoder 使用完整 Qwen3-VL-32B，并向 H3 Transformer 提供第 50 层 hidden states；视觉 VAE 为 `f16t4d24`（空间 16 倍、时间 4 倍、24 latent channels）；音频 VAE 为 32 kHz、每声道 40 Hz latent；Omni-Transformer 为 33B dense 单流 Transformer。[MiniMax H3 README](https://huggingface.co/MiniMaxAI/MiniMax-H3#model-architecture)

官方推荐本地部署框架包括 SGLang、vLLM、diffusers 和 ComfyUI；SGLang 示例使用 4 张 GPU，说明官方原始 BF16 部署并非面向单张 16GB 卡。[MiniMax H3 README：Local Deployment of H3-Base](https://huggingface.co/MiniMaxAI/MiniMax-H3#local-deployment-of-h3-base)

ComfyUI 官方教程明确：更新到 v0.30.0+，从 Template Library 的 Video 分类选择 H3 T2V/I2V/R2V；T2V/I2V 使用 `fl2va`，R2V 使用不同的 `ref2va` 权重。教程给出的官方组合是 INT8 ConvRot 扩散模型、NVFP4/AWQ Qwen3-VL 文本编码器，以及视频 FP16 VAE 和音频 FP32 VAE。[ComfyUI H3 教程](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)

ComfyUI 原生 H3 节点包括 `MiniMaxH3ImageToVideo`、`MiniMaxH3ReferenceToVideo` 和 `EmptyMiniMaxH3LatentAV`；默认 latent 长度为 124 帧（约 5 秒），长度按 `17k+5` 网格取整，训练范围约 124--362 帧，代码注明更长范围未经测试。[ComfyUI H3 节点源码](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)

H3 原生支持由 ComfyUI commit [`57500fc5`](https://github.com/Comfy-Org/ComfyUI/commit/57500fc5bc92566a63f2046824f522cd55c335ca) 引入。当前本机 v0.31.1 已包含其后的 latent noise mask 修复 [`563b98ee`](https://github.com/Comfy-Org/ComfyUI/commit/563b98eefbe643a4cd510ee7f0b43e79880d5a3f)，无需再为该修复单独更新 Core。

## 本机文件核对

| 文件 | 实际大小 | 识别到的格式/任务 | 判断 |
|---|---:|---|---|
| `minimax_h3_fl2va_pruned_int4_convrot.safetensors` | 10.56 GiB | 200 层 `convrot_w4a4` | **已实测加载成功**，识别为 `MiniMaxH3Model`；首选基线 |
| `minimax_h3_ref2va_pruned_int4_convrot.safetensors` | 10.56 GiB | 200 层 `convrot_w4a4` | **已实测加载成功**，识别为 `MiniMaxH3Model`；R2V 候选 |
| `minimaxH3INT8INT4_flREF2VAPruned.safetensors` | 11.68 GiB | `_quantization_metadata`，层格式 `asym_w4a8_int8` | **v0.31.1 已实测加载成功**，识别为 `MiniMaxH3Model` |
| `minimaxH3INT8INT4_flREF2VAPruned_rl2va.safetensors` | 10.96 GiB | 200 层 `asym_w4a8_int8` + 8 层 `int8_tensorwise` | **v0.31.1 已实测加载成功**，识别为 `MiniMaxH3Model` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.61 GiB | 350 层 `nvfp4` + 1 层 `int8_tensorwise` | **已实测 CPU 加载和前向成功**，识别为 `MiniMaxH3TEModel_` |
| `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` | 12.20 GiB | GGUF Q2_K_M，902 tensors，无通用架构 metadata | **当前 ComfyUI-GGUF 明确拒绝加载**；来源是 stable-diffusion.cpp 转换 |
| `minimax_h3_video_vae_fp16.safetensors` | 4.85 GiB | 官方视频 VAE | **已实测识别为 `MiniMaxH3VideoVAE`** |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.56 GiB | 官方音频 VAE | **已实测识别为 `MiniMaxH3AudioVAE`** |

以上大小由本机文件系统读取；模型文件来源和目录约定可在 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) 与官方教程中交叉核对。

## 量化格式与 Blackwell 兼容性

ComfyUI 0.30.2 的 `comfy.quant_ops` 注册了 `convrot_w4a4`、`int8_tensorwise`、`nvfp4`、FP8 等格式；本机 `ops.py` 还包含部分 W4A8/ConvRot 处理代码，但这不等于官方 stock loader 已稳定支持 `asym_w4a8_int8`。`supports_nvfp4_compute()` 对 NVIDIA compute capability major >= 10 返回支持，因此本机 capability 12.0 通过该门槛。[ComfyUI quant_ops.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/quant_ops.py)、[ComfyUI model_management.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py)

不过本机 PyTorch 是 cu128。当前 comfy-kitchen 会在 CUDA 版本低于 cu130 时禁用其优化 CUDA backend；实际启动诊断为 `cuda available=True, disabled=True`，同时保留 eager backend 的 NVFP4/ConvRot 解量化能力。因此“能加载/运行”和“获得优化 CUDA kernel 性能”是两件事：当前可以尝试 eager fallback，但不能据此承诺速度；若要使用官方/社区建议的优化 ConvRot kernel，应评估升级到 cu130+ 的 ComfyUI Python 环境，升级前需完整备份并做兼容性回归。[ComfyUI quant_ops.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/quant_ops.py)

Hugging Face 的 Comfy-Org 模型卡把 NVFP4 标为 Blackwell 选项，并说明 NVFP4 文本编码器不要求 Blackwell；同一模型卡建议能使用 cu130 时优先 INT8 ConvRot，FP8 scaled 仅作为不能使用 INT8 ConvRot 时的备选。[Comfy-Org/MiniMax-H3 README](https://huggingface.co/Comfy-Org/MiniMax-H3#minimax-h3)

GGUF 路径来自社区 `ComfyUI-GGUF`，不是 ComfyUI Core stock loader。该节点虽然提供 `CLIPLoaderGGUF` 且架构白名单包含 `qwen3vl`，但你现有的 Q2 文件只有 `GGUF.version`、`GGUF.tensor_count`、`GGUF.kv_count` 三个头字段，没有 `general.architecture` 等 llama.cpp metadata。本机调用 `gguf_clip_loader` 时明确报错 `This gguf file is incompatible with llama.cpp`。其来源仓库 `leejet/MiniMax-H3-GGUF` 也明确说明文件由 stable-diffusion.cpp 转换并用于 stable-diffusion.cpp，因此不要把它当作当前 ComfyUI 的低内存编码器方案。[ComfyUI-GGUF loader](https://github.com/city96/ComfyUI-GGUF/blob/main/nodes.py)、[leejet MiniMax-H3-GGUF README](https://huggingface.co/leejet/MiniMax-H3-GGUF)

## 本机运行验证数据

- FL2VA pruned INT4 ConvRot 完整搬入 RTX 5070 Ti 后，PyTorch 显存为约 10.63 GiB allocated、10.70 GiB reserved，模型 loaded size 约 10.56 GiB。16 GiB 显存可以容纳主模型，但必须把文本编码器留在 CPU，并让 VAE 按阶段换入换出。
- NVFP4/AWQ 文本编码器强制 CPU 加载后，对 9 token 的短提示词做完整前向成功，耗时约 56.9 秒；输出 `[1, 9, 5120]`，进程 RSS 峰值约 16.5 GiB，GPU 显存占用为 0。32 GiB RAM 可以承载，但当前回退路径较慢。
- 视频 VAE 模型大小约 4.85 GiB，音频 VAE 约 0.56 GiB；二者均完成对象构建验证。
- 测试时系统 pagefile 为 38 GiB，建议保留；模型与编码器位于 E 盘，当前约有 296.5 GiB 可用空间。

## ComfyUI 0.31.1 与 cu130 升级评估

### 当前状态与直接结论

后续只读核对确认，秋叶整合包中的 Core 已更新到 ComfyUI `v0.31.1`（commit `fe4195f7f4275f2626cbafc703acc3ddde1e5490`），Python 仍为 3.11.9，环境仍是 `torch 2.7.0+cu128`、`torchvision 0.22.0+cu128`、`torchaudio 2.7.0+cu128`、`xformers 0.0.30`、`comfy-kitchen 0.2.28`。GPU 驱动为 610.88，RTX 5070 Ti 被 PyTorch 正确识别为 compute capability 12.0。

**cu130 值得做一次可回滚的隔离试验，但不建议无备份地直接覆盖唯一的秋叶主环境。** 这不是为了让 H3 从“不能运行”变成“能运行”，而是为了让 ComfyUI 0.31.1 的 comfy-kitchen 启用针对 NVFP4、ConvRot 和 W4A8 的优化 CUDA backend。v0.31.1 源码明确在 `torch.version.cuda < 13` 时调用 `ck.registry.disable("cuda")`，并提示需要 cu130 或更高版本；cu128 下仍可走 eager fallback，只是量化矩阵运算可能慢很多。[ComfyUI v0.31.1 `quant_ops.py`](https://github.com/Comfy-Org/ComfyUI/blob/v0.31.1/comfy/quant_ops.py)

换成 cu130 不会减少模型文件、latent 或 VAE 本身占用的显存，也不能消除 16GB VRAM / 32GB RAM 的容量压力。它主要改变量化算子的执行后端和速度，因此应把成功标准设为“优化 backend 确实启用且完整 H3 生成显著改善”，不能只看 `torch.version.cuda` 字符串。

### PyTorch cu130、Windows 与 Blackwell

PyTorch cu130 已是正式发布渠道，不再只是 nightly。PyTorch 官方历史版本页为 Linux 和 Windows 同时给出 2.9.0、2.9.1、2.10.0、2.11.0 等 cu130 安装组合；官方 cu130 wheel 索引也存在 Python 3.11 的 Windows x64 wheels。本机 Python 3.11.9 因而有对应的正式包。[PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)、[PyTorch cu130 torch wheel index](https://download.pytorch.org/whl/cu130/torch/)

本机驱动 610.88 高于 NVIDIA 对 CUDA 13.0 GA 列出的 580.65.06 最低驱动线，因此驱动侧没有明显门槛。pip/启动器安装的 PyTorch wheel 自带所需 CUDA runtime；这不等同于、通常也不要求另装完整 CUDA Toolkit。[CUDA Toolkit Release Notes：driver requirements](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)

秋叶启动器本机缓存的环境维护清单显示，“Torch 2.9.1 (CUDA 13.0)”实际执行的是一组配套升级：`torch 2.9.1+cu130`、`torchvision 0.24.1+cu130`、`torchaudio 2.9.1+cu130`。这三个版本在 PyTorch 官方 cu130 索引中都有 Python 3.11 Windows wheel，组合本身合理。[torchvision cu130 index](https://download.pytorch.org/whl/cu130/torchvision/)、[torchaudio cu130 index](https://download.pytorch.org/whl/cu130/torchaudio/)

### ComfyUI 0.31.1 对 H3 的变化

0.31.1 的 `requirements.txt` 固定 `comfy-kitchen==0.2.28`，且源码已注册 `convrot_w4a4`、`nvfp4`、`int8_tensorwise` 和 `asym_w4a8_int8`。其中 `asym_w4a8_int8` 由 ComfyUI commit [`344b43989`](https://github.com/Comfy-Org/ComfyUI/commit/344b43989) 引入，并已包含在本机 v0.31.1。此前在 v0.30.2 上两个 INT8/INT4 文件触发的 `KeyError: 'asym_w4a8_int8'` 已消失，两个文件均完成模型对象加载；是否能在 16GB 显存上完成采样仍需另行实测。[ComfyUI v0.31.1 requirements](https://github.com/Comfy-Org/ComfyUI/blob/v0.31.1/requirements.txt)、[ComfyUI v0.31.1 `quant_ops.py`](https://github.com/Comfy-Org/ComfyUI/blob/v0.31.1/comfy/quant_ops.py)

这也意味着不应为了这两个 W4A8 文件优先安装社区 W4A8 loader：0.31.1 已有 stock 格式支持，而且该社区 loader 的文档仍标注 Windows 不支持。先在原生 loader 上验证，避免引入第二套量化实现。[ComfyUI-W4A8-Loader README](https://github.com/starsFriday/ComfyUI-W4A8-Loader/blob/main/README.zh-CN.md)

### 是否需要额外自定义节点

标准 H3 T2V、I2V、R2V **不需要新增自定义节点**。ComfyUI 官方教程写明原生支持 H3，v0.31.1 自带 `EmptyMiniMaxH3LatentAV`、`MiniMaxH3ImageToVideo`、`MiniMaxH3ReferenceToVideo`；官方模板使用的模型、文本编码器、VAE、采样与 H3 条件节点均来自 Core。[ComfyUI H3 教程](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)、[H3 原生节点源码](https://github.com/Comfy-Org/ComfyUI/blob/v0.31.1/comfy_extras/nodes_minimax_h3.py)、[官方 T2V 模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)

以下组件仅在特定需求下可选，而不是运行 H3 的前置条件：

- 视频合成、音频混流或批量保存若超出 Core 模板能力，可再按工作流需要选 VideoHelperSuite 等工具节点；首轮可行性验证不要先增加变量。
- ComfyUI-GGUF 只与 GGUF loader 路线有关；现有 `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` 已因元数据格式不兼容而失败，安装或更新节点并不能自动修复这个文件。
- 社区 W4A8 loader 对当前 v0.31.1 stock 路径不是必需，且 Windows 支持状况反而更差。
- Flash Attention、SageAttention、Triton 或 Nunchaku 等二进制加速节点不是 H3 原生工作流依赖，并会显著扩大 cu130 升级的兼容性面；第一轮应保持禁用或维持原状。

### 秋叶环境维护升级的风险

启动器的 cu130 项只列出替换 PyTorch 三件套，没有列出同步替换 xformers。本机 `xformers 0.0.30` 的包元数据明确要求 `torch==2.7.0`，因此升级到 torch 2.9.1 后，现有 xformers 至少会形成依赖冲突，严重时可能因二进制 ABI/CUDA 不匹配而导入失败。秋叶的该 cu130 配置不是一套包含 xformers 的完整锁定环境，不能把按钮存在视为所有自定义节点均已兼容。

同类风险还包括：

- 任何包含已编译 `.pyd`/CUDA 扩展的自定义节点或 Python 包，例如 Flash Attention、SageAttention、Nunchaku、bitsandbytes、自编译 Triton kernel，可能绑定特定 torch、CUDA 或 Python ABI。
- `torchvision`、`torchaudio` 必须和 torch 使用官方配套版本，不能只替换单个包；启动器此处会成组升级，这是正确方向。
- 当前环境在升级前运行 `pip check` 已存在多项与 OpenCV、`av`、pydantic、aiohttp 等无关的历史冲突，不能把升级后的所有 `pip check` 输出都归因于 cu130；应保存升级前基线，只比较新增冲突。
- 系统级 CUDA Toolkit 与 wheel runtime 是两个层次；为这个试验额外修改系统 CUDA、PATH 或编译工具链会扩大破坏面，没有必要。

### 推荐试验与回滚门槛（本次未执行）

最安全的路径是复制整个 `C:\App\ComfyUI-aki-v1.6` 整合包或至少完整复制其中的 `python` 目录，在副本上用秋叶环境维护选择 cu130。仅备份 `ComfyUI` Git 目录不足以回滚 Python 二进制依赖；模型和 `E:\Documents\ComfyUI` 可以继续通过现有路径配置共享，不必重复复制。

升级前建议记录：`pip freeze`、`pip check`、Python/torch/vision/audio/xformers/comfy-kitchen 版本、ComfyUI 启动日志，以及一条已知可用的旧工作流结果。升级后依次验证：

1. `torch.__version__` 为 2.9.1、`torch.version.cuda` 为 13.0，`torch.cuda.is_available()` 为真，设备仍识别为 RTX 5070 Ti / capability 12.0。
2. `import torchvision`、`import torchaudio`、`import comfy_kitchen` 成功；`xformers` 若仍是 0.0.30，应视为不兼容并在测试副本中禁用 xformers 路径，而不是强行沿用。
3. ComfyUI 启动日志不再出现“需要 cu130”的警告，并报告 comfy-kitchen CUDA backend 已启用；否则升级没有取得本次试验的核心收益。
4. 先回归一条现有、短小的图像工作流，再以约 576p、124 帧测试 H3 INT4 ConvRot；最后才试两个 `asym_w4a8_int8` 文件。
5. 比较每步耗时、峰值显存、系统内存、是否 OOM、输出视频与音频完整性。若只改变版本号但没有速度收益，或旧工作流/关键节点出现二进制导入错误，就恢复整套旧 `python` 目录，而不是在主环境中连续补包。

因此，用户看到的 cu130 选项**值得一试，但前提是以“整合包副本 / Python 目录快照 + 明确验收表”的方式试**。直接点击覆盖也许能让 H3 量化 kernel 变快，但它当前确定会留下 xformers 版本冲突，风险高于普通 Core 更新。

## 对本机资源的保守估计

- GPU：主模型已实测完整驻留约占 10.63 GiB。NVFP4 文本编码器文件 14.61 GiB，不可能与它同时完整驻留 16GB GPU；必须把编码器放 CPU，并依赖 VAE/主模型按需换入。
- RAM：本机 31.84 GiB。NVFP4 文本编码器 + 一个 INT4 denoiser + 两个 VAE 的磁盘权重约 30.6 GiB，尚未计 Python/PyTorch、解量化临时张量、视频 latent、参考输入和操作系统；应保留分页文件，先以短视频和小画幅做烟测。社区 INT4 ConvRot 模型卡对 12GB GPU 也建议 32GB+ RAM 和快速 NVMe，本机 32GB RAM 只是在建议下限附近。[Merserk INT4 ConvRot 模型卡](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)
- 视频 latent：ComfyUI 默认 1344x768、124 帧是接近 1 MP、5 秒的模板基线；16GB 卡建议先降到约 576p--768p 短边、124 帧，成功后再增加画幅或帧数。官方教程说明约 1.0 MP/16:9 会得到约 1344x768，且画幅须按 32 对齐。[ComfyUI H3 教程](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- R2V：参考图最多 9 张、视频最多 3 段、音频最多 3 段；参考 token 会随采样保留，官方节点说明 `max` 参考图尺寸会显著变慢，因此本机首轮应使用少量、低分辨率参考输入。[MiniMax 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3#model-variants-and-input-specifications)、[ComfyUI H3 节点源码](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)

## 推荐验证顺序

1. 确认当前 ComfyUI 0.31.1 已加载 H3 原生节点；使用官方 `video_minimax_h3_t2v.json` 模板。[T2V 模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)
2. 只放入 FL2VA INT4 ConvRot、NVFP4/AWQ text encoder、video/audio VAE；文本编码器选 CPU/offload；初始分辨率设置为 576p 左右、124 帧、CFG 1.0。H3 是 CFG-distilled 权重，官方/社区说明 CFG 不应高于 1.0。[MiniMax H3 README](https://huggingface.co/MiniMaxAI/MiniMax-H3#local-deployment-of-h3-base)、[Unsloth GGUF README](https://huggingface.co/unsloth/MiniMax-H3-GGUF)
3. 记录启动日志中的显存、是否发生 OOM、首帧/完整 MP4、音频是否存在；不要以“模型文件能出现在下拉框”作为成功标准。
4. 再换 Ref2VA INT4 做单张参考图 R2V；最后才测试多参考输入、长视频和更高分辨率。
5. INT4 ConvRot 基线成功后，再对两个已经能加载的 `asym_w4a8_int8` 文件做短视频采样验证；不要使用当前 Q2 GGUF text encoder，它已在本机确定加载失败。

## 已知限制与法律

- 官方 H3-Context-IR 和 Regenerate-2K 尚未开源；本地 ComfyUI 工作流是 H3-Base 768p 级别，不等同于官方 API 的完整 2K 工作流。[MiniMax H3 README](https://huggingface.co/MiniMaxAI/MiniMax-H3#h3-context-ir)
- 初始开源版本使用 full attention，长上下文会比未来 sparse-attention 版本更重。[MiniMax H3 README](https://huggingface.co/MiniMaxAI/MiniMax-H3#h3-base)
- MiniMax H3 及其量化衍生物使用 MiniMax H3 Community License；量化仓库属于衍生模型，使用前应阅读适用地区、用途和分发限制。[许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)、[许可证问答](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md)

### 总判断

**本机可以把 H3 作为实验性本地视频模型运行，首发基线应是 INT4 ConvRot FL2VA/Ref2VA + NVFP4/AWQ text encoder + ComfyUI 原生 H3 节点；这些组件已分别通过本机加载/前向验证。成功标准仍应是短视频完整生成并有音频。** 16GB VRAM、32GB RAM 和 cu128 使它处在“需要 offload/eager fallback 的边缘配置”，因此速度、最长视频长度和 R2V 复杂参考能力不能预先承诺。两个 `asym_w4a8_int8` 文件在 v0.31.1 也已完成模型加载，但尚未完成采样验证；当前 Q2_K_M GGUF 仍不能由现有加载路径使用。
