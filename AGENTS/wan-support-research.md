# Wan 视频支持可行性研究

> 调研时间：2026-08-09
> 目标机器：RTX 5070 Ti 16GB、i5-13490F、32GB 系统内存
> 范围：Wan2.1/Wan2.2 文生视频与图生视频、FP8/GGUF 支持、与当前 Core 的接入难度。

## 结论

技术上可行。ComfyUI Core 已原生识别 Wan2.1/Wan2.2 模型，并提供 Wan 视频条件节点；
ComfyUI 官方也提供 Wan2.2 5B、14B T2V 和 14B I2V 工作流。当前项目的障碍不在
ComfyUI 上游能力，而在本项目 Worker 目前只实现了单模型、二维 latent、单次采样和
PNG 输出，尚未把 ComfyUI 的 Wan 工作流语义接进来。

对这台机器，建议按以下顺序选择：

1. **默认首选：Wan2.2-TI2V-5B FP16 + UMT5 XXL FP8。** 一个模型同时覆盖 T2V/I2V；
   ComfyUI 官方文档明确称 5B 版本通过原生 offloading 可较好适配 8GB VRAM，因此
   16GB 显存有足够可行性。官方工作流使用约 9.31GiB 的 5B FP16 扩散模型、FP8 文本
   编码器和 Wan2.2 VAE。[ComfyUI 官方 Wan2.2 指南](https://docs.comfy.org/tutorials/video/wan/wan2_2.md)
2. **低资源 T2V 备用：Wan2.1-T2V-1.3B。** 官方给出的峰值需求是 8.19GB VRAM，
   目标分辨率以 480P 为宜；但它不解决 I2V，因此不适合作为统一的视频模式。
   [Wan2.1 官方仓库](https://github.com/Wan-Video/Wan2.1)
3. **质量实验档：Wan2.2 A14B FP8，先 480P、短帧。** 它可以通过 ComfyUI 原生
   FP8 和 offload 尝试，但不应作为 16GB/32GB 机器的默认模型。A14B 是两个约 14B
   的专家，总参数约 27B，只是每个采样阶段激活其中一个；实际必须同时准备 high-noise
   和 low-noise 两个模型。[Wan2.2 官方 MoE 说明](https://github.com/Wan-Video/Wan2.2#1-mixture-of-experts-moe-architecture)
4. **GGUF 仅作为 A14B 的可选降级后端。** Q4_K_S/Q4_K_M 比双 FP8 更可能适应
   32GB 系统内存，但属于社区 custom node 路径，维护风险和兼容性风险均高于原生
   FP8；不建议让第一版 Wan 支持依赖 GGUF。

如果准备长期使用 A14B，**把系统内存升级到 64GB 比继续压低量化等级更有价值**。
32GB 标称内存约只有 29.8GiB 可用，而 A14B 的两个 FP8 专家和 FP8 文本编码器的文件
体积合计已经约 32.9GiB，尚未计 VAE、latent、Python/PyTorch、视频帧和操作系统。

## 模型与硬件适配

### Wan2.2 TI2V-5B

Wan2.2-TI2V-5B 是 5B dense 模型，使用高压缩 Wan2.2 VAE，原生统一支持 T2V/I2V；
Wan 官方说明它支持 720P、24fps，并称无特定优化时可在单张消费级 GPU 上于 9 分钟内
生成 5 秒 720P 视频。Wan 官方原生推理脚本标注至少 24GB VRAM，但 ComfyUI 官方说明
借助其原生 offloading，5B 版本可较好适配 8GB VRAM。两者并不矛盾，后者通过更积极的
CPU/显存卸载换取了更低显存占用和更长运行时间。

来源：

- [Wan2.2 官方仓库：模型能力与原生推理显存要求](https://github.com/Wan-Video/Wan2.2)
- [ComfyUI 官方 Wan2.2 工作流：8GB VRAM、模型组合与节点流程](https://docs.comfy.org/tutorials/video/wan/wan2_2.md)
- [Comfy-Org 官方重打包模型](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files)

对目标机器的判断：

- 16GB 显存足以作为首发支持目标；32GB 内存也能容纳 5B 模型、FP8 UMT5 和运行时，
  但生成 720P 长序列时仍应保留分页文件并控制帧数。
- 不应承诺固定生成耗时。Wan 官方的“低于 9 分钟”不是 5070 Ti 基准，实际耗时会受
  帧数、分辨率、步数、attention 实现和 offload 频率影响。
- 当前 Worker 默认超时是 300 秒，而官方消费级 GPU 基准已经接近 9 分钟；视频模式必须
  使用显著更长的超时，否则即使显存足够也会被 Core 主动终止。

### Wan2.1-T2V-1.3B

Wan2.1 官方明确给出 T2V-1.3B 只需 8.19GB VRAM；在 RTX 4090 上生成 5 秒 480P
约 4 分钟，且该数字没有使用量化优化。官方同时建议 1.3B 以 480P 为目标，虽然能尝试
720P，但稳定性较差。Wan2.1 官方 I2V 是 14B，因此 1.3B 只能作为 T2V 的轻量备用。

来源：[Wan2.1 官方仓库](https://github.com/Wan-Video/Wan2.1)

### Wan2.2 T2V/I2V-A14B

A14B 的命名容易造成误判。它是两专家 MoE：high-noise 和 low-noise 专家各约 14B，
总参数约 27B，每一步只激活一个专家。ComfyUI 官方 14B 工作流也确实包含两个模型
加载器和两段高级采样，而不是加载单个“14B”文件。

Wan 官方原生脚本对 T2V-A14B 和 I2V-A14B 均标注至少 80GB VRAM；ComfyUI 的
FP8/offload 路径可以显著降低显存门槛，但不会消除两个专家的存储和系统内存压力。

来源：

- [Wan2.2 官方仓库：A14B 结构、80GB 原生推理要求](https://github.com/Wan-Video/Wan2.2)
- [ComfyUI 官方 14B T2V/I2V 工作流](https://docs.comfy.org/tutorials/video/wan/wan2_2.md)
- [ComfyUI 官方 14B T2V 模板](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_14B_t2v.json)
- [ComfyUI 官方 14B I2V 模板](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_14B_i2v.json)

## 量化支持现状

| 路径 | 上游支持度 | 对本机的意义 | 当前 Core 状态 |
| --- | --- | --- | --- |
| FP16/BF16 safetensors | ComfyUI Core 原生；官方 5B 工作流使用 FP16 扩散模型 | 5B 首选；A14B 太大 | safetensors loader 已有，但视频流程未接入 |
| FP8 scaled safetensors | ComfyUI 官方 14B T2V 工作流和官方重打包仓库提供 high/low FP8；FP8 UMT5 也是官方工作流组合 | A14B 可尝试，但 32GB 系统内存是主要瓶颈 | 文件格式可被现有目录发现，单模型 Worker 无法表达双专家流程 |
| GGUF Q2-Q8 | ComfyUI 官方文档列为 Community Resources；依赖 City96/ComfyUI-GGUF，自述仍是 WIP | Q3/Q4 可显著减小 A14B 权重规模，可能以速度/质量和兼容性换容量 | 完全不支持：目录不扫描 `.gguf`，且 Worker 使用 stock `UNETLoader` |

### FP8

ComfyUI 对 Wan FP8 已不是只能依赖第三方 wrapper 的实验路径：官方 Wan2.2 14B T2V
工作流直接使用 `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` 和
`wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors`，并配合
`umt5_xxl_fp8_e4m3fn_scaled.safetensors`。官方 Comfy-Org 仓库目前还提供 I2V 的
high/low FP8 scaled 文件。

需要注意官方教程存在一处文档不一致：I2V 下载列表仍写 FP16 文件，但后续步骤误写成
T2V FP8 文件名；实际模型仓库同时存在 I2V FP16 和 I2V FP8 scaled。接入时应以当前
官方模型仓库和工作流模板为准，不应机械照抄该段文字。

来源：

- [ComfyUI 官方 Wan2.2 指南](https://docs.comfy.org/tutorials/video/wan/wan2_2.md)
- [Comfy-Org Wan2.2 diffusion_models](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)
- [ComfyUI 原生 Wan 模型识别代码](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py)

### GGUF

GGUF 不是 ComfyUI Core 的 stock `Load Diffusion Model` 路径。ComfyUI 官方 Wan2.2
文档将 GGUF 放在“Community Resources”，要求安装 `ComfyUI-GGUF`；该项目 README
也要求用 `Unet Loader (GGUF)` 替换 stock loader，并明确标注项目仍在 WIP。其量化 T5
loader 和 LoRA 支持也不应默认视为与原生 safetensors 等价成熟。

Wan2.2 A14B GGUF 仍必须下载 high-noise 和 low-noise 两个专家。以 bullerwins T2V
仓库的文件元数据为例：

| 量化 | 单个专家 | 两个专家 | 加 FP8 UMT5 后的权重文件规模 |
| --- | ---: | ---: | ---: |
| FP8 scaled safetensors | 约 13.31GiB | 约 26.62GiB | 约 32.89GiB |
| Q3_K_M GGUF | 约 6.68GiB | 约 13.36GiB | 约 19.63GiB |
| Q4_K_M GGUF | 约 8.99GiB | 约 17.98GiB | 约 24.25GiB |

这些数字只是仓库文件体积，不是峰值 RAM/VRAM；运行时仍需 VAE、latent、视频帧、
解量化临时空间和框架开销。对 32GB 内存机器，Q4_K_M 是偏质量的尝试档，Q3_K_M
是偏容量的保守档。GGUF 的优势是容量，不代表一定比原生 FP8 更快。

来源：

- [ComfyUI 官方 Wan2.2 指南：GGUF 社区资源](https://docs.comfy.org/tutorials/video/wan/wan2_2.md)
- [City96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [bullerwins Wan2.2 T2V-A14B GGUF](https://huggingface.co/bullerwins/Wan2.2-T2V-A14B-GGUF/tree/main)
- [bullerwins Wan2.2 I2V-A14B GGUF](https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/tree/main)
- [QuantStack Wan2.2 GGUF 集合](https://huggingface.co/collections/QuantStack/wan22-ggufs-6887ec891bdea453a35b95f3)

## 当前 Core 的接入可行性

### 有利条件

- Worker 已经在 ComfyUI 自带 Python 环境中直接导入 Core，能复用 ComfyUI 的原生
  Wan 模型识别、采样、VAE 和显存 offload，不需要另起一套推理后端。
- 单 GPU 串行队列与视频生成天然匹配，长驻 Worker 也有利于同模型连续任务复用。
- 当前 `UNETLoader`、`CLIPLoader`、`VAELoader` 和底层 `comfy.sample` 路径可继续作为
  5B 原生 safetensors 实现的基础。

### 必须扩展的边界

1. **领域模型不是视频模型。** [`domain.py`](../diffusion_workbench_core/domain.py)
   的 `Mode` 没有 Wan，`GenerationSettings` 没有帧数、FPS、输入图像、任务类型、
   flow shift 或双专家配置。
2. **资源模型只能表达一个扩散模型。** A14B 需要 high-noise/low-noise 两个模型和
   两阶段采样；当前 [`config.py`](../diffusion_workbench_core/config.py) 与 JobRecord 只有
   单个 `model_path`。
3. **latent 和条件构造是图片专用。** [`comfy_worker.py`](../diffusion_workbench_core/comfy_worker.py)
   当前创建 `[B,C,H,W]` 四维零 latent；Wan 使用带时间轴的五维 latent。5B 官方流程
   使用 `Wan22ImageToVideoLatent`，Wan2.1/14B I2V 使用 `WanImageToVideo`，不能仅把模型
   文件名换成 Wan。
4. **采样流程不同。** 5B 官方模板包含 `ModelSamplingSD3`；A14B 模板包含两个
   `UNETLoader`、两个 `ModelSamplingSD3` 和两次 `KSamplerAdvanced`，在切换点把第一段
   latent 交给第二个专家。当前 Worker 只有一次 `_sample()`。
5. **产物链路完全是 PNG。** Worker 只取 `VAEDecode` 的第一张图并写 PNG iTXt；
   [`storage.py`](../diffusion_workbench_core/storage.py) 固定分配 `.png`；API、album、前端
   也固定使用 `image/png` 和图片 URL。Wan 需要帧序列到视频容器的编码、视频 MIME、
   下载/播放接口，以及替代 PNG iTXt 的可追溯元数据方案。
6. **I2V 输入没有进入 Core 契约。** API `CreateJobsRequest` 没有输入图像引用。新增时
   仍应遵守“客户端不能提交服务器路径”，通过受控上传/资产 ID 映射到服务端路径。
7. **默认超时过短。** [`persistent_runtime.py`](../diffusion_workbench_core/persistent_runtime.py)
   对整个任务使用 `worker_timeout_seconds`；默认 300 秒低于 Wan2.2 5B 官方消费级 GPU
   基准，视频模式必须有更合理的超时策略。
8. **GGUF 需要独立 loader 适配。** [`catalog.py`](../diffusion_workbench_core/catalog.py)
   只扫描 `.safetensors`/`.sft`，Worker 直接实例化 stock `UNETLoader`；即使用户安装了
   ComfyUI-GGUF，当前 Worker 也不会自动改用 `UnetLoaderGGUF`。

因此，**5B 原生 safetensors 是最小且最稳的首发范围**；A14B 不只是“换一个大模型”，
而是双模型资源、双阶段采样和更强内存治理；GGUF 又在其上增加 custom node 生命周期与
loader 抽象。合理的演进顺序是：

1. 先接 Wan2.2 TI2V-5B 原生 FP16/FP8 UMT5，完成 T2V/I2V、视频产物和长任务协议。
2. 再接 A14B FP8 high/low 双专家和两阶段进度/取消/模型复用。
3. 最后把 GGUF 做成明确可选的 loader/backend，不让它污染默认原生路径。

## 推荐配置基线

首发验证建议使用 ComfyUI 官方 5B 组合：

- 扩散模型：`wan2.2_ti2v_5B_fp16.safetensors`
- 文本编码器：`umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- VAE：`wan2.2_vae.safetensors`
- 后端：最新 ComfyUI Core 的 stock loader 和 native offloading
- 初始压力控制：单任务、较短帧数；先验证低于 720P 或较小画幅，再逐步增加到模型
  官方 720P 目标；分页文件保持启用并留出足够磁盘空间。

不建议首发即默认以下组合：

- Wan2.2 A14B 双 FP8 + 32GB RAM；权重文件规模已经超过可用物理内存。
- Wan2.2 A14B FP16/BF16；对 16GB 显存和 32GB 内存不现实。
- A14B GGUF 作为唯一实现；它会把核心功能绑定到 WIP custom node。
