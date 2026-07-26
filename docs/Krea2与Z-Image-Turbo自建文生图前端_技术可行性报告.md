# Krea 2 与 Z-Image-Turbo 自建文生图前端：技术可行性分析与实施计划

> 目标：抛弃 ComfyUI，用纯 Python（diffusers）+ 自建前端页面实现固定工作流——输入 prompt、选尺寸、选模型、定步数、点生成；二期加 Upscale。本报告覆盖技术可行性、实现路径、官方文档源地址、扩展路线、5070 Ti 16GB 的硬件适配与环境方案（原生 / WSL2 / Docker）。

---

## 1. 结论摘要（TL;DR）

- **完全可行。** 两个模型都有官方/社区维护的 diffusers Pipeline（`Krea2Pipeline` 与 `ZImagePipeline`），纯 Python 十几行代码即可完成一次文生图，不依赖 ComfyUI。ComfyUI 只是调度壳，底层同样是 PyTorch + safetensors 权重 + VAE 解码，绕过它没有技术障碍。
- **5070 Ti 16GB 能跑，但要注意精度与显存策略。** Z-Image-Turbo 官方明确定位"fits comfortably within 16G VRAM consumer devices"，bf16 下可跑，fp8 更从容 [^33^]。Krea 2 Turbo 是 12B DiT，bf16 全量约需 16GB 显存的"舒适线"上沿，官方与社区已提供 fp8 / nvfp4 量化版，fp8 版约 10–12GB，5070 Ti 可以轻松驾驭 [^2^]。
- **环境推荐：Windows 原生 + uv/venv 起步，或 WSL2；暂不建议 Docker。** 5070 Ti 是 Blackwell 架构（sm_120），必须 CUDA 12.8+ 与 PyTorch 2.7+（cu128）[^31^][^21^]。单人单机自用场景，Docker 的隔离价值低、GPU 穿透和镜像体积反而添麻烦。
- **前端方案：FastAPI 后端 + 简单网页（React/Vue 或单页 HTML），WebSocket 推进度。** 工作流固定意味着后端只需暴露一个 `/generate` 接口，比 ComfyUI 的节点图简单一个数量级。
- **Upscale 二期实现：** 快速档用 Real-ESRGAN x4+（1024→4096 约 2 秒、约 1GB 显存）；质量档用 Tile + 扩散重绘[^41^]。

---

## 2. 两个模型的官方信息与文档源地址

### 2.1 Krea 2（krea-ai）

Krea 2 是 Krea.ai 于 **2026 年 6 月 22 日**开源的 **12B 参数 Diffusion Transformer**，单流（single-stream）架构，文本与图像 token 共享 attention 与 MLP；文本编码器为 **Qwen3-VL 4B**，VAE 复用 **Qwen Image VAE** [^2^][^36^]。发布时即开放权重，社区数小时内就出现了 fp8 / GGUF / INT8 量化 [^1^]。

| 检查点 | 定位 | 步数 / CFG | 典型用途 |
|---|---|---|---|
| **Krea 2 Turbo** | 蒸馏 + RL 后训练的生产推理模型 | **8 步，guidance=0.0**，原生 2K，官方称约 2 秒出图 [^2^] | 你的主生成模型 |
| **Krea 2 Raw** | 未蒸馏的中途训练检查点 | 52 步，guidance=3.5，算力开销大 [^36^] | LoRA 训练 / 后训练研究，不适合日常出图 |

**官方源地址：**

| 资源 | 地址 |
|---|---|
| HuggingFace 模型卡（Turbo） | https://huggingface.co/krea/Krea-2-Turbo |
| HuggingFace 模型卡（Raw） | https://huggingface.co/krea/Krea-2-Raw |
| ModelScope 镜像（国内下载更快） | https://www.modelscope.cn/models/krea/Krea-2-Turbo [^43^] |
| 官方推理代码库 | https://github.com/krea-ai/krea-2 [^34^] |
| 文本编码器 + VAE（Comfy-Org 打包） | https://huggingface.co/Comfy-Org/Krea-2 [^4^] |
| diffusers 文档 | https://huggingface.co/docs/diffusers/ |

许可证为 **Krea 2 Community License**：个人与 50 人以下团队可商用，50 人以上需企业授权；研究与非商用免费 [^2^]。

### 2.2 Z-Image-Turbo（Tongyi-MAI / 阿里通义）

Z-Image-Turbo 是阿里 Tongyi-MAI 团队 **2025 年 11 月 26 日**发布的 6B 参数模型，采用 **S3-DiT 单流扩散 Transformer**，通过 Decoupled-DMD 蒸馏实现 **8 步（NFE）出图**，擅长照片级真实感与中英双语文字渲染；**Apache 2.0 协议，完全可商用** [^33^][^5^]。2025 年 12 月在 Artificial Analysis 文生图榜排第 8、开源模型第 1 [^33^]。同家族还有 Z-Image（50 步基础版，2026-01-27 发布，支持 CFG 与微调）、Z-Image-Edit、Z-Image-Omni-Base [^33^][^5^]。

| 资源 | 地址 |
|---|---|
| 官方 GitHub（含技术报告链接） | https://github.com/Tongyi-MAI/Z-Image [^33^] |
| HuggingFace 模型卡 | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo |
| ModelScope 模型页 | https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo |
| ModelScope 部署指南（中文） | https://www.modelscope.cn/learn/4979 [^8^] |

Z-Image-Turbo 的组件结构与 Krea 2 类似：DiT 主体（transformer/）、文本编码器（text_encoder/，Qwen3 系）、**VAE（vae/）**、tokenizer——ModelScope 的推理示例按这四个子目录分别加载 [^24^]。你说本地已有 VAE，完全对得上：两个模型的 VAE 都是独立 safetensors，可以在自建管线里复用本地文件，不必重复下载。

---

## 3. 纯 Python 文生图：官方推荐做法

### 3.1 Krea 2 Turbo（diffusers）

官方模型卡给出的标准做法 [^43^][^47^]：

```bash
# Krea2Pipeline 目前需要源码安装 diffusers（新版本合入后可直接 pip install diffusers）
pip install git+https://github.com/huggingface/diffusers.git
```

```python
import torch
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Turbo",           # 也可指向本地目录
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    "a fox in the snow",
    num_inference_steps=8,
    guidance_scale=0.0,            # Turbo 为蒸馏模型，CFG 关闭
    width=2048, height=2048,       # 原生 2K；显存吃紧时用 1024–1536
).images[0]
image.save("krea2.png")
```

官方也提供独立代码库（`github.com/krea-ai/krea-2`），通过 `export OSS_TURBO=<turbo.safetensors 路径>` 后运行 `uv run inference.py ... --steps 8 --cfg 0.0 --mu 1.15`[^34^]。自建前端建议走 diffusers 路线——与 Z-Image 统一接口，切换模型只换 Pipeline 类。另外还有 SGLang / vLLM-Omni 的服务化路径（`sglang generate --model-path krea/Krea-2-Turbo ...`），适合日后批量高吞吐场景 [^45^]。

### 3.2 Z-Image-Turbo（diffusers）

官方与社区验证的写法 [^22^][^28^]：

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",    # 或本地路径
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")                    # 16GB 卡建议：pipe.enable_model_cpu_offload()

image = pipe(
    prompt="一只赛博朋克风格的猫",   # 中文 prompt 原生支持
    num_inference_steps=9,         # 官方推荐 8–9
    guidance_scale=0.0,            # Turbo 无 CFG
    height=1024, width=1024,
    generator=torch.manual_seed(42),
).images[0]
image.save("z-image-output.png")
```

**三个坑要注意：**

1. **不要用 fp16。** Z-Image 在 fp16 下会产出纯黑图（NaN latent），官方 issue 已确认；用 **bf16**（5070 Ti 原生支持 bf16，不存在老卡的 emulation 问题）[^11^]。
2. **guidance_scale 必须 0**，蒸馏模型没有 CFG [^5^]。
3. 可选加速：`pipe.transformer.set_attention_backend("flash")`（Flash Attention）与 `pipe.transformer.compile()`（PyTorch 编译，首次慢、后续快）[^5^]。

### 3.3 VAE 的位置

两个模型的 VAE 都是 diffusers 标准 `AutoencoderKL` 组件，`from_pretrained` 会自动从仓库子目录加载；你已下载到本地的话，把模型目录指到本地路径即可，或者单独加载再注入：

```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("/你的本地路径/vae", torch_dtype=torch.bfloat16)
pipe = ZImagePipeline.from_pretrained("...", vae=vae, torch_dtype=torch.bfloat16)
```

注意 Krea 2 用的是 **Qwen Image VAE** [^2^][^4^]，Z-Image 用的是它自带的 VAE，两者**不可互换**；如果本地只有 ComfyUI 目录里的 VAE，要确认它属于哪个模型系。

---

## 4. 5070 Ti 16GB 硬件适配评估

### 4.1 显卡关键参数

RTX 5070 Ti：Blackwell 架构、**16GB GDDR7、896 GB/s 带宽**、FP16 约 44 TFLOPS、支持 FP8/FP4 第五代 Tensor Core，TDP 300W [^35^][^48^]。文生图实测中，FLUX FP8 单图约 14.3 秒、FP4 约 7.2 秒，FP8 性能已贴近 RTX 5080 [^zhihu^]。

### 4.2 两个模型的显存账

| 模型 | 精度 | 显存需求（约） | 5070 Ti 16GB 适配 | 参考速度 |
|---|---|---|---|---|
| Z-Image-Turbo (6B) | bf16 | ~16GB（官方定位"fits 16G"）[^33^] | 可跑，建议 `enable_model_cpu_offload()` 或 fp8 文本编码器 | RTX 4090 上 8 步约 8–13s；5070 Ti 预计 10–15s [^5^] |
| Z-Image-Turbo | fp8 / GGUF Q4 | 6–8GB / 4–6GB [^5^] | 从容 | 更快 |
| Krea 2 Turbo (12B) | bf16 | ~16GB 舒适线上沿 [^2^] | 勉强，需 cpu_offload | 官方称 2K 下约 2 秒（高端卡）[^2^] |
| Krea 2 Turbo | **fp8（推荐）** | ~10–12GB [^2^] | 舒适 | 5070 Ti 的 FP8 Tensor Core 正好用上 |
| Krea 2 Raw (12B) | bf16 | 重负载，52 步 [^36^] | 能跑但慢，不建议日常用 | — |
| Real-ESRGAN x4+（二期） | fp16 | ~1GB [^41^] | 无压力 | 1024→4096 约 2s |

**结论：5070 Ti 16GB 足以支撑"Krea 2 Turbo(fp8) + Z-Image-Turbo(bf16/fp8)"双模型方案。** 建议系统内存 ≥ 32GB（模型加载与 offload 走内存），磁盘预留 ~100GB（Krea 2 全套 bf16 约 24GB、Z-Image-Turbo bf16 约 30–33GB，加量化版与输出目录）[^2^][^9^]。

### 4.3 一个现实约束

16GB 同时驻留两个 12B/6B 模型不现实。架构上要做**模型管理器**：同一时刻只在显存里保留一个 Pipeline，切换模型时卸载旧的再加载新的（safetensors + `low_cpu_mem_usage` 加载约 10–30 秒），前端给出"模型切换中"提示即可。这与 ComfyUI 的行为一致，用户体验无损。

---

## 5. 环境方案：原生 / WSL2 / Docker

### 5.1 硬性前提：Blackwell（sm_120）适配

5070 Ti 计算能力为 **sm_120**，低于 CUDA 12.8 的 PyTorch 会直接报 `sm_120 is not compatible` 或 `no kernel image is available` [^31^][^30^]。要求：

| 组件 | 最低要求 | 说明 |
|---|---|---|
| NVIDIA 驱动 | 570+ | 对应 CUDA 12.8 [^21^] |
| PyTorch | **2.7+ cu128**（建议 2.9 stable） | `pip install torch --index-url https://download.pytorch.org/whl/cu128` [^21^][^23^] |
| CUDA Toolkit | 12.8+（或直接用 PyTorch 自带 runtime，无需单独装） | [^31^] |
| Python | 3.10–3.12 | — |
| 关键库 | diffusers（源码版以含 Krea2Pipeline）、transformers、accelerate、safetensors | [^43^][^28^] |
| 可选加速 | flash-attn / sage-attention、`torch.compile` | [^5^] |

验证命令：`python -c "import torch; print(torch.cuda.get_arch_list())"`，输出含 `sm_120` 即正常 [^21^]。

### 5.2 三种方案对比

| 维度 | Windows 原生 | WSL2 | Docker |
|---|---|---|---|
| 搭建难度 | **最低**（装驱动 + venv） | 中（装 WSL2 + 发行版，驱动走 Windows 穿透） | 高（镜像 + nvidia-container-toolkit） |
| GPU 性能 | 原生 | 约为原生的 90–95%，有虚拟化层与已知限制 [^20^][^13^] | Windows 上 Docker Desktop 底层还是 WSL2，多一层 [^16^] |
| 显存占用 | 桌面合成器占约 1–2GB [^13^] | 同上 + VM 分配开销 | 同上 |
| 文件互操作 | 无需考虑 | 跨文件系统（/mnt/c）I/O 慢 15–30% [^12^] | 卷挂载同理 |
| 生态兼容性 | 个别库 Windows 下麻烦（flash-attn 编译） | **最好**（Linux 生态） | 最好 |
| 适用场景 | 单人单机自用 ✅ | 想要 Linux 生态又不想双系统 ✅ | 多机部署 / 交付他人 |

**给你的建议：**

- **首选 Windows 原生 + `uv`（或 venv）**：单人单机、固定工作流，Docker 的隔离与可移植性价值用不上，反而引入 GPU 穿透配置与数十 GB 镜像的维护成本。
- **次选 WSL2**：如果后面要用 flash-attn、SGLang 这类 Linux 优先的库，WSL2 是性价比最高的折中，性能损失对 8 步出图场景几乎无感 [^20^]。
- **Docker 只在一种情况下值得**：未来要把这套服务部署到别的机器（家里服务器 / 云主机），那时再补一个 `nvidia/cuda:12.8` 基础镜像的 Dockerfile 即可，代码不用改 [^23^]。

---

## 6. 自建前端的实现路径（替代 ComfyUI）

### 6.1 总体架构

![系统架构图](系统架构图.png)

工作流固定是这个项目最大的简化来源。ComfyUI 的复杂性来自"任意节点编排"，而你的需求——prompt / 尺寸 / 模型 / 步数 / seed → 生成——只对应 diffusers 的一次 `pipe(...)` 调用，因此架构可以非常薄：

**后端（FastAPI 单进程即可）：**

```python
# 伪代码骨架
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

app = FastAPI()
manager = ModelManager()          # 模型懒加载/切换/常驻显存

class GenRequest(BaseModel):
    prompt: str
    model: str                    # "krea2-turbo" | "z-image-turbo"
    width: int; height: int
    steps: int
    seed: int | None = None

@app.post("/api/generate")
async def generate(req: GenRequest):
    task_id = manager.submit(req) # 入队，串行消费（单卡只能一次一张）
    return {"task_id": task_id}

@app.websocket("/ws/{task_id}")
async def progress(ws: WebSocket, task_id: str):
    ...  # 用 diffusers 的 callback_on_step_end 逐步推送进度
```

要点：

- **串行任务队列**：单卡一次只跑一张，用 `asyncio.Queue` 或线程池排队，避免 OOM。
- **进度推送**：diffusers Pipeline 支持 `callback_on_step_end`，8 步模型也有 8 次回调，经 WebSocket 推给前端做进度条，体验等同 ComfyUI。
- **模型管理器**：`{"krea2-turbo": Krea2Pipeline, "z-image-turbo": ZImagePipeline}` 注册表 + LRU 单驻留策略；启动时预载默认模型，切换时 `del pipe; torch.cuda.empty_cache()` 后加载新模型。
- **参数校验**：分辨率对齐到 64 的倍数（两个模型的 latent 要求），限制最大边长防止显存爆掉；Turbo 系模型把 guidance 固定为 0，不暴露给用户。
- **历史记录**：SQLite 存（prompt、参数、seed、文件路径），图片落盘到输出目录，前端做个简单的历史画廊——这正好是你固定工作流里最值得留存的元数据。

**前端：**

- 轻量路线：**单页 HTML + 原生 JS / Alpine.js**，一个表单 + 进度条 + 图片墙，一天能写完，FastAPI 直接以静态文件托管。
- 工程路线：**React/Vue + Vite**，便于后续扩展（LoRA 选择器、批量生成、图片对比视图）。你的需求描述（下拉选模型、滑块调步数、点按钮）用任何框架都很直接。
- 不建议 Gradio/Streamlit：固然能 10 分钟出 demo，但自定义交互（进度推送、画廊、二期 upscale 入口）会很快顶到天花板，既然目标是"长期替代 ComfyUI"，一步到位做 FastAPI + 前端更划算。

### 6.2 开发里程碑建议

| 阶段 | 内容 | 预估工作量 |
|---|---|---|
| M1 | 环境装好（cu128 PyTorch + 源码版 diffusers），命令行跑通两个模型各出一张图 | 0.5–1 天 |
| M2 | FastAPI 后端：`/api/generate` + 模型管理器 + 串行队列 | 1–2 天 |
| M3 | 前端页面：参数表单、生成按钮、WebSocket 进度、结果展示与历史 | 1–3 天 |
| M4 | 打磨：seed 固定/随机、参数模板（对应你"固定工作流"的预设）、错误提示 | 1 天 |
| M5（二期） | Upscale 模块（见下） | 1–2 天 |

---

## 7. 后续功能扩展路线

### 7.1 Upscale（你已提到的二期功能）

两条路线，按需求选或并存：

| 路线 | 方案 | 速度 / 显存 | 质量 | 集成方式 |
|---|---|---|---|---|
| 快速放大（默认） | **Real-ESRGAN x4+**（动漫图换 Anime 版） | 1024→4096 约 2s，~1GB 显存 [^41^] | 优秀，通用首选 [^41^] | `pip install realesrgan`，几行代码；作为生成后的可选开关 |
| 高质量重绘 | Tile 分块 + 扩散模型重绘（denoise 0.1–0.3） | 30–120s，6–10GB [^41^] | 上限最高，适合"英雄图" | 二期后段，复用已有 Pipeline 的 img2img/latent 能力 |

Krea 2 Turbo 原生支持 2K 出图，日常其实可以"原生 2K + 需要时再 ESRGAN 到 4K"，显存与速度都最划算。

### 7.2 其他可预见的扩展

- **LoRA 支持**：Krea 官方已放出 4 个风格 LoRA（coolblue / darkbrush / plasmoid / warmpastel），社区 LoRA 在增长；diffusers 侧 `pipe.load_lora_weights(...)` 即可，前端加个多选列表 [^2^][^40^]。Z-Image 系 LoRA 训练走 DiffSynth-Studio [^24^]。
- **img2img / 图像编辑**：Z-Image-Edit 已发布（自然语言编辑）[^33^]；Krea 2 社区有免训练的 Untwisting RoPE 风格迁移方案（diffusers 版已有人移植）[^7^]。
- **批量生成 / 高吞吐**：把推理层换成 **SGLang** 或 **vLLM-Omni**（两者均已 day-0/早期支持这两个模型），HTTP 接口形态不变，前端无感 [^2^][^45^]。
- **生成加速**：`torch.compile`、Flash Attention、Cache-DiT 类缓存加速 [^5^]。
- **服务化 / 多机**：届时再上 Docker + `nvidia/cuda:12.8` 镜像，或直接迁移到 Linux 服务器。

---

## 8. 风险与注意事项

1. **`Krea2Pipeline` 太新**：需源码安装 diffusers（`pip install git+https://github.com/huggingface/diffusers.git`）[^43^]，升级 diffusers 时注意回归；可用 `pip freeze` 锁定可用版本组合。
2. **精度坑**：Z-Image 系**禁用 fp16**（黑图问题），统一 bf16 [^11^]；Krea 2 官方也用 bf16 [^43^]。
3. **VAE 不通用**：Krea 2 = Qwen Image VAE，Z-Image = 自带 VAE，本地复用前先核对归属 [^4^][^24^]。
4. **许可证**：Z-Image-Turbo 是 Apache 2.0 无顾虑 [^10^]；Krea 2 社区版对个人/小团队免费可商用，但若用于 ≥50 人组织的产品需企业授权，且要求实现内容过滤 [^2^]。
5. **Windows 桌面占显存**：16GB 卡上系统会占 1–2GB，跑 bf16 Krea 2 时建议关掉占显存的程序，或直接上 fp8 [^13^][^2^]。
6. **依赖版本漂移**：PyTorch cu128、diffusers 源码版、transformers 之间的兼容是本项目最大的维护点；建议用 `uv.lock` / `requirements.txt` 固定整套环境。

---

*本报告基于 2026 年 7 月 26 日可获取的公开资料整理，模型版本与库接口迭代较快，实施前建议以官方模型卡为准复核一次。*

[^1^]: https://setuproll.com/news
[^2^]: https://www.buildfastwithai.com/blogs/krea-2-open-source-review-raw-turbo
[^4^]: https://www.runcomfy.com/comfyui-workflows/krea-2-turbo-image-to-image-comfyui-workflow-fast-style-transfer
[^5^]: https://blog.bymar.co/posts/z-image-vs-z-image-turbo-comparison/
[^7^]: https://gist.github.com/apolinario/9ecc9e0efffbbf133fe997b7181b6cfa
[^8^]: https://www.modelscope.cn/learn/4979
[^9^]: https://zimageturbo.org/z-image-turbo
[^10^]: https://willitrunai.com/image-models/z-image-turbo
[^11^]: https://github.com/Tongyi-MAI/Z-Image/issues/14
[^12^]: https://www.kunalganglani.com/blog/wsl2-vs-native-linux-development
[^13^]: https://www.kunalganglani.com/blog/local-ai-linux-windows-macos
[^16^]: https://zenvanriel.com/ai-engineer-blog/wsl2-falls-short-local-ai-development/
[^20^]: https://cdfed.com/blog/linux-vs-wsl-ai-workloads/
[^21^]: https://github.com/devnen/Chatterbox-TTS-Server
[^22^]: https://zimageturbo.ink/
[^23^]: https://github.com/pluja/whishper/issues/172
[^24^]: https://www.modelscope.cn/models/laonansheng/Rihefu-SSyouya-Z-Image-Turbo-v1.0
[^28^]: https://blog.csdn.net/weixin_42603332/article/details/157660869
[^30^]: https://github.com/hirotomusiker/CLRerNet/issues/79
[^31^]: https://www.cnblogs.com/arwen-xu/p/19162267
[^33^]: https://github.com/Tongyi-MAI/Z-Image
[^34^]: https://comfyui-wiki.com/en/news/2026-06-22-krea-2-open-source-text-to-image
[^35^]: https://gpuhunter.io/gpu/rtx-5070-ti
[^36^]: https://novalogiq.com/2026/06/24/enterprise-grade-ai-image-generation-in-2-seconds-is-here-krea-2-raw-and-turbo-available-as-open-weights-under-custom-license/
[^40^]: https://hackernoon.com/krea-2-realism-lora-brings-candid-photorealism-to-krea-2
[^41^]: https://zsky.ai/blog/ai-upscaling-comparison
[^43^]: https://www.modelscope.cn/models/krea/Krea-2-Turbo
[^45^]: https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/text_to_image/
[^47^]: https://opencsg.com/models/AIWizards/Krea-2-Turbo?tab=summary
[^48^]: https://willitrunai.com/de/gpus/rtx-5070-ti-16gb
[^zhihu^]: https://zhuanlan.zhihu.com/p/23918455117
