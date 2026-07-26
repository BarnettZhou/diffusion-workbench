# Krea2 非 ComfyUI 后端可行性研究

更新时间：2026-07-26。研究对象是本机 `E:\Documents\ComfyUI\models\diffusion_models\krea2` 下的 13 个 checkpoint，以及 RTX 5070 Ti 16GB / Windows 环境。

## 结论

Krea2 **不是只能通过 ComfyUI 推理**。官方仓库本身提供独立 PyTorch 推理代码，Diffusers 也已经包含 `Krea2Pipeline` 和 `Krea2Transformer2DModel`。真正的限制是：当前 13 个文件不是普通 Diffusers 权重，而是包含 `comfy_quant`、`weight_scale` 和原始 Krea2/ComfyUI key 的专用单文件格式。

因此需要区分两件事：

1. **Krea2 架构能否脱离 ComfyUI运行：能。** 官方 BF16、Diffusers、SGLang 都是可选路线。
2. **当前这些 checkpoint 能否被上述框架直接加载：不能。** 13/13 都有 `model.diffusion_model.*` 前缀；其中 12 个还带 ComfyUI 量化元数据。普通 Diffusers 加载器不会解释这些元数据。

对这台 16GB 显卡，官方 BF16 路线不是日常可用方案：每个 checkpoint 的 DiT 基础张量展开为 BF16 都约 **23.879 GiB**，尚未计入文本编码器、VAE、激活和 CUDA workspace。CPU offload 可以避免立即 OOM，但会重复搬运大权重，正是共享 GPU 内存和生成变慢的来源之一。

最值得继续研究的是：**保留 Diffusers 的 pipeline、scheduler、文本编码器和 VAE，只替换 DiT 的线性层及加载器**。scaled-FP8 可以直接保留文件中的 FP8 权重和 scale，并通过 `torch._scaled_mm` 运算，不必展开整个 BF16 模型。该路线的单层真实权重探针已经成功，但整模型尚未完成。INT8 ConvRot 则需要额外旋转算法或专用 kernel，不能当成普通 INT8 Linear。

## Checkpoint 兼容矩阵

以下数据来自 [research_krea2_backends.py](../research_krea2_backends.py) 的 safetensors 头部扫描和 [.cache/krea2_alternative_backend_probe.json](../.cache/krea2_alternative_backend_probe.json)。所有文件展开 BF16 后的基础权重均约 23.879 GiB。

| 类别 | 模型 | 文件布局 | Diffusers 直接加载 | 自定义 scaled-FP8 路线 | 主要阻碍 |
|---|---|---:|---:|---:|---|
| 标准 scaled-FP8 | `[GPT逼真版]krea2GPTGrandPUSSYTruth_krea2GPT` | 256 FP8 + 256 标量 scale + 256 `comfy_quant` | 否 | 候选，单层已验证 | key 映射、模块替换、整模型加载 |
| 标准 scaled-FP8 | `[色情大师2fp8]pornmasterKrea2_turboV2FP8` | 同上 | 否 | 候选 | 同上 |
| 标准 scaled-FP8 | `[赤佬1.1]redcraft22KREA2INT8_11INT8Native` | 实际同上，文件名虽含 INT8 | 否 | 候选 | 同上 |
| 标准 scaled-FP8 | `[赤佬3]redcraft23INT8INT4FP8_30Krea2` | 实际同上 | 否 | 候选 | 同上 |
| 标准 scaled-FP8 | `Krea2RedMix2.1-8Steps-fp8-scaled` | 同上 | 否 | 候选 | 同上 |
| 标准 scaled-FP8 | `pornmasterKrea2_turboV1FP8` | 同上 | 否 | 候选 | 同上 |
| 混合 scaled-FP8 | `[tyjr]moodyKrea2Mix_tyjrEDITION` | 224 scaled-FP8，另含 F16/BF16/F32 | 否 | 候选，未做整模型验证 | 混合精度加载策略 |
| 混合 scaled-FP8 | `moodyKrea2Mix_v20` | 同上 | 否 | 候选，未做整模型验证 | 同上 |
| 混合 scaled-FP8 | `moodyKrea2Mix_v30` | 224 scaled-FP8，另含 F16 | 否 | 候选，未做整模型验证 | 同上 |
| 无 scale 的 FP8 | `darkBeastINT8Convrot2_darkBeastKREA2FP8` | 430 个 FP8，无 `comfy_quant`/scale | 否，仍是 Comfy key | 未验证 | 必须确认 scale=1 是否为作者意图；非线性参数不能盲目保持 FP8 |
| INT8 ConvRot | `[色情大师2]pornmasterKrea2_v2TurboInt8` | 224 INT8 + per-row scale + ConvRot 元数据 | 否 | 不适用 | 需要 ConvRot 激活旋转及 INT8 kernel |
| INT8 ConvRot | `darkBeastINT8Convrot2_krea211INT8Convrot` | 同上 | 否 | 不适用 | 同上 |
| INT8 ConvRot | `Krea2RedMix2.1-INT8-Convrot` | 同上 | 否 | 不适用 | 同上 |

这里的“候选”只表示文件信息足以构造兼容层，不等于整模型已经跑通。当前只有 ComfyUI 后端完成了 13/13 的完整出图测试。

## 路线一：官方 PyTorch BF16

[Krea2 官方仓库](https://github.com/krea-ai/krea-2)提供独立推理入口 [inference.py](https://github.com/krea-ai/krea-2/blob/main/inference.py)，Turbo 的官方参数是 8 steps、CFG 0、`mu=1.15`。这条路线完全不依赖 ComfyUI，也是验证模型语义最可靠的参考实现。

但它针对官方普通 checkpoint，不认识本机文件里的 `comfy_quant`。若先把当前文件完整反量化为 BF16，单是 DiT 基础权重就是 23.879 GiB；5070 Ti 16GB 无法完整驻留。使用 Accelerate/CPU offload 只能改变权重放置位置，不能恢复量化语义，也不能消除 PCIe 搬运成本。

结论：官方 BF16 路线适合作为正确性基准，或用于显存大于约 32GB 的设备；不适合当前机器的生产 demo。

## 路线二：Diffusers + TorchAO 重量化

Diffusers 已有 Krea2 模型实现，并提供 [量化总览](https://huggingface.co/docs/diffusers/main/en/quantization/overview)与 [TorchAO 集成](https://huggingface.co/docs/diffusers/main/en/quantization/torchao)。量化器源码位于 [diffusers/quantizers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/quantizers)。可行流程是：

1. 以 meta device 构造 `Krea2Transformer2DModel`，避免先分配完整模型。
2. 把原始 Krea2/ComfyUI key 映射为 Diffusers key。
3. 逐层读取 checkpoint；scaled-FP8 用 `weight = qdata * scale` 恢复计算值，ConvRot 则先执行逆旋转。
4. 在单层临时 BF16 张量仍存在时，立即用 TorchAO 重新量化，然后释放临时张量。
5. 保存为 TorchAO 自己的序列化格式，后续直接加载。

它不会让整个 23.879 GiB BF16 模型同时驻留，因此理论上适合 16GB。代价是发生一次“反量化再重量化”，可能引入额外误差；而且 TorchAO 不会自动识别 `comfy_quant`，必须先有转换器。

当前项目环境没有安装 `torchao`，本机尚未验证 Windows + RTX 5070 Ti 上的 Krea2 整模型 TorchAO kernel、序列化和速度。因此这条路线是**合理但未验证的第二优先级方案**。bitsandbytes 和 Quanto 同样只识别自己的量化格式；不能直接加载当前文件。Quanto 在当前 Diffusers 源码中已标记为弃用，不建议新投入。

## 路线三：自定义 `torch._scaled_mm` scaled-FP8

这是最接近当前文件本身、也最可能避免二次量化误差的非 ComfyUI 路线。每个量化线性层保留：

- `weight`: `torch.float8_e4m3fn`
- `weight_scale`: F32 标量
- 计算 dtype: BF16

前向时动态计算输入 scale，把输入转为 FP8，然后调用：

```python
torch._scaled_mm(
    input_fp8,
    weight_fp8.T,
    input_scale,
    weight_scale,
    out_dtype=torch.bfloat16,
)
```

真实 checkpoint 单层探针已在 RTX 5070 Ti 上成功：

| 项目 | 结果 |
|---|---:|
| PyTorch / CUDA | `2.11.0+cu128` / 12.8 |
| 测试层 | `txtfusion.layerwise_blocks.0.attn.gate.weight` |
| 权重 shape | 2560 x 2560 |
| 输入 rows | 64 |
| 20 次平均 kernel 时间 | 0.0317 ms |
| 相对 MAE（对 BF16 反量化参考） | 0.026587 |
| 输出有限值 | 是 |
| CUDA peak allocated | 68.16 MiB |

该结果证明三件事：5070 Ti 的 PyTorch 构建能执行 `_scaled_mm`；文件中的 `qdata + scale` 数学上可直接使用；单层不需要 ComfyUI runtime。它**没有**证明完整 Krea2 能生成正确图片。

整模型仍需实现：Krea2 到 Diffusers 的 key 映射、`nn.Linear` 替换、偏置与非线性参数的混合 dtype、LoRA 行为、CPU/GPU 放置，以及一轮完整 576x576 回归。`torch._scaled_mm` 是 PyTorch 的下划线内部 API，升级 PyTorch 时必须有回归测试。

## INT8 ConvRot 为什么更难

本机 [ComfyUI `ops.py`](C:/App/ComfyUI-aki-v1.6/ComfyUI/comfy/ops.py) 第 1060-1128 行显示，它先解析每层 `comfy_quant`，再按 `format` 创建带布局信息的 `QuantizedTensor`。对 `int8_tensorwise`，除了 `weight_scale`，还读取 `convrot` 和默认 `convrot_groupsize=256`。

ConvRot checkpoint 存的不是普通 INT8 权重。权重在离线量化前经过分组正交旋转；推理时输入也必须执行匹配的在线旋转，再做动态 INT8 量化和矩阵乘法。直接把这些权重交给 TorchAO `Int8WeightOnly`、bitsandbytes `Linear8bitLt` 或普通 `F.linear`，数学结果都不等价。

有两个实现方向：

1. 复刻归一化 Hadamard 分组旋转，使用 `torch._int_mm` 或 Triton/CUDA kernel 保持 INT8。该方向尚未完成整模型验证。
2. 逐层执行 ConvRot 逆变换得到 BF16，再重量化成 TorchAO 支持的格式。这更容易验证，但会产生二次量化误差。

所以 INT8 ConvRot 不应作为第一版非 ComfyUI backend 的目标。先跑通 9 个带 scale 的 FP8/混合 FP8 模型更稳妥。

## SGLang

Krea2 官方仓库把 [SGLang Krea2 cookbook](https://docs.sglang.io/cookbook/diffusion/Krea/Krea-2)列为支持平台，因此 SGLang 是另一条不依赖 ComfyUI 的服务化路线。但这只能证明 SGLang 支持官方 Krea2 模型布局，不能推出它支持 `comfy_quant` 或 ConvRot 单文件。

当前未在本机验证 SGLang，也未验证 Windows 原生支持。若采用这条路线，应优先在 WSL2/Linux 中使用官方 checkpoint；在 16GB 上还必须确认其量化版本和显存峰值。它不适合作为直接消费当前 13 个文件的短期方案。

## 建议的研究顺序

1. 保留现有 ComfyUI runner 作为结果和性能基线。
2. 新建独立实验 backend，只支持标准 scaled-FP8；用 Diffusers Krea2 架构加自定义 `ScaledFP8Linear`。
3. 先做单 block forward，再做单 denoise step，最后跑固定 576x576 / 8 steps 对比图。
4. 每一步都在独立子进程中运行，设置超时，并记录进程 commit、CUDA peak 和输出有限值；发现 BF16 全模型展开立即终止。
5. scaled-FP8 整模型稳定后，再比较 TorchAO 重量化。INT8 ConvRot 放到最后。

最终判断：**有救，而且无需把完整 ComfyUI 当作永久依赖。** 但不能用普通 `from_single_file(..., torch_dtype=bf16)` 解决；正确的替代方案必须理解并保留现有 checkpoint 的逐层量化语义。

## 资料来源

- [Krea2 官方仓库](https://github.com/krea-ai/krea-2)
- [Krea2 官方推理代码](https://github.com/krea-ai/krea-2/blob/main/inference.py)
- [Diffusers 量化总览](https://huggingface.co/docs/diffusers/main/en/quantization/overview)
- [Diffusers TorchAO 文档](https://huggingface.co/docs/diffusers/main/en/quantization/torchao)
- [Diffusers 量化器源码](https://github.com/huggingface/diffusers/tree/main/src/diffusers/quantizers)
- 本机 ComfyUI：`C:\App\ComfyUI-aki-v1.6\ComfyUI\comfy\ops.py` 第 1041-1128 行
- 本项目探针：[research_krea2_backends.py](../research_krea2_backends.py)及其[缓存结果](../.cache/krea2_alternative_backend_probe.json)
