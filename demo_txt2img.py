"""Z-Image-Turbo 加载实现，以及 Krea2 checkpoint 格式校验。

可执行入口已经拆分：
    uv run demo_zit.py --prompt "一只赛博朋克风格的猫"
    uv run demo_krea2.py --prompt "a fox in the snow"

说明：
- 权重全部使用本地 ComfyUI 目录下的 safetensors，不重复下载大文件；
- ZIT 的 config / tokenizer / scheduler 已离线保存到 configs/zit/，
  全程无需访问 HuggingFace（本机对 HF 的 HEAD 请求被网络阻断）；
- Z-Image 系禁用 fp16（会出纯黑图），统一 bf16；
- Krea2 只通过 ComfyUI 原生 scaled-FP8 后端运行，不在此处反量化；
- ZIT 使用 enable_model_cpu_offload() 逐模块上卡。
"""

import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

from krea2_config import KREA2_MODEL_PATH

# ---------------------------------------------------------------------------
# ComfyUI 权重加载：同时兼容纯 bf16 文件与 ComfyUI scaled 量化文件。
# 实测本地文件格式（2026-07 检查 safetensors 头部）：
#   zit 全套              → 纯 BF16，无量化
#   krea2 unet / text enc → "ComfyUI scaled fp8"：weight 为 F8_E4M3，
#                           每个量化 weight 配一个 F32 的 <name>_scale，
#                           另有 comfy_quant 元数据 blob（跳过即可）。
#                           反量化方式：bf16 = fp8.to(f32) * scale
# ---------------------------------------------------------------------------
QUANT_SKIP_SUFFIXES = ("_scale", ".comfy_quant")
CACHE_VERSION = 7
CHECKPOINT_CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "krea2_checkpoint_formats.json"
KREA2_STANDARD_FP8_WEIGHTS = 256
KREA2_STANDARD_BF16_TENSORS = 174
KREA2_STRUCTURE_SIGNATURE = "4c302c490305dfb3b59ec31a49c5108da24ed04e6e42c8423752d8974e187f11"


def _read_checkpoint_cache(cache_path: Path) -> dict:
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"version": CACHE_VERSION, "checkpoints": {}}
    if not isinstance(cache, dict):
        return {"version": CACHE_VERSION, "checkpoints": {}}
    if cache.get("version") != CACHE_VERSION or not isinstance(cache.get("checkpoints"), dict):
        return {"version": CACHE_VERSION, "checkpoints": {}}
    return cache


def _write_checkpoint_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, cache_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _inspect_krea2_scaled_fp8(path: Path) -> dict:
    dtype_counts = Counter()
    base_dtypes = Counter()
    fp8_keys = []
    bf16_count = 0
    structure_entries = []
    errors = []

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        quant_configs = {}

        for key in keys:
            dtype = str(f.get_slice(key).get_dtype())
            dtype_counts[dtype] += 1

            if key.endswith("_scale"):
                if dtype != "F32":
                    errors.append(f"scale {key} 的 dtype 是 {dtype}，预期 F32")
                shape = f.get_slice(key).get_shape()
                if shape:
                    errors.append(f"scale {key} 的 shape 是 {shape}，预期标量")
                else:
                    scale_value = float(f.get_tensor(key).item())
                    if not math.isfinite(scale_value) or scale_value <= 0:
                        errors.append(f"scale {key} 必须是有限正数，实际为 {scale_value}")
                continue

            if key.endswith(".comfy_quant"):
                if dtype != "U8":
                    errors.append(f"量化元数据 {key} 的 dtype 是 {dtype}，预期 U8")
                    continue
                try:
                    raw_config = f.get_tensor(key).numpy().tobytes().decode("utf-8")
                    quant_configs[key] = json.loads(raw_config)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    errors.append(f"无法解析量化元数据 {key}: {exc}")
                continue

            base_dtypes[dtype] += 1
            shape = ",".join(map(str, f.get_slice(key).get_shape()))
            structure_entries.append(f"{key}|{dtype}|{shape}")
            if dtype == "F8_E4M3":
                fp8_keys.append(key)
            elif dtype == "BF16":
                bf16_count += 1

        unsupported = sorted(dtype for dtype in base_dtypes if dtype not in {"F8_E4M3", "BF16"})
        if "I8" in unsupported:
            errors.append("检测到 INT8/INT8 ConvRot 权重")
        if unsupported:
            errors.append("检测到非标准基础权重 dtype: " + ", ".join(unsupported))
        if not fp8_keys:
            errors.append("未检测到 FP8 E4M3 权重")
        if not bf16_count:
            errors.append("未检测到标准 scaled FP8 模型应保留的 BF16 权重")
        if len(fp8_keys) != KREA2_STANDARD_FP8_WEIGHTS:
            errors.append(
                f"FP8 权重数量为 {len(fp8_keys)}，标准 Krea2 应为 {KREA2_STANDARD_FP8_WEIGHTS}"
            )
        if bf16_count != KREA2_STANDARD_BF16_TENSORS:
            errors.append(
                f"BF16 权重数量为 {bf16_count}，标准 Krea2 应为 {KREA2_STANDARD_BF16_TENSORS}"
            )
        structure_signature = hashlib.sha256(
            "\n".join(sorted(structure_entries)).encode("utf-8")
        ).hexdigest()
        if structure_signature != KREA2_STRUCTURE_SIGNATURE:
            errors.append(
                "基础权重 key/shape 与标准 Krea2 结构不匹配："
                f"{structure_signature[:16]}"
            )
        invalid_prefixes = [
            key
            for key in keys
            if not key.endswith(QUANT_SKIP_SUFFIXES) and not key.startswith("model.diffusion_model.")
        ]
        if invalid_prefixes:
            errors.append(f"检测到非 Krea2 权重 key，例如 {invalid_prefixes[0]}")

        missing_scales = []
        missing_quant_metadata = []
        wrong_quant_formats = []
        for key in fp8_keys:
            scale_key = key + "_scale"
            quant_key = (
                key[: -len("weight")] + "comfy_quant"
                if key.endswith(".weight")
                else key + ".comfy_quant"
            )
            if scale_key not in keys:
                missing_scales.append(key)
            if quant_key not in keys:
                missing_quant_metadata.append(key)
                continue
            quant_format = quant_configs.get(quant_key, {}).get("format")
            if quant_format != "float8_e4m3fn":
                wrong_quant_formats.append(f"{key}={quant_format!r}")

        if missing_scales:
            errors.append(f"{len(missing_scales)} 个 FP8 权重缺少 scale，例如 {missing_scales[0]}")
        if missing_quant_metadata:
            errors.append(
                f"{len(missing_quant_metadata)} 个 FP8 权重缺少 comfy_quant，例如 "
                f"{missing_quant_metadata[0]}"
            )
        if wrong_quant_formats:
            errors.append(
                f"{len(wrong_quant_formats)} 个 FP8 权重的量化格式不是 float8_e4m3fn，例如 "
                f"{wrong_quant_formats[0]}"
            )

        fp8_key_set = set(fp8_keys)
        if fp8_key_set:
            dangling_scales = []
            dangling_quant_metadata = []
            for key in keys:
                if key.endswith("_scale") and key[: -len("_scale")] not in fp8_key_set:
                    dangling_scales.append(key)
                elif key.endswith(".comfy_quant"):
                    weight_key = key[: -len("comfy_quant")] + "weight"
                    if weight_key not in fp8_key_set:
                        dangling_quant_metadata.append(key)
            if dangling_scales:
                errors.append(
                    f"{len(dangling_scales)} 个 scale 不属于 FP8 权重，例如 {dangling_scales[0]}"
                )
            if dangling_quant_metadata:
                errors.append(
                    f"{len(dangling_quant_metadata)} 个 comfy_quant 不属于 FP8 权重，例如 "
                    f"{dangling_quant_metadata[0]}"
                )

    unique_errors = list(dict.fromkeys(errors))
    if len(unique_errors) > 8:
        omitted = len(unique_errors) - 8
        unique_errors = [*unique_errors[:8], f"另有 {omitted} 项格式问题已省略"]
    return {
        "valid": not unique_errors,
        "format": "scaled_fp8_e4m3fn" if not unique_errors else "unsupported",
        "dtypes": dict(sorted(dtype_counts.items())),
        "fp8_weights": len(fp8_keys),
        "bf16_tensors": bf16_count,
        "structure_signature": structure_signature,
        "error": "；".join(unique_errors) if unique_errors else None,
    }


def validate_krea2_scaled_fp8(
    path: str, cache_path: str | Path = CHECKPOINT_CACHE_PATH
) -> dict:
    """只允许标准 ComfyUI scaled FP8 Krea2 checkpoint，并缓存头部扫描结果。"""
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"Krea2 checkpoint 不存在: {checkpoint_path}")

    stat = checkpoint_path.stat()
    cache_path = Path(cache_path)
    cache = _read_checkpoint_cache(cache_path)
    cache_key = str(checkpoint_path)
    cached = cache["checkpoints"].get(cache_key)
    cache_matches_file = (
        cached
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
    )
    cache_matches_validator = (
        cached
        and (not cached.get("valid") or cached.get("structure_signature") == KREA2_STRUCTURE_SIGNATURE)
    )
    if cache_matches_file and cache_matches_validator:
        if not cached.get("valid"):
            raise ValueError(f"Krea2 checkpoint 不是标准 scaled FP8: {cached.get('error', '格式不兼容')}")
        return {**cached, "cache_hit": True}

    try:
        result = _inspect_krea2_scaled_fp8(checkpoint_path)
    except Exception as exc:
        raise ValueError(f"无法读取 Krea2 safetensors 头部: {exc}") from exc

    record = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        **result,
    }
    cache["checkpoints"][cache_key] = record
    _write_checkpoint_cache(cache_path, cache)

    if not result["valid"]:
        raise ValueError(f"Krea2 checkpoint 不是标准 scaled FP8: {result['error']}")
    return {**record, "cache_hit": False}


def load_comfy_state_dict(path: str, dtype: torch.dtype = torch.bfloat16) -> dict:
    """读取 ComfyUI safetensors 为普通 state dict；scaled 量化权重就地反量化为 dtype。"""
    state_dict = {}
    dequant_count = 0
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for key in keys:
            if key.endswith(QUANT_SKIP_SUFFIXES):
                continue
            tensor = f.get_tensor(key)
            scale_key = key + "_scale"
            is_quant = not tensor.dtype.is_floating_point or tensor.dtype in (
                torch.float8_e4m3fn, torch.float8_e5m2
            )
            if scale_key in keys and is_quant:
                scale = f.get_tensor(scale_key).float()
                while scale.dim() < tensor.dim():  # 兼容 per-tensor 标量与 per-channel
                    scale = scale.unsqueeze(-1)
                tensor = (tensor.float() * scale).to(dtype)
                dequant_count += 1
            else:
                tensor = tensor.to(dtype)
            state_dict[key] = tensor
    if dequant_count:
        print(f"      检测到 scaled 量化权重，已反量化 {dequant_count} 个张量 → {dtype}")
    return state_dict

# ---------------------------------------------------------------------------
# 模型注册表：权重指向本地 ComfyUI 文件；config/tokenizer/scheduler 已离线
# 下载到 configs/<model>/（本机 huggingface_hub 的 HEAD 请求被网络阻断，
# 无法在线拉取；zit 配置来自 hf-mirror，krea2 为 HF 受限仓库、配置来自
# ModelScope 镜像）。
# ---------------------------------------------------------------------------
MODELS = {
    "zit": {
        "config_dir": "configs/zit",
        "pipeline_cls": "ZImagePipeline",
        "transformer_cls": "ZImageTransformer2DModel",
        "vae_cls": "AutoencoderKL",
        "te_kind": "qwen3",          # Qwen3Model，文件 key 去 "model." 前缀即可
        "pipeline_extra": {},
        "transformer": r"E:\Documents\ComfyUI\models\diffusion_models\zit\redcraftRedzimage_unet.safetensors",
        "vae": r"E:\Documents\ComfyUI\models\vae\zit\zit-ae.safetensors",
        "text_encoder": r"E:\Documents\ComfyUI\models\text_encoders\qwen_3_4b.safetensors",
        "default_steps": 9,
    },
    "krea2": {
        "backend": "comfyui",
        "transformer": str(KREA2_MODEL_PATH),
        "default_steps": 8,
    },
}

def simple_sigmas(steps: int) -> list[float]:
    """返回 ComfyUI simple 调度传给 diffusers 的 sigma 序列（终点 0 由 scheduler 追加）。"""
    if steps <= 0:
        raise ValueError("steps 必须大于 0")
    return [1.0 - index / steps for index in range(steps)]


def remap_te_keys(state_dict: dict, te_kind: str) -> dict:
    """ComfyUI text encoder 的 key → transformers 模型 key（两种映射均已按 key 集合精确验证）。

    qwen3:   model.layers.X          → layers.X          (Qwen3Model)
    qwen3vl: model.visual.X          → visual.X          (Qwen3VLModel 视觉塔)
             model.<text>.X          → language_model.<text>.X
    """
    out = {}
    for key, value in state_dict.items():
        key = key.removeprefix("model.")
        if te_kind == "qwen3vl" and not key.startswith("visual."):
            key = "language_model." + key
        out[key] = value
    return out


def load_pipeline(model_key: str):
    cfg = MODELS[model_key]
    if cfg.get("backend") == "comfyui":
        raise ValueError("Krea2 必须通过 demo_krea2.py 使用 ComfyUI 量化后端")
    cfg_dir = cfg["config_dir"]
    dtype = torch.bfloat16

    import diffusers
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    print(f"[1/5] 加载 transformer: {cfg['transformer']}")
    transformer_cls = getattr(diffusers, cfg["transformer_cls"])
    transformer_sd = load_comfy_state_dict(cfg["transformer"], dtype)
    transformer = transformer_cls.from_single_file(
        transformer_sd, config=cfg_dir, subfolder="transformer", torch_dtype=dtype
    )
    del transformer_sd

    print(f"[2/5] 加载 VAE: {cfg['vae']}")
    vae_cls = getattr(diffusers, cfg["vae_cls"])
    vae = vae_cls.from_single_file(
        cfg["vae"], config=cfg_dir, subfolder="vae", torch_dtype=dtype
    )

    print(f"[3/5] 加载 text encoder: {cfg['text_encoder']}")
    te_config = AutoConfig.from_pretrained(f"{cfg_dir}/text_encoder")
    text_encoder = AutoModel.from_config(te_config, torch_dtype=dtype)
    state_dict = remap_te_keys(load_comfy_state_dict(cfg["text_encoder"], dtype), cfg["te_kind"])
    missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
    del state_dict
    print(f"      权重匹配: missing={len(missing)}, unexpected={len(unexpected)}")

    print("[4/5] 加载 tokenizer / scheduler（本地 configs 目录）")
    tokenizer = AutoTokenizer.from_pretrained(f"{cfg_dir}/tokenizer")
    scheduler_cls = diffusers.FlowMatchEulerDiscreteScheduler
    scheduler = scheduler_cls.from_pretrained(f"{cfg_dir}/scheduler")

    print("[5/5] 组装 pipeline")
    pipeline_cls = getattr(diffusers, cfg["pipeline_cls"])
    pipe = pipeline_cls(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        **cfg["pipeline_extra"],
    )
    pipe.enable_model_cpu_offload()
    return pipe


def run_demo(
    model_key: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    out: str | None = None,
) -> dict:
    """加载固定模型并执行一次 Euler + simple 基准生成。"""
    sigmas = simple_sigmas(steps)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    pipe = load_pipeline(model_key)
    load_elapsed = time.perf_counter() - load_started

    print(
        f"生成中: model={model_key} {width}x{height} steps={steps} seed={seed} "
        "sampler=euler scheduler=simple"
    )
    generation_started = time.perf_counter()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        sigmas=sigmas,
        guidance_scale=0.0,  # Turbo 蒸馏模型关闭 CFG
        generator=torch.manual_seed(seed),
    ).images[0]
    generation_elapsed = time.perf_counter() - generation_started

    output_path = out or os.path.join("outputs", f"{model_key}_{seed}_{width}x{height}.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)

    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    if torch.cuda.is_available():
        peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"基准完成: load={load_elapsed:.1f}s generate={generation_elapsed:.1f}s "
        f"torch_cuda_peak_allocated={peak_allocated_gib:.2f}GiB "
        f"torch_cuda_peak_reserved={peak_reserved_gib:.2f}GiB"
    )
    print(f"已保存: {output_path}")
    return {
        "output_path": output_path,
        "load_seconds": load_elapsed,
        "generation_seconds": generation_elapsed,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
    }


if __name__ == "__main__":
    raise SystemExit("请使用 demo_krea2.py 或 demo_zit.py")
