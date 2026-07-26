"""研究 Krea2 checkpoint 在非 ComfyUI 后端中的可行性。

默认只读取 safetensors 头部。传入 --kernel-probe 时，会把一个 scaled-FP8
线性层加载到 CUDA，并用 PyTorch torch._scaled_mm 与 BF16 参考值比较。
此脚本不会实例化完整 Krea2 模型。
"""

import argparse
import importlib.util
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from krea2_config import KREA2_MODEL_PATH


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = KREA2_MODEL_PATH.parent
DEFAULT_CACHE_PATH = WORKSPACE_ROOT / ".cache" / "krea2_alternative_backend_probe.json"
CACHE_VERSION = 1
QUANT_SUFFIXES = ("_scale", ".comfy_quant")
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查 Krea2 的非 ComfyUI 加载路线；默认不加载完整模型"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=KREA2_MODEL_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--kernel-probe",
        action="store_true",
        help="在一个真实 scaled-FP8 线性层上测试 torch._scaled_mm",
    )
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--refresh", action="store_true", help="忽略已有头部扫描缓存")
    return parser


def decode_quant_metadata(handle, key: str) -> dict | None:
    try:
        payload = handle.get_tensor(key).numpy().tobytes().decode("utf-8")
        metadata = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def tensor_numel(shape: list[int]) -> int:
    value = 1
    for dimension in shape:
        value *= dimension
    return value


def inspect_checkpoint(path: Path) -> dict:
    path = path.resolve()
    stat = path.stat()
    dtype_counts = Counter()
    base_dtypes = Counter()
    quant_formats = Counter()
    quant_metadata_count = 0
    quant_metadata_errors = 0
    base_numel = 0
    stored_base_bytes = 0
    fp8_weights = 0
    scaled_fp8_weights = 0
    int8_weights = 0
    requires_convrot_kernel = False
    has_comfy_prefix = False

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in keys:
            tensor_slice = handle.get_slice(key)
            dtype = tensor_slice.get_dtype()
            shape = tensor_slice.get_shape()
            dtype_counts[dtype] += 1

            if key.endswith(".comfy_quant"):
                quant_metadata_count += 1
                metadata = decode_quant_metadata(handle, key)
                if metadata is None:
                    quant_metadata_errors += 1
                    continue
                quant_format = str(metadata.get("format", "unknown"))
                quant_formats[quant_format] += 1
                params = metadata.get("params", {})
                params = params if isinstance(params, dict) else {}
                if metadata.get("convrot", params.get("convrot", False)):
                    requires_convrot_kernel = True
                continue
            if key.endswith("_scale"):
                continue

            has_comfy_prefix = has_comfy_prefix or key.startswith("model.diffusion_model.")
            base_dtypes[dtype] += 1
            numel = tensor_numel(shape)
            base_numel += numel
            stored_base_bytes += numel * DTYPE_BYTES.get(dtype, 0)
            if dtype in {"F8_E4M3", "F8_E5M2"}:
                fp8_weights += 1
                if key + "_scale" in keys:
                    scaled_fp8_weights += 1
            elif dtype == "I8":
                int8_weights += 1

    native_fp8_candidate = (
        fp8_weights > 0
        and int8_weights == 0
        and not requires_convrot_kernel
        and quant_metadata_errors == 0
        and set(quant_formats).issubset({"float8_e4m3fn", "float8_e5m2"})
    )
    diffusers_direct_candidate = quant_metadata_count == 0 and not has_comfy_prefix
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "file_size_gib": round(stat.st_size / 1024**3, 3),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "base_dtypes": dict(sorted(base_dtypes.items())),
        "quant_formats": dict(sorted(quant_formats.items())),
        "quant_metadata_count": quant_metadata_count,
        "quant_metadata_errors": quant_metadata_errors,
        "fp8_weights": fp8_weights,
        "scaled_fp8_weights": scaled_fp8_weights,
        "int8_weights": int8_weights,
        "requires_convrot_kernel": requires_convrot_kernel,
        "stored_base_bytes": stored_base_bytes,
        "bf16_expanded_bytes": base_numel * 2,
        "bf16_expanded_gib": round(base_numel * 2 / 1024**3, 3),
        "has_comfy_prefix": has_comfy_prefix,
        "diffusers_direct_candidate": diffusers_direct_candidate,
        "native_fp8_candidate": native_fp8_candidate,
    }


def cache_matches(path: Path, cached: dict) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns


def load_cache(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"version": CACHE_VERSION, "checkpoints": {}}
    if payload.get("version") != CACHE_VERSION or not isinstance(payload.get("checkpoints"), dict):
        return {"version": CACHE_VERSION, "checkpoints": {}}
    return payload


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def choose_scaled_fp8_layer(path: Path) -> str:
    candidates = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in keys:
            tensor_slice = handle.get_slice(key)
            shape = tensor_slice.get_shape()
            if (
                tensor_slice.get_dtype() == "F8_E4M3"
                and len(shape) == 2
                and key + "_scale" in keys
                and not handle.get_slice(key + "_scale").get_shape()
            ):
                candidates.append((tensor_numel(shape), key))
    if not candidates:
        raise ValueError("checkpoint 中没有带标量 scale 的二维 FP8 E4M3 权重")
    return min(candidates)[1]


def run_native_fp8_kernel_probe(
    checkpoint: Path, rows: int = 64, iterations: int = 20
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if rows <= 0 or rows % 16:
        raise ValueError("rows 必须为正数且是 16 的倍数")
    if iterations <= 0:
        raise ValueError("iterations 必须大于 0")

    checkpoint = checkpoint.resolve()
    key = choose_scaled_fp8_layer(checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key).cuda()
        weight_scale = handle.get_tensor(key + "_scale").cuda().float()

    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    inputs = torch.randn(rows, weight.shape[1], device="cuda", dtype=torch.bfloat16)
    input_scale = (inputs.abs().amax().float() / 448.0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    inputs_fp8 = (inputs.float() / input_scale).clamp(-448, 448).to(
        torch.float8_e4m3fn
    )

    # _scaled_mm requires the second matrix in column-major view; do not call contiguous().
    weight_transposed = weight.T
    for _ in range(3):
        output = torch._scaled_mm(
            inputs_fp8,
            weight_transposed,
            input_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        output = torch._scaled_mm(
            inputs_fp8,
            weight_transposed,
            input_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
        )
    torch.cuda.synchronize()
    average_ms = (time.perf_counter() - started) * 1000 / iterations

    reference = F.linear(inputs, weight.float().mul(weight_scale).to(torch.bfloat16))
    relative_mae = (output.float() - reference.float()).abs().mean() / reference.float().abs().mean().clamp_min(1e-8)
    return {
        "status": "success",
        "checkpoint": str(checkpoint),
        "size": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
        "layer": key,
        "weight_shape": list(weight.shape),
        "rows": rows,
        "iterations": iterations,
        "average_ms": round(average_ms, 4),
        "relative_mae": round(relative_mae.item(), 6),
        "output_finite": bool(torch.isfinite(output).all()),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def candidate_routes() -> list[dict]:
    return [
        {
            "route": "official_pytorch_bf16",
            "full_inference_available": True,
            "accepts_current_comfy_checkpoints_directly": False,
            "note": "官方仓库可完全脱离 ComfyUI，但要求官方普通 checkpoint，并将 DiT 转为 BF16。",
        },
        {
            "route": "diffusers_plus_torchao",
            "full_inference_available": package_available("torchao"),
            "accepts_current_comfy_checkpoints_directly": False,
            "note": "可从普通 BF16/diffusers 权重重新量化；不能直接解释 comfy_quant 或 INT8 ConvRot。",
        },
        {
            "route": "custom_native_pytorch_fp8",
            "full_inference_available": False,
            "accepts_current_comfy_checkpoints_directly": "scaled-FP8 only after key mapping",
            "note": "torch._scaled_mm 可运行真实权重，但仍需完成模型模块替换、流式加载和 offload。",
        },
        {
            "route": "custom_int8_convrot",
            "full_inference_available": False,
            "accepts_current_comfy_checkpoints_directly": False,
            "note": "需要复刻 ConvRot 预处理与专用 kernel；普通 INT8 Linear 不等价。",
        },
    ]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    if not model_dir.is_dir():
        parser.exit(2, f"错误: 找不到模型目录: {model_dir}\n")
    if not checkpoint.is_file():
        parser.exit(2, f"错误: 找不到 kernel probe checkpoint: {checkpoint}\n")

    cache = load_cache(args.cache)
    reports = {}
    for model_path in sorted(model_dir.glob("*.safetensors"), key=lambda item: item.name.casefold()):
        cache_key = str(model_path.resolve())
        cached = cache["checkpoints"].get(cache_key)
        if not args.refresh and cached and cache_matches(model_path, cached):
            report = {**cached, "cache_hit": True}
        else:
            report = {**inspect_checkpoint(model_path), "cache_hit": False}
        reports[cache_key] = report
        print(
            f"{model_path.name}: dtypes={report['base_dtypes']} "
            f"quant={report['quant_formats']} bf16_expand={report['bf16_expanded_gib']:.2f}GiB "
            f"native_fp8_candidate={report['native_fp8_candidate']}",
            flush=True,
        )

    cached_kernel_probe = cache.get("kernel_probe")
    kernel_probe = (
        cached_kernel_probe
        if isinstance(cached_kernel_probe, dict)
        and cached_kernel_probe.get("checkpoint") == str(checkpoint)
        and cache_matches(checkpoint, cached_kernel_probe)
        else None
    )
    if args.kernel_probe:
        print(f"运行单层原生 PyTorch FP8 kernel probe: {checkpoint.name}", flush=True)
        try:
            kernel_probe = run_native_fp8_kernel_probe(
                checkpoint, rows=args.rows, iterations=args.iterations
            )
        except Exception as exc:
            kernel_probe = {
                "status": "error",
                "checkpoint": str(checkpoint),
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(kernel_probe, ensure_ascii=False, indent=2), flush=True)

    payload = {
        "version": CACHE_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "packages": {
                name: package_available(name)
                for name in ("torchao", "bitsandbytes", "optimum", "triton")
            },
        },
        "routes": candidate_routes(),
        "checkpoints": reports,
        "kernel_probe": kernel_probe,
    }
    write_json_atomic(args.cache, payload)
    print(f"研究缓存: {args.cache.resolve()}", flush=True)


if __name__ == "__main__":
    main()
