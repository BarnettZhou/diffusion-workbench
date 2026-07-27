"""在 ComfyUI 自带 Python 中执行无界面的 Krea2 文生图。"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from krea2_config import (
    KREA2_MODEL_NAME,
    KREA2_MODEL_PATH,
    KREA2_TEXT_ENCODER_NAME,
    KREA2_VAE_NAME,
)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", default=KREA2_MODEL_NAME)
    parser.add_argument("--expected-model-path", type=Path, default=KREA2_MODEL_PATH)
    parser.add_argument(
        "--benchmark-any-format",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def initialize_comfyui(comfy_root: Path):
    comfy_root = comfy_root.resolve()
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import comfy.options

    comfy.options.args_parsing = False

    import folder_paths
    from utils.extra_config import load_extra_path_config

    load_extra_path_config(str(comfy_root / "extra_model_paths.yaml"))
    return folder_paths


def require_model(folder_paths, category: str, name: str) -> None:
    if name not in folder_paths.get_filename_list(category):
        raise FileNotFoundError(f"ComfyUI {category} 中找不到: {name}")


def validate_generation_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("steps 必须大于 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("图片宽高必须大于 0")
    if args.width % 16 or args.height % 16:
        raise ValueError("Krea2 图片宽高必须是 16 的倍数")


def main() -> None:
    args = build_parser().parse_args()
    validate_generation_args(args)
    output_path = args.out.resolve()
    folder_paths = initialize_comfyui(args.comfy_root)

    require_model(folder_paths, "diffusion_models", args.model_name)
    require_model(folder_paths, "text_encoders", KREA2_TEXT_ENCODER_NAME)
    require_model(folder_paths, "vae", KREA2_VAE_NAME)
    resolved_model_path = Path(
        folder_paths.get_full_path_or_raise("diffusion_models", args.model_name)
    )
    expected_model_path = args.expected_model_path.resolve()
    if not expected_model_path.is_file() or not resolved_model_path.samefile(expected_model_path):
        raise RuntimeError(
            "ComfyUI 解析到的 Krea2 文件与请求的精确文件不一致: "
            f"resolved={resolved_model_path} expected={expected_model_path}"
        )
    if not args.benchmark_any_format:
        from demo.demo_txt2img import validate_krea2_scaled_fp8

        validate_krea2_scaled_fp8(str(resolved_model_path))

    logging.getLogger("xformers").setLevel(logging.ERROR)
    import comfy.samplers

    if "euler" not in comfy.samplers.KSampler.SAMPLERS:
        raise RuntimeError("当前 ComfyUI 不支持 euler sampler")
    if "simple" not in comfy.samplers.KSampler.SCHEDULERS:
        raise RuntimeError("当前 ComfyUI 不支持 simple scheduler")

    if args.dry_run:
        print(
            f"ComfyUI 后端检查通过: model={args.model_name} "
            f"text_encoder={KREA2_TEXT_ENCODER_NAME} vae={KREA2_VAE_NAME} "
            f"sampler=euler scheduler=simple {args.width}x{args.height}"
        )
        print(
            "KREA2_BENCHMARK_RESULT="
            + json.dumps(
                {
                    "model": args.model_name,
                    "dry_run": True,
                    "output_path": str(output_path),
                },
                ensure_ascii=False,
            )
        )
        return

    import torch
    from PIL import Image

    import comfy.model_management
    import nodes

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    print(f"[1/3] ComfyUI 加载 Krea2 checkpoint: {args.model_name}")
    model = nodes.UNETLoader().load_unet(args.model_name, "default")[0]
    print(f"[2/3] ComfyUI 加载 Krea2 text encoder: {KREA2_TEXT_ENCODER_NAME}")
    clip = nodes.CLIPLoader().load_clip(KREA2_TEXT_ENCODER_NAME, type="krea2")[0]
    print(f"[3/3] ComfyUI 加载 VAE: {KREA2_VAE_NAME}")
    vae = nodes.VAELoader().load_vae(KREA2_VAE_NAME)[0]

    positive = nodes.CLIPTextEncode().encode(clip, args.prompt)[0]
    negative = positive  # cfg=1.0 不执行 classifier-free guidance
    del clip
    comfy.model_management.soft_empty_cache()
    load_elapsed = time.perf_counter() - load_started

    latent = {
        "samples": torch.zeros(
            [1, 16, args.height // 8, args.width // 8],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        ),
        "downscale_ratio_spacial": 8,
    }
    print(
        f"生成中: backend=comfyui model=krea2 {args.width}x{args.height} steps={args.steps} "
        f"seed={args.seed} sampler=euler scheduler=simple cfg=1.0"
    )
    generation_started = time.perf_counter()
    samples = nodes.KSampler().sample(
        model,
        args.seed,
        args.steps,
        1.0,
        "euler",
        "simple",
        positive,
        negative,
        latent,
        denoise=1.0,
    )[0]
    images = nodes.VAEDecode().decode(vae, samples)[0]
    generation_elapsed = time.perf_counter() - generation_started

    pixels = images[0].detach().cpu().clamp(0, 1).mul(255).byte().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"基准完成: load={load_elapsed:.1f}s generate={generation_elapsed:.1f}s "
        f"torch_cuda_peak_allocated={peak_allocated:.2f}GiB "
        f"torch_cuda_peak_reserved={peak_reserved:.2f}GiB"
    )
    print(f"已保存: {output_path}")
    print(
        "KREA2_BENCHMARK_RESULT="
        + json.dumps(
            {
                "model": args.model_name,
                "load_seconds": round(load_elapsed, 3),
                "generation_seconds": round(generation_elapsed, 3),
                "cuda_peak_allocated_gib": round(peak_allocated, 3),
                "cuda_peak_reserved_gib": round(peak_reserved, 3),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
