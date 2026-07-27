"""ComfyUI-side runner for the local Z-Image Base verification demo."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path


MODEL_SPECS = {
    "base": {
        "filename": "moodyWildMixZIBZID_v40BASE40STEPSCFG4.safetensors",
        "steps": 40,
        "cfg": 4.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "sgm_uniform",
    },
    "distilled": {
        "filename": "moodyWildMixZIBZID_v40Distilled10STEPS.safetensors",
        "steps": 10,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "distilled", "all"), default="all")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distilled-cfg", type=float, default=1.0)
    parser.add_argument(
        "--distilled-sampler",
        choices=("euler", "dpmpp_2m_sde"),
        default="euler",
    )
    parser.add_argument(
        "--distilled-scheduler",
        choices=("simple", "beta"),
        default="simple",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
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


def register_exact(folder_paths, category: str, path: Path) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {category}: {path}")
    folder_paths.add_model_folder_path(category, str(path.parent), is_default=True)
    resolved = Path(folder_paths.get_full_path_or_raise(category, path.name))
    if not resolved.samefile(path):
        raise RuntimeError(f"{category} 路径解析不一致: {resolved} != {path}")
    return path.name


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("图片宽高必须为正数且是 16 的倍数")
    if not 1.0 <= args.distilled_cfg <= 1.5:
        raise ValueError("蒸馏模型 CFG 必须在 1.0 到 1.5 之间")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"找不到模型目录: {args.model_dir}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    folder_paths = initialize_comfyui(args.comfy_root)

    import torch
    from PIL import Image

    import comfy.model_management
    import comfy.samplers
    import nodes

    selected = tuple(MODEL_SPECS) if args.variant == "all" else (args.variant,)
    for variant in selected:
        spec = dict(MODEL_SPECS[variant])
        if variant == "distilled":
            spec.update(
                cfg=args.distilled_cfg,
                sampler=args.distilled_sampler,
                scheduler=args.distilled_scheduler,
            )
        if spec["sampler"] not in comfy.samplers.KSampler.SAMPLERS:
            raise RuntimeError(f"ComfyUI 不支持 sampler: {spec['sampler']}")
        if spec["scheduler"] not in comfy.samplers.KSampler.SCHEDULERS:
            raise RuntimeError(f"ComfyUI 不支持 scheduler: {spec['scheduler']}")

    text_encoder_name = register_exact(folder_paths, "text_encoders", args.text_encoder)
    vae_name = register_exact(folder_paths, "vae", args.vae)
    print(f"加载 Z-Image text encoder: {args.text_encoder.name}", flush=True)
    clip = nodes.CLIPLoader().load_clip(text_encoder_name, "stable_diffusion")[0]
    print(f"加载 Z-Image VAE: {args.vae.name}", flush=True)
    vae = nodes.VAELoader().load_vae(vae_name)[0]
    positive = nodes.CLIPTextEncode().encode(clip, args.prompt)[0]
    negative = nodes.CLIPTextEncode().encode(clip, args.negative_prompt)[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for variant in selected:
        spec = dict(MODEL_SPECS[variant])
        if variant == "distilled":
            spec.update(
                cfg=args.distilled_cfg,
                sampler=args.distilled_sampler,
                scheduler=args.distilled_scheduler,
            )
        model_path = args.model_dir / spec["filename"]
        model_name = register_exact(folder_paths, "diffusion_models", model_path)
        print(f"加载 {variant}: {model_path.name}", flush=True)
        load_started = time.perf_counter()
        model = nodes.UNETLoader().load_unet(model_name, "default")[0]
        load_seconds = time.perf_counter() - load_started
        latent = {
            "samples": torch.zeros(
                [1, 16, args.height // 8, args.width // 8],
                device=comfy.model_management.intermediate_device(),
                dtype=comfy.model_management.intermediate_dtype(),
            ),
            "downscale_ratio_spacial": 8,
        }
        torch.cuda.reset_peak_memory_stats()
        print(
            f"生成 {variant}: {args.width}x{args.height}, steps={spec['steps']}, "
            f"cfg={spec['cfg']}, {spec['sampler']} + {spec['scheduler']}",
            flush=True,
        )
        generation_started = time.perf_counter()
        samples = nodes.KSampler().sample(
            model,
            args.seed,
            spec["steps"],
            spec["cfg"],
            spec["sampler"],
            spec["scheduler"],
            positive,
            negative,
            latent,
            denoise=1.0,
        )[0]
        images = nodes.VAEDecode().decode(vae, samples)[0]
        generation_seconds = time.perf_counter() - generation_started
        pixels = images[0].detach().cpu().clamp(0, 1).mul(255).byte().numpy()
        output_path = args.out_dir / f"zib_{variant}_{args.seed}_{args.width}x{args.height}.png"
        Image.fromarray(pixels).save(output_path)
        result = {
            "variant": variant,
            "model": model_path.name,
            "output_path": str(output_path.resolve()),
            "steps": spec["steps"],
            "cfg": spec["cfg"],
            "sampler": spec["sampler"],
            "scheduler": spec["scheduler"],
            "load_seconds": round(load_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "cuda_peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "cuda_peak_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        }
        print("ZIB_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
        del model, latent, samples, images, pixels
        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache(force=True)


if __name__ == "__main__":
    main()
