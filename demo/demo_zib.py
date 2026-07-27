"""Run the two local Z-Image Base checkpoints through ComfyUI.

Examples:
    uv run python -m demo.demo_zib
    uv run python -m demo.demo_zib --variant distilled --distilled-cfg 1.5
"""

import argparse
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMFY_ROOT = Path(r"C:\App\ComfyUI-aki-v1.6\ComfyUI")
DEFAULT_MODEL_DIR = Path(r"E:\Documents\ComfyUI\models\diffusion_models\zib")
DEFAULT_TEXT_ENCODER = Path(
    r"E:\Documents\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
)
DEFAULT_VAE = Path(r"E:\Documents\ComfyUI\models\vae\zit\zit-ae.safetensors")
RUNNER_PATH = Path(__file__).with_name("comfyui_zib_runner.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证本机 Z-Image Base checkpoint 生图")
    parser.add_argument("--variant", choices=("base", "distilled", "all"), default="all")
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic portrait photograph of a woman in a red coat, natural skin "
            "texture, soft window light, realistic details, shallow depth of field"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted anatomy, extra fingers, text, watermark",
    )
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
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "zib")
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-encoder", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    return parser


def build_runner_command(args: argparse.Namespace, variant: str) -> list[str]:
    comfy_root = args.comfy_root.resolve()
    comfy_python = comfy_root.parent / "python" / "python.exe"
    return [
        str(comfy_python),
        "-u",
        str(RUNNER_PATH),
        "--comfy-root",
        str(comfy_root),
        "--variant",
        variant,
        "--model-dir",
        str(args.model_dir.resolve()),
        "--text-encoder",
        str(args.text_encoder.resolve()),
        "--vae",
        str(args.vae.resolve()),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--seed",
        str(args.seed),
        "--distilled-cfg",
        str(args.distilled_cfg),
        "--distilled-sampler",
        args.distilled_sampler,
        "--distilled-scheduler",
        args.distilled_scheduler,
        "--out-dir",
        str(args.out_dir.resolve()),
    ]


def main() -> None:
    args = build_parser().parse_args()
    variants = ("base", "distilled") if args.variant == "all" else (args.variant,)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for variant in variants:
        command = build_runner_command(args, variant)
        if not Path(command[0]).is_file():
            raise SystemExit(f"找不到 ComfyUI Python: {command[0]}")
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
