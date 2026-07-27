"""Krea 2 单模型入口：调用本机 ComfyUI 的原生量化后端。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.demo_txt2img import validate_krea2_scaled_fp8
from krea2_config import KREA2_MODEL_PATH


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMFY_ROOT = Path(r"C:\App\ComfyUI-aki-v1.6\ComfyUI")
RUNNER_PATH = WORKSPACE_ROOT / "comfyui_krea2_runner.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Krea 2 文生图 demo（Euler + simple）")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="默认 outputs/krea2_<seed>_<width>x<height>.png")
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="只检查 ComfyUI 后端和模型路径")
    return parser


def build_runner_command(args: argparse.Namespace) -> list[str]:
    comfy_root = args.comfy_root.resolve()
    comfy_python = comfy_root.parent / "python" / "python.exe"
    output_path = Path(
        args.out or WORKSPACE_ROOT / "outputs" / f"krea2_{args.seed}_{args.width}x{args.height}.png"
    ).resolve()
    command = [
        str(comfy_python),
        "-u",
        str(RUNNER_PATH),
        "--comfy-root",
        str(comfy_root),
        "--prompt",
        args.prompt,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--out",
        str(output_path),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_krea2_scaled_fp8(str(KREA2_MODEL_PATH))
    except ValueError as exc:
        parser.exit(2, f"错误: {exc}\n")

    command = build_runner_command(args)
    if not Path(command[0]).is_file():
        parser.exit(2, f"错误: 找不到 ComfyUI Python: {command[0]}\n")
    if not args.comfy_root.is_dir():
        parser.exit(2, f"错误: 找不到 ComfyUI 根目录: {args.comfy_root}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=WORKSPACE_ROOT, env=env, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
