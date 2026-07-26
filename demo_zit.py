"""Z-Image-Turbo 单模型文生图基准入口。"""

import argparse

from demo_txt2img import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Z-Image-Turbo 文生图 demo（Euler + simple）")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="默认 outputs/zit_<seed>_<width>x<height>.png")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_demo(
            "zit",
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            out=args.out,
        )
    except ValueError as exc:
        parser.exit(2, f"错误: {exc}\n")


if __name__ == "__main__":
    main()
