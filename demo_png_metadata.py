"""Technical demo for reading generation settings embedded in a PNG."""

import argparse
import json
from pathlib import Path

from diffusion_workbench_core.png_metadata import read_generation_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 diffusion-workbench PNG 中的生成参数"
    )
    parser.add_argument("image", type=Path, help="diffusion-workbench 生成的 PNG")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = read_generation_metadata(args.image)
    if metadata is None:
        print("图片不包含 diffusion-workbench 生成参数")
        return 1
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
