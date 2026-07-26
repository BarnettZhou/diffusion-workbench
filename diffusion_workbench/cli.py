import argparse
from dataclasses import replace
from pathlib import Path

from diffusion_workbench_core import WorkbenchCore
from diffusion_workbench_core.config import load_config

from .tui import WorkbenchApp


PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "workbench.yaml"
PACKAGED_CONFIG = Path(__file__).resolve().with_name("workbench.yaml")
DEFAULT_CONFIG = PROJECT_CONFIG if PROJECT_CONFIG.is_file() else PACKAGED_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffusion-workbench")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.config.resolve() == PACKAGED_CONFIG:
        config = replace(
            config,
            output_dir=(Path.cwd() / "output").resolve(),
            database=(Path.cwd() / ".cache" / "diffusion_workbench.sqlite3").resolve(),
        )
    core = WorkbenchCore(config)
    try:
        WorkbenchApp(core).run()
    finally:
        core.shutdown()


if __name__ == "__main__":
    main()
