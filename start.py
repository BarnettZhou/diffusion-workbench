#!/usr/bin/env python3
"""一键启动:构建前端(有变更时)并用 uv 启动 FastAPI 服务。

用法:
    python start.py [--host 127.0.0.1] [--port 8188]

前端源码无变更时跳过 npm 构建(以 dist/.build-hash 判断)。
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"
HASH_FILE = DIST / ".build-hash"
# 参与变更检测的前端输入
SOURCE_PATHS = ("src", "index.html", "package.json", "package-lock.json", "vite.config.js")


def source_hash() -> str:
    digest = hashlib.sha256()
    files = []
    for name in SOURCE_PATHS:
        path = FRONTEND / name
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.is_file())
        elif path.is_file():
            files.append(path)
    for path in files:
        digest.update(str(path.relative_to(FRONTEND)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_needed() -> bool:
    if not DIST.is_dir() or not HASH_FILE.is_file():
        return True
    return HASH_FILE.read_text(encoding="utf-8").strip() != source_hash()


def run(command: str, cwd: Path) -> None:
    print(f"+ {command}", flush=True)
    try:
        subprocess.run(command, cwd=cwd, check=True, shell=True)
    except KeyboardInterrupt:
        # CTRL+C 会同时发给子进程(uvicorn 自行优雅退出),脚本静默结束
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    args = parser.parse_args()

    if not (FRONTEND / "node_modules").is_dir():
        run("npm install", FRONTEND)
    if build_needed():
        run("npm run build", FRONTEND)
        HASH_FILE.write_text(source_hash(), encoding="utf-8")
    else:
        print("前端无变更,跳过构建", flush=True)

    run(
        f"uv run uvicorn diffusion_workbench_api.app:app "
        f"--host {args.host} --port {args.port} --workers 1 "
        f"--timeout-graceful-shutdown 3",
        ROOT,
    )


if __name__ == "__main__":
    sys.exit(main())
