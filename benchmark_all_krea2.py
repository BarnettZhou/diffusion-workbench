"""逐个隔离进程测试本机全部 Krea2 checkpoint。"""

import argparse
import ctypes
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_COMFY_ROOT = Path(r"C:\App\ComfyUI-aki-v1.6\ComfyUI")
DEFAULT_MODEL_DIR = Path(r"E:\Documents\ComfyUI\models\diffusion_models\krea2")
RUNNER_PATH = WORKSPACE_ROOT / "comfyui_krea2_runner.py"
FORMAT_CACHE_PATH = WORKSPACE_ROOT / ".cache" / "krea2_checkpoint_formats.json"
RESULT_CACHE_PATH = WORKSPACE_ROOT / ".cache" / "krea2_benchmark_results.json"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "krea2_matrix"
LOG_DIR = WORKSPACE_ROOT / ".cache" / "krea2_benchmark_logs"
RESULT_PREFIX = "KREA2_BENCHMARK_RESULT="
DEFAULT_PROMPT = (
    "A cinematic portrait photograph, natural skin texture, soft daylight, "
    "realistic details, shallow depth of field"
)


class ProcessCleanupError(RuntimeError):
    """Raised when the current model process cannot be confirmed stopped."""


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试目录中的全部 Krea2 checkpoint")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0, help="单模型超时秒数")
    parser.add_argument(
        "--min-free-ram-gib",
        type=float,
        default=0.0,
        help="可选的系统内存保护线；默认 0 表示禁用，避免 Windows 缓存造成误判",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("图片宽高必须为正数且是 16 的倍数")
    if args.steps <= 0:
        raise ValueError("steps 必须大于 0")
    if args.timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if args.min_free_ram_gib < 0:
        raise ValueError("min-free-ram-gib 不能小于 0")


def discover_checkpoints(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"找不到 Krea2 模型目录: {model_dir}")
    return sorted(model_dir.glob("*.safetensors"), key=lambda path: path.name.casefold())


def output_filename(index: int, model_path: Path) -> str:
    safe_stem = "".join(
        character if character.isalnum() or character in "-_.[]" else "_"
        for character in model_path.stem
    )
    return f"{index:02d}_{safe_stem}.png"


def build_runner_command(
    args: argparse.Namespace, model_path: Path, output_path: Path
) -> list[str]:
    comfy_root = args.comfy_root.resolve()
    comfy_python = comfy_root.parent / "python" / "python.exe"
    model_path = model_path.resolve()
    model_name = str(Path(model_path.parent.name) / model_path.name)
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
        str(output_path.resolve()),
        "--model-name",
        model_name,
        "--expected-model-path",
        str(model_path),
        "--benchmark-any-format",
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def parse_runner_result(output: str) -> dict | None:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                payload = json.loads(line.removeprefix(RESULT_PREFIX))
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def available_physical_memory_gib() -> float | None:
    if os.name != "nt":
        return None
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys / 1024**3


def terminate_process(process: subprocess.Popen) -> None:
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=10)
            return
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProcessCleanupError(
                f"无法确认 PID {process.pid} 已终止，停止后续模型测试"
            ) from exc
    except OSError as exc:
        raise ProcessCleanupError(
            f"无法确认 PID {process.pid} 已终止，停止后续模型测试"
        ) from exc


def run_checkpoint_process(
    command: list[str],
    log_path: Path,
    timeout_seconds: float,
    min_free_ram_gib: float,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    status = "error"
    stop_reason = None
    process_stopped = False

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=WORKSPACE_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    status = "timeout"
                    stop_reason = f"超过单模型超时 {timeout_seconds:.0f}s"
                    terminate_process(process)
                    process_stopped = True
                    break
                free_ram = available_physical_memory_gib() if min_free_ram_gib else None
                if free_ram is not None and free_ram < min_free_ram_gib:
                    status = "memory_guard"
                    stop_reason = (
                        f"系统可用物理内存仅 {free_ram:.2f} GiB，"
                        f"低于保护线 {min_free_ram_gib:.2f} GiB"
                    )
                    terminate_process(process)
                    process_stopped = True
                    break
                time.sleep(1)

            return_code = process.wait()
        finally:
            if not process_stopped and process.poll() is None:
                terminate_process(process)

    wall_seconds = time.monotonic() - started
    output = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_runner_result(output)
    if stop_reason is None:
        if return_code == 0 and metrics is not None:
            status = "success"
        elif return_code == 0:
            stop_reason = "runner 未输出结构化结果"
        else:
            stop_reason = f"runner 返回非零退出码 {return_code}"
    return {
        "status": status,
        "pid": process.pid,
        "return_code": return_code,
        "wall_seconds": round(wall_seconds, 3),
        "stop_reason": stop_reason,
        "metrics": metrics,
        "output": output,
    }


def condensed_error(output: str, limit: int = 6000) -> str | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    return "\n".join(lines[-30:])[-limit:]


def format_record(format_cache: dict, model_path: Path) -> dict:
    cached = format_cache.get(str(model_path.resolve()), {})
    try:
        stat = model_path.stat()
    except OSError as exc:
        return {
            "strict_fp8_valid": None,
            "detected_format": "unknown",
            "dtypes": {},
            "format_detection_error": f"无法读取模型文件: {exc}",
        }
    if cached and (
        cached.get("size") != stat.st_size or cached.get("mtime_ns") != stat.st_mtime_ns
    ):
        cached = {
            "valid": None,
            "format": "unknown",
            "dtypes": {},
            "error": "格式检测缓存已过期，请重新运行 checkpoint 检测",
        }
    return {
        "strict_fp8_valid": cached.get("valid"),
        "detected_format": cached.get("format", "unknown"),
        "dtypes": cached.get("dtypes", {}),
        "format_detection_error": cached.get("error"),
    }


def run_single_checkpoint(
    args: argparse.Namespace,
    model_path: Path,
    index: int,
    total: int,
    format_cache: dict,
) -> dict:
    output_path = OUTPUT_DIR / output_filename(index, model_path)
    log_path = LOG_DIR / f"{index:02d}_{model_path.stem}.log"
    command = build_runner_command(args, model_path, output_path)
    print(f"[{index}/{total}] 启动: {model_path.name}", flush=True)
    process_result = run_checkpoint_process(
        command,
        log_path,
        timeout_seconds=args.timeout,
        min_free_ram_gib=args.min_free_ram_gib,
    )
    status = process_result["status"]
    if status == "success" and not args.dry_run and not output_path.is_file():
        status = "error"
        process_result["stop_reason"] = "runner 成功退出但没有生成图片"
    print(
        f"[{index}/{total}] {status}: {model_path.name} "
        f"wall={process_result['wall_seconds']:.1f}s",
        flush=True,
    )
    return {
        "model": model_path.name,
        "model_path": str(model_path.resolve()),
        **format_record(format_cache, model_path),
        "status": status,
        "return_code": process_result["return_code"],
        "pid": process_result["pid"],
        "wall_seconds": process_result["wall_seconds"],
        "stop_reason": process_result["stop_reason"],
        "metrics": process_result["metrics"],
        "output_path": str(output_path.resolve()) if status == "success" else None,
        "log_path": str(log_path.resolve()),
        "error_tail": condensed_error(process_result["output"]) if status != "success" else None,
    }


def run_all(
    args: argparse.Namespace,
    checkpoints: list[Path],
    format_cache: dict,
    save_results: Callable[[list[dict]], None],
) -> list[dict]:
    results = []
    total = len(checkpoints)
    for index, model_path in enumerate(checkpoints, start=1):
        try:
            result = run_single_checkpoint(args, model_path, index, total, format_cache)
        except ProcessCleanupError:
            raise
        except Exception as exc:
            print(
                f"[{index}/{total}] orchestrator_error: {model_path.name}: {exc}",
                flush=True,
            )
            result = {
                "model": model_path.name,
                "model_path": str(model_path.resolve()),
                **format_record(format_cache, model_path),
                "status": "orchestrator_error",
                "return_code": None,
                "pid": None,
                "wall_seconds": None,
                "stop_reason": f"{type(exc).__name__}: {exc}",
                "metrics": None,
                "output_path": None,
                "log_path": None,
                "error_tail": None,
            }
        results.append(result)
        save_results(results)
    return results


def load_format_cache() -> dict:
    try:
        cache = json.loads(FORMAT_CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    checkpoints = cache.get("checkpoints", {})
    return checkpoints if isinstance(checkpoints, dict) else {}


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        checkpoints = discover_checkpoints(args.model_dir.resolve())
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"错误: {exc}\n")
    comfy_python = args.comfy_root.resolve().parent / "python" / "python.exe"
    if not comfy_python.is_file() or not args.comfy_root.is_dir():
        parser.exit(2, f"错误: 找不到 ComfyUI 或其 Python: {args.comfy_root}\n")
    if not checkpoints:
        parser.exit(2, f"错误: 目录中没有 safetensors: {args.model_dir}\n")

    format_cache = load_format_cache()
    started_at = datetime.now(timezone.utc).isoformat()

    def save_results(records: list[dict]) -> None:
        write_json_atomic(
            RESULT_CACHE_PATH,
            {
                "version": 1,
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "prompt": args.prompt,
                    "width": args.width,
                    "height": args.height,
                    "steps": args.steps,
                    "seed": args.seed,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "cfg": 1.0,
                    "timeout_seconds": args.timeout,
                    "min_free_ram_gib": args.min_free_ram_gib,
                    "dry_run": args.dry_run,
                },
                "model_count": len(checkpoints),
                "completed_count": len(records),
                "results": records,
            },
        )

    print(
        f"将顺序测试 {len(checkpoints)} 个模型: {args.width}x{args.height}, "
        f"{args.steps} steps, "
        f"euler + simple, timeout={args.timeout:.0f}s",
        flush=True,
    )
    results = run_all(args, checkpoints, format_cache, save_results)
    counts = {
        status: sum(record["status"] == status for record in results)
        for status in sorted({record["status"] for record in results})
    }
    print(f"全部测试结束: {counts}", flush=True)
    print(f"结果缓存: {RESULT_CACHE_PATH}", flush=True)


if __name__ == "__main__":
    main()
