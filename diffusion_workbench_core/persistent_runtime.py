import atexit
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from .config import WorkbenchConfig
from .domain import JobRecord
from .runtime import GenerationCancelled


EVENT_PREFIX = "DWB_EVENT="
DEFAULT_WORKER_SCRIPT = Path(__file__).resolve().parent / "comfy_worker.py"


class PersistentComfyRuntime:
    def __init__(
        self, config: WorkbenchConfig, worker_script: Path = DEFAULT_WORKER_SCRIPT
    ):
        self.config = config
        self.worker_script = Path(worker_script).resolve()
        self._process: subprocess.Popen | None = None
        self._events: queue.Queue[dict] = queue.Queue()
        self._process_lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._logs: deque[str] = deque(maxlen=100)
        self._loaded_model: str | None = None
        self._gpu_value = "查询中"
        self._gpu_probe_running = False
        self._gpu_probe_time = 0.0
        self._closed = False
        atexit.register(self.close)
        self._request_gpu_probe()

    def generate(self, job: JobRecord, progress, stage) -> dict:
        with self._generation_lock:
            if self._closed:
                raise RuntimeError("GPU Worker 已关闭")
            self._cancelled.clear()
            process = self._ensure_process()
            command = {
                "type": "generate",
                "job_id": job.id,
                "mode": job.mode.value,
                "model_path": str(job.model_path.resolve()),
                "vae_path": str(job.vae_path.resolve()),
                "text_encoder_path": str(job.text_encoder_path.resolve()),
                "clip_type": self.config.resources[job.mode].clip_type,
                "prompt": job.prompt,
                "width": job.width,
                "height": job.height,
                "steps": job.steps,
                "seed": job.seed,
                "cfg": job.cfg,
                "sampler": job.sampler,
                "scheduler": job.scheduler,
                "output_path": str(job.output_path.resolve()),
            }
            self._write_command(process, command)
            deadline = time.monotonic() + self.config.worker_timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.cancel()
                    raise TimeoutError(
                        f"任务超过 Worker 超时 {self.config.worker_timeout_seconds:.0f}s"
                    )
                try:
                    event = self._events.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if process.poll() is not None:
                        self._raise_worker_exit(process)
                    continue
                event_type = event.get("type")
                if event_type == "step_progress" and event.get("job_id") == job.id:
                    progress(int(event["step"]), int(event["total"]))
                elif event_type == "stage_progress" and event.get("job_id") == job.id:
                    total = event.get("total")
                    stage(str(event["stage"]), int(total) if total is not None else None)
                elif event_type == "result" and event.get("job_id") == job.id:
                    self._loaded_model = event.get("model_path", command["model_path"])
                    return event
                elif event_type == "error" and event.get("job_id") == job.id:
                    error = event.get("error", "GPU Worker 生成失败")
                    traceback_text = event.get("traceback")
                    raise RuntimeError(
                        error + (f"\n{traceback_text}" if traceback_text else "")
                    )
                elif event_type == "process_eof":
                    self._raise_worker_exit(process)

    def cancel(self) -> None:
        self._cancelled.set()
        self._terminate_process()

    def close(self) -> None:
        with self._process_lock:
            if self._closed and self._process is None:
                return
            self._closed = True
            process = self._process
        if process is not None and process.poll() is None:
            try:
                self._write_command(process, {"type": "shutdown"})
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._terminate_process()
        if process is not None:
            self._close_pipes(process)
        with self._process_lock:
            self._process = None
            self._loaded_model = None

    def status(self) -> dict:
        self._request_gpu_probe()
        with self._process_lock:
            process = self._process
            running = process is not None and process.poll() is None
            pid = process.pid if running else None
        return {
            "worker": "ready" if running else "stopped",
            "pid": pid,
            "loaded_model": self._loaded_model,
            "gpu": self._gpu_value,
        }

    def _request_gpu_probe(self) -> None:
        with self._process_lock:
            if (
                self._closed
                or self._gpu_probe_running
                or time.monotonic() - self._gpu_probe_time < 2
            ):
                return
            self._gpu_probe_running = True
        threading.Thread(
            target=self._refresh_gpu_memory,
            name="gpu-memory-probe",
            daemon=True,
        ).start()

    def _refresh_gpu_memory(self) -> None:
        try:
            value = self._gpu_memory()
        except Exception:
            value = "不可用"
        finally:
            with self._process_lock:
                self._gpu_value = value
                self._gpu_probe_time = time.monotonic()
                self._gpu_probe_running = False

    def _ensure_process(self) -> subprocess.Popen:
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                return self._process
            if not self.config.comfyui.python.is_file():
                raise FileNotFoundError(f"找不到 ComfyUI Python: {self.config.comfyui.python}")
            if not self.config.comfyui.root.is_dir():
                raise FileNotFoundError(f"找不到 ComfyUI root: {self.config.comfyui.root}")
            if not self.worker_script.is_file():
                raise FileNotFoundError(f"找不到 GPU Worker: {self.worker_script}")
            self._logs.clear()
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                }
            )
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(
                [
                    str(self.config.comfyui.python),
                    "-u",
                    str(self.worker_script),
                    "--comfy-root",
                    str(self.config.comfyui.root),
                ],
                cwd=self.config.path.parent,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self._events = queue.Queue()
            self._process = process
            threading.Thread(
                target=self._read_output,
                args=(process, self._events),
                name=f"comfy-worker-{process.pid}",
                daemon=True,
            ).start()

        startup_deadline = time.monotonic() + min(
            self.config.worker_timeout_seconds, 60
        )
        while True:
            remaining = startup_deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_process()
                raise TimeoutError("GPU Worker 启动超时")
            try:
                event = self._events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if process.poll() is not None:
                    self._raise_worker_exit(process)
                continue
            if event.get("type") == "ready":
                return process
            if event.get("type") == "startup_error":
                self._terminate_process()
                raise RuntimeError(event.get("error", "GPU Worker 初始化失败"))
            if event.get("type") == "process_eof":
                self._raise_worker_exit(process)

    def _read_output(
        self, process: subprocess.Popen, events: queue.Queue[dict]
    ) -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if line.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(line.removeprefix(EVENT_PREFIX))
                except json.JSONDecodeError:
                    self._logs.append(line)
                    continue
                if isinstance(event, dict):
                    events.put(event)
            elif line:
                self._logs.append(line)
        events.put({"type": "process_eof", "return_code": process.poll()})

    @staticmethod
    def _write_command(process: subprocess.Popen, command: dict) -> None:
        if process.stdin is None or process.poll() is not None:
            raise RuntimeError("GPU Worker 不可写")
        process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _terminate_process(self) -> None:
        with self._process_lock:
            process = self._process
            if process is None:
                return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        self._close_pipes(process)
        with self._process_lock:
            if self._process is process:
                self._process = None
                self._loaded_model = None

    def _raise_worker_exit(self, process: subprocess.Popen) -> None:
        self._close_pipes(process)
        with self._process_lock:
            if self._process is process:
                self._process = None
                self._loaded_model = None
        if self._cancelled.is_set():
            raise GenerationCancelled("任务已取消，GPU Worker 已重启")
        log_tail = "\n".join(list(self._logs)[-20:])
        raise RuntimeError(
            f"GPU Worker 异常退出，code={process.poll()}"
            + (f"\n{log_tail}" if log_tail else "")
        )

    @staticmethod
    def _gpu_memory() -> str:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if result.returncode == 0:
                used, total = result.stdout.strip().split(",", 1)
                return f"{int(used.strip()) / 1024:.2f}/{int(total.strip()) / 1024:.2f} GiB"
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        return "不可用"

    @staticmethod
    def _close_pipes(process: subprocess.Popen) -> None:
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass
