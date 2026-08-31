import atexit
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil
from uuid import uuid4

from .config import WorkbenchConfig
from .domain import JobRecord, Mode, VideoJobRecord, resolve_video_diffusion_pair
from .logging_config import silent_logger, worker_output_level
from .runtime import GenerationCancelled


EVENT_PREFIX = "DWB_EVENT="
DEFAULT_WORKER_SCRIPT = Path(__file__).resolve().parent / "comfy_worker.py"
WORKER_EVENT_REQUIRED_FIELDS = {
    "ready": (),
    "startup_error": (),
    "stage_progress": ("job_id", "stage"),
    "step_progress": ("job_id", "step", "total"),
    "preview_image": ("job_id", "data"),
    "result": ("job_id",),
    "cancelled": ("job_id",),
    "error": ("job_id",),
    "released": (),
    "stopped": (),
}


def _workbench_version() -> str:
    try:
        return version("diffusion-workbench")
    except PackageNotFoundError:
        return "development"


class PersistentComfyRuntime:
    def __init__(
        self,
        config: WorkbenchConfig,
        worker_script: Path = DEFAULT_WORKER_SCRIPT,
        logger=None,
    ):
        self.config = config
        self.worker_script = Path(worker_script).resolve()
        self._logger = logger or silent_logger()
        self._process: subprocess.Popen | None = None
        self._events: queue.Queue[dict] = queue.Queue()
        self._process_lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._logs: deque[str] = deque(maxlen=100)
        self._reader_thread: threading.Thread | None = None
        self._loaded_model: str | None = None
        self._loaded_resources: dict = {}
        self._active_job_id: str | None = None
        self._preview_enabled = False
        self._gpu_value = "查询中"
        self._memory_status: dict = {
            "used_gib": None,
            "total_gib": None,
            "percent": None,
        }
        self._gpu_memory_used_gib: float | None = None
        self._gpu_memory_total_gib: float | None = None
        self._gpu_memory_percent: float | None = None
        self._gpu_utilization_percent: float | None = None
        self._gpu_probe_running = False
        self._gpu_probe_time = 0.0
        self._closed = False
        atexit.register(self.close)
        self._request_gpu_probe()

    def generate(
        self, job: JobRecord | VideoJobRecord, progress, stage, preview=None
    ) -> dict:
        with self._generation_lock:
            if self._closed:
                raise RuntimeError("GPU Worker 已关闭")
            if self._cancelled.is_set():
                self._cancelled.clear()
                raise GenerationCancelled("任务在 Worker 启动前已取消")
            try:
                return self._generate(job, progress, stage, preview)
            finally:
                with self._process_lock:
                    self._active_job_id = None
                self._cancelled.clear()

    def _generate(
        self, job: JobRecord | VideoJobRecord, progress, stage, preview=None
    ) -> dict:
        stage("starting_worker", job.steps)
        process = self._ensure_process()
        if self._cancelled.is_set():
            raise GenerationCancelled("任务在 Worker 启动期间已取消")
        if isinstance(job, VideoJobRecord):
            command = self._video_command(job)
            timeout_seconds = self.config.video_worker_timeout_seconds
        else:
            command = self._image_command(job)
            timeout_seconds = self.config.worker_timeout_seconds
        with self._process_lock:
            if self._cancelled.is_set():
                raise GenerationCancelled("任务在 Worker 启动期间已取消")
            self._active_job_id = job.id
            try:
                self._write_command(process, command)
            except Exception:
                self._active_job_id = None
                raise
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._cancelled.set()
                self._logger.warning("worker generation timed out; terminating process")
                self._terminate_process()
                raise TimeoutError(f"任务超过 Worker 超时 {timeout_seconds:.0f}s")
            try:
                event = self._events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if process.poll() is not None:
                    self._raise_worker_exit(process)
                continue
            event_type = event.get("type")
            if event_type == "step_progress" and event.get("job_id") == job.id:
                metrics = {
                    key: event.get(key)
                    for key in (
                        "elapsed_seconds",
                        "step_seconds",
                        "seconds_per_step",
                        "steps_per_second",
                        "eta_seconds",
                        "stage",
                    )
                }
                progress(int(event["step"]), int(event["total"]), metrics)
            elif event_type == "preview_image" and event.get("job_id") == job.id:
                if preview is not None:
                    preview(event)
            elif event_type == "stage_progress" and event.get("job_id") == job.id:
                total = event.get("total")
                stage(str(event["stage"]), int(total) if total is not None else None)
            elif event_type == "result" and event.get("job_id") == job.id:
                self._loaded_model = event.get("model_path", command["model_path"])
                self._loaded_resources = event.get("loaded_resources") or {
                    "model": self._loaded_model
                }
                return event
            elif event_type == "cancelled" and event.get("job_id") == job.id:
                if event.get("resources_released"):
                    self._loaded_model = None
                    self._loaded_resources = {}
                elif event.get("model_path"):
                    self._loaded_model = event["model_path"]
                raise GenerationCancelled("任务已取消")
            elif event_type == "error" and event.get("job_id") == job.id:
                self._loaded_model = None
                self._loaded_resources = {}
                error = event.get("error", "GPU Worker 生成失败")
                traceback_text = event.get("traceback")
                raise RuntimeError(
                    error + (f"\n{traceback_text}" if traceback_text else "")
                )
            elif event_type == "process_eof":
                self._raise_worker_exit(process)

    def describe_image(
        self,
        *,
        text_encoder_path: Path,
        image_path: Path,
        prompt: str,
        max_length: int,
        seed: int,
    ) -> dict:
        """同步图片反推：与 generate 共用 _generation_lock 串行排队，结果不落库、不发事件。

        stop/skip 不作用于反推（不登记 _active_job_id）；与生成/释放的互斥
        完全由 _generation_lock 保证。
        """
        with self._generation_lock:
            if self._closed:
                raise RuntimeError("GPU Worker 已关闭")
            process = self._ensure_process()
            job_id = uuid4().hex
            command = {
                "type": "describe_image",
                "job_id": job_id,
                "text_encoder_path": str(Path(text_encoder_path).resolve()),
                "image_path": str(Path(image_path).resolve()),
                "prompt": prompt,
                "max_length": int(max_length),
                "seed": int(seed),
            }
            self._write_command(process, command)
            deadline = time.monotonic() + self.config.worker_timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._logger.warning(
                        "worker describe_image timed out; terminating process"
                    )
                    self._terminate_process()
                    raise TimeoutError(
                        f"反推超过 Worker 超时 {self.config.worker_timeout_seconds:.0f}s"
                    )
                try:
                    event = self._events.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if process.poll() is not None:
                        self._raise_worker_exit(process)
                    continue
                event_type = event.get("type")
                if event_type == "result" and event.get("job_id") == job_id:
                    loaded = event.get("loaded_resources")
                    if loaded:
                        self._loaded_resources = loaded
                    return event
                if event_type == "error" and event.get("job_id") == job_id:
                    error = event.get("error", "GPU Worker 反推失败")
                    traceback_text = event.get("traceback")
                    raise RuntimeError(
                        error + (f"\n{traceback_text}" if traceback_text else "")
                    )
                if event_type == "process_eof":
                    self._raise_worker_exit(process)

    def _image_command(self, job: JobRecord) -> dict:
        # 编辑/参考图重排模式复用 krea2 的资源：clip_type 等仍按 krea2 配置读取，但 mode 字段
        # 保持原值，让 Worker 走各自的节点分支。
        clip_lookup_mode = (
            Mode.KREA2
            if job.mode in (Mode.KREA2_EDIT, Mode.KREA2_REBALANCE)
            else job.mode
        )
        is_edit = job.mode == Mode.KREA2_EDIT
        krea2_resources = (
            self.config.resources.get(Mode.KREA2) if is_edit else None
        )
        edit_lora_path = (
            str(krea2_resources.edit_lora.resolve())
            if is_edit and krea2_resources and krea2_resources.edit_lora
            else None
        )
        return {
            "type": "generate",
            "job_id": job.id,
            "batch_id": job.batch_id,
            "mode": job.mode.value,
            "model_loader": job.model_loader.value,
            "model_path": str(job.model_path.resolve()),
            "vae_path": str(job.vae_path.resolve()) if job.vae_path else None,
            "text_encoder_path": (
                str(job.text_encoder_path.resolve()) if job.text_encoder_path else None
            ),
            "clip_type": self.config.resources[clip_lookup_mode].clip_type,
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt,
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "seed": job.seed,
            "cfg": job.cfg,
            "sampler": job.sampler,
            "scheduler": job.scheduler,
            "output_path": str(job.output_path.resolve()),
            "upscaled_output_path": (
                str(job.upscaled_output_path.resolve())
                if job.upscaled_output_path is not None
                else None
            ),
            "upscale": job.upscale.to_dict(),
            "preview_enabled": self._preview_enabled,
            "workbench_version": _workbench_version(),
            # Krea2 编辑模式专用字段；其他任务一律 None。
            "input_image_path": (
                str(job.input_image_path.resolve())
                if job.input_image_path is not None
                else None
            ),
            "secondary_input_image_path": (
                str(job.secondary_input_image_path.resolve())
                if job.secondary_input_image_path is not None
                else None
            ),
            "grounding_px": (
                int(job.grounding_px) if job.grounding_px is not None else None
            ),
            "ref_boost": (
                float(job.ref_boost) if job.ref_boost is not None else None
            ),
            "edit_lora_path": edit_lora_path,
            # Krea2 参考图重排专用字段；其他任务一律空列表。
            "reference_image_paths": [
                str(path.resolve()) for path in job.reference_image_paths
            ],
            "reference_image_tokens": list(job.reference_image_tokens),
            "remote_encoder": {
                "enabled": self.config.remote_encoder.enabled,
                "host": self.config.remote_encoder.host,
                "port": self.config.remote_encoder.port,
                "connect_timeout_seconds": self.config.remote_encoder.connect_timeout_seconds,
                "request_timeout_seconds": self.config.remote_encoder.request_timeout_seconds,
                "fallback_to_local": self.config.remote_encoder.fallback_to_local,
            },
        }

    def _video_command(self, job: VideoJobRecord) -> dict:
        model_high, model_low = resolve_video_diffusion_pair(
            job.video_model, job.model_path
        )
        video_resources = self.config.video_resources[job.video_model]
        return {
            "type": "generate_video",
            "job_id": job.id,
            "batch_id": job.batch_id,
            "video_model": job.video_model.value,
            "model_path": str(job.model_path.resolve()),
            "model_high_path": str(model_high),
            "model_low_path": str(model_low) if model_low is not None else None,
            "vae_path": str(job.vae_path.resolve()),
            "text_encoder_path": str(job.text_encoder_path.resolve()),
            "audio_vae_path": (
                str(job.audio_vae_path.resolve()) if job.audio_vae_path else None
            ),
            # turbo 蒸馏 LoRA 为服务端固定配置，客户端不可指定；仅 minimax-h3-turbo 非空
            "turbo_lora_path": (
                str(video_resources.turbo_lora.resolve())
                if video_resources.turbo_lora
                else None
            ),
            "clip_type": video_resources.clip_type,
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt,
            "input_image_path": (
                str(job.input_image_path.resolve()) if job.input_image_path else None
            ),
            "last_frame_image_path": (
                str(job.last_frame_image_path.resolve())
                if job.last_frame_image_path
                else None
            ),
            "reference_image_paths": [
                str(path.resolve()) for path in job.reference_image_paths
            ],
            "reference_video_paths": [
                str(path.resolve()) for path in job.reference_video_paths
            ],
            "reference_audio_paths": [
                str(path.resolve()) for path in job.reference_audio_paths
            ],
            "width": job.width,
            "height": job.height,
            "duration_seconds": job.duration_seconds,
            "fps": job.fps,
            "length": job.length,
            "steps": job.steps,
            "seed": job.seed,
            "cfg": job.cfg,
            "sampler": job.sampler,
            "scheduler": job.scheduler,
            "denoise": job.denoise,
            "shift": job.shift,
            "latent_multiplier": job.latent_multiplier,
            "output_path": str(job.output_path.resolve()),
            "preview_enabled": False,
            "workbench_version": _workbench_version(),
        }

    def cancel(self) -> None:
        self._cancelled.set()
        self._logger.warning("worker cancellation requested")
        with self._process_lock:
            process = self._process
            job_id = self._active_job_id
            if process is None or process.poll() is not None or job_id is None:
                return
            self._write_command(process, {"type": "cancel", "job_id": job_id})

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = bool(enabled)

    def release_resources(self) -> None:
        with self._generation_lock:
            with self._process_lock:
                process = self._process
                if process is None or process.poll() is not None:
                    self._loaded_model = None
                    self._loaded_resources = {}
                    return
                if self._active_job_id is not None:
                    raise RuntimeError("任务运行期间不能释放 Worker 资源")
                self._write_command(process, {"type": "release"})
            deadline = time.monotonic() + min(self.config.worker_timeout_seconds, 60)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("释放 Worker 资源超时")
                try:
                    event = self._events.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if process.poll() is not None:
                        self._raise_worker_exit(process)
                    continue
                if event.get("type") == "released":
                    self._loaded_model = None
                    self._loaded_resources = {}
                    return
                if event.get("type") == "error" and event.get("job_id") is None:
                    raise RuntimeError(event.get("error", "释放 Worker 资源失败"))
                if event.get("type") == "process_eof":
                    self._raise_worker_exit(process)

    def close(self) -> None:
        with self._process_lock:
            if self._closed and self._process is None:
                return
            self._closed = True
            process = self._process
        if process is not None and process.poll() is None:
            try:
                self._logger.info("worker shutdown requested pid=%s", process.pid)
                self._write_command(process, {"type": "shutdown"})
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._logger.warning(
                    "worker graceful shutdown failed pid=%s; terminating",
                    process.pid,
                )
                self._terminate_process()
        if process is not None:
            self._finish_reader()
            self._close_pipes(process)
        with self._process_lock:
            self._process = None
            self._loaded_model = None
            self._loaded_resources = {}
            self._active_job_id = None

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
            "loaded_resources": dict(self._loaded_resources),
            "preview_enabled": self._preview_enabled,
            "gpu": self._gpu_value,
            "memory": dict(self._memory_status),
            "gpu_memory_used_gib": self._gpu_memory_used_gib,
            "gpu_memory_total_gib": self._gpu_memory_total_gib,
            "gpu_memory_percent": self._gpu_memory_percent,
            "gpu_utilization_percent": self._gpu_utilization_percent,
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
            probe = self._gpu_memory()
            if isinstance(probe, dict):
                value = str(probe["display"])
                memory = probe.get("memory")
                if isinstance(memory, dict):
                    self._memory_status = memory
                self._gpu_memory_used_gib = probe.get("gpu_memory_used_gib")
                self._gpu_memory_total_gib = probe.get("gpu_memory_total_gib")
                self._gpu_memory_percent = probe.get("gpu_memory_percent")
                self._gpu_utilization_percent = probe.get("gpu_utilization_percent")
            else:
                # 兼容旧测试或外部覆盖的字符串探测器。
                value = str(probe)
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
            self._logger.info(
                "worker starting python=%s script=%s comfy_root=%s",
                self.config.comfyui.python,
                self.worker_script,
                self.config.comfyui.root,
            )
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
            self._logger.info("worker process started pid=%s", process.pid)
            self._reader_thread = threading.Thread(
                target=self._read_output,
                args=(process, self._events),
                name=f"comfy-worker-{process.pid}",
                daemon=True,
            )
            self._reader_thread.start()

        startup_deadline = time.monotonic() + min(
            self.config.worker_timeout_seconds, 60
        )
        while True:
            remaining = startup_deadline - time.monotonic()
            if remaining <= 0:
                self._logger.error("worker startup timed out pid=%s", process.pid)
                self._terminate_process()
                raise TimeoutError("GPU Worker 启动超时")
            try:
                event = self._events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if process.poll() is not None:
                    self._raise_worker_exit(process)
                continue
            if event.get("type") == "ready":
                self._logger.info("worker ready pid=%s", process.pid)
                return process
            if event.get("type") == "startup_error":
                self._logger.error(
                    "worker startup failed pid=%s error=%s",
                    process.pid,
                    event.get("error", "GPU Worker 初始化失败"),
                )
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
                    self._logger.warning(
                        "worker pid=%s emitted invalid event: %s", process.pid, line
                    )
                    continue
                if isinstance(event, dict):
                    issue = _worker_event_issue(event)
                    if issue is not None:
                        self._logger.warning(
                            "worker pid=%s emitted invalid event issue=%s keys=%s",
                            process.pid,
                            issue,
                            sorted(event),
                        )
                    events.put(event)
                else:
                    self._logs.append(line)
                    self._logger.warning(
                        "worker pid=%s emitted non-object event: %s", process.pid, line
                    )
            elif line:
                self._logs.append(line)
                self._logger.log(
                    worker_output_level(line), "worker pid=%s %s", process.pid, line
                )
        events.put({"type": "process_eof", "return_code": process.poll()})

    def _finish_reader(self) -> None:
        thread = self._reader_thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join()
        if self._reader_thread is thread:
            self._reader_thread = None

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
            self._logger.warning("worker terminating pid=%s", process.pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._logger.warning("worker kill required pid=%s", process.pid)
                process.kill()
                process.wait(timeout=10)
        self._finish_reader()
        self._close_pipes(process)
        with self._process_lock:
            if self._process is process:
                self._process = None
                self._loaded_model = None
                self._loaded_resources = {}
                self._active_job_id = None

    def _raise_worker_exit(self, process: subprocess.Popen) -> None:
        self._finish_reader()
        self._close_pipes(process)
        with self._process_lock:
            if self._process is process:
                self._process = None
                self._loaded_model = None
                self._loaded_resources = {}
                self._active_job_id = None
        if self._cancelled.is_set():
            self._logger.warning(
                "worker exited after cancellation pid=%s code=%s",
                process.pid,
                process.poll(),
            )
            raise GenerationCancelled("任务已取消，GPU Worker 已退出")
        log_tail = "\n".join(list(self._logs)[-20:])
        self._logger.error(
            "worker exited unexpectedly pid=%s code=%s",
            process.pid,
            process.poll(),
        )
        raise RuntimeError(
            f"GPU Worker 异常退出，code={process.poll()}"
            + (f"\n{log_tail}" if log_tail else "")
        )

    @staticmethod
    def _gpu_memory() -> dict | str:
        memory = psutil.virtual_memory()
        memory_status = {
            "used_gib": round(memory.used / 1024**3, 2),
            "total_gib": round(memory.total / 1024**3, 2),
            "percent": round(float(memory.percent), 1),
        }
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if result.returncode == 0:
                used, total, utilization = result.stdout.strip().split(",", 2)
                used_gib = round(int(used.strip()) / 1024, 2)
                total_gib = round(int(total.strip()) / 1024, 2)
                memory_percent = round(used_gib / total_gib * 100, 1) if total_gib else None
                utilization_percent = float(utilization.strip())
                return {
                    "display": f"{used_gib:.2f}/{total_gib:.2f} GiB",
                    "memory": memory_status,
                    "gpu_memory_used_gib": used_gib,
                    "gpu_memory_total_gib": total_gib,
                    "gpu_memory_percent": memory_percent,
                    "gpu_utilization_percent": utilization_percent,
                }
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        return {
            "display": "不可用",
            "memory": memory_status,
            "gpu_memory_used_gib": None,
            "gpu_memory_total_gib": None,
            "gpu_memory_percent": None,
            "gpu_utilization_percent": None,
        }

    @staticmethod
    def _close_pipes(process: subprocess.Popen) -> None:
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass


def _worker_event_issue(event: dict) -> str | None:
    event_type = event.get("type")
    if not isinstance(event_type, str):
        return "missing string type"
    required = WORKER_EVENT_REQUIRED_FIELDS.get(event_type)
    if required is None:
        return f"unknown type {event_type!r}"
    missing = [field for field in required if field not in event]
    if missing:
        return f"missing fields {missing}"
    return None
