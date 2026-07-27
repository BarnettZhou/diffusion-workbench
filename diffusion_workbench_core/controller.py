import queue
import secrets
import threading
import time
from collections.abc import Callable
from datetime import datetime

from .domain import GenerationSettings, JobRecord
from .logging_config import silent_logger
from .runtime import GenerationCancelled, GenerationRuntime
from .storage import JobStore


EventSink = Callable[[dict], None]
RANDOM_SEED_UPPER_BOUND = 2**53


class GenerationController:
    def __init__(
        self,
        runtime: GenerationRuntime,
        store: JobStore,
        event_sink: EventSink | None = None,
        seed_source: Callable[[], int] | None = None,
        logger=None,
    ):
        self.runtime = runtime
        self.store = store
        self._event_sink = event_sink or (lambda _event: None)
        self._seed_source = seed_source or (
            lambda: secrets.randbelow(RANDOM_SEED_UPPER_BOUND)
        )
        self._logger = logger or silent_logger()
        self._queue: queue.Queue[tuple[int, JobRecord] | None] = queue.Queue()
        self._lock = threading.RLock()
        self._current: JobRecord | None = None
        self._current_cancellable = False
        self._cancel_requested: set[str] = set()
        self._generation = 0
        self._queue_event_sequence = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="generation-controller", daemon=True)
        self._thread.start()

    def set_event_sink(self, event_sink: EventSink) -> None:
        self._event_sink = event_sink

    def submit(self, settings: GenerationSettings, count: int) -> list[JobRecord]:
        with self._lock:
            if self._closed:
                raise RuntimeError("core 已关闭")
            jobs = self.store.create_jobs(settings, count)
            for job in jobs:
                self._queue.put((self._generation, job))
        self._emit_queue()
        first = jobs[0]
        self._logger.info(
            "jobs queued count=%d batch_id=%s mode=%s size=%dx%d steps=%d model=%s",
            len(jobs),
            first.batch_id,
            first.mode.value,
            first.width,
            first.height,
            first.steps,
            first.model_path.name,
        )
        return jobs

    def stop(self) -> None:
        queued = []
        with self._lock:
            self._generation += 1
            current = self._current
            if current is not None:
                self._cancel_requested.add(current.id)
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._queue.task_done()
                    continue
                _, job = item
                queued.append(job)
                self._queue.task_done()
        for job in queued:
            try:
                self.store.mark_cancelled(job.id)
            except Exception as exc:
                self._logger.exception(
                    "queued job cancellation persistence failed job_id=%s", job.id
                )
                self._emit(
                    {
                        "type": "job_error",
                        "job_id": job.id,
                        "error": f"取消状态写入失败: {exc}",
                    }
                )
        if current is not None:
            try:
                self.runtime.cancel()
            except Exception as exc:
                self._logger.exception(
                    "worker cancellation failed job_id=%s", current.id
                )
                self._emit(
                    {
                        "type": "job_error",
                        "job_id": current.id,
                        "error": f"GPU Worker 取消失败: {exc}",
                    }
                )
        if current is not None or queued:
            self._logger.warning(
                "stop requested running_job=%s cancelled_queued=%d",
                current.id if current else None,
                len(queued),
            )
        else:
            self._logger.info("stop requested while idle")
        self._emit_queue()

    def skip_current(self) -> str | None:
        cancel_error = None
        with self._lock:
            current = self._current
            if current is None or not self._current_cancellable:
                self._logger.info("skip requested with no cancellable job")
                return None
            cancellation_already_requested = current.id in self._cancel_requested
            self._cancel_requested.add(current.id)
            queued_remaining = self._queue.qsize()
            try:
                self.runtime.cancel()
            except Exception as exc:
                cancel_error = exc
                if not cancellation_already_requested:
                    self._cancel_requested.discard(current.id)
                self._logger.exception(
                    "worker cancellation failed while skipping job_id=%s", current.id
                )
            else:
                self._logger.warning(
                    "skip requested job_id=%s queued_remaining=%d",
                    current.id,
                    queued_remaining,
                )
        if cancel_error is not None:
            self._emit(
                {
                    "type": "job_error",
                    "job_id": current.id,
                    "error": f"GPU Worker 跳过失败: {cancel_error}",
                }
            )
            raise RuntimeError(f"跳过任务失败: {cancel_error}") from cancel_error
        self._emit_queue()
        return current.id

    def wait_idle(self) -> None:
        self._queue.join()

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._logger.info("controller shutting down")
        self.stop()
        try:
            self.runtime.close()
        except Exception as exc:
            self._logger.exception("worker close failed")
            self._emit(
                {
                    "type": "job_error",
                    "job_id": self._current.id if self._current else None,
                    "error": f"GPU Worker 关闭失败: {exc}",
                }
            )
        self.wait_idle()
        self._queue.put(None)
        self._thread.join(timeout=10)
        self._logger.info("controller stopped")

    def status(self) -> dict:
        with self._lock:
            current = self._current.id if self._current else None
            queued = self._queue.qsize()
        return {"queue": queued, "running": current, **self.runtime.status()}

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            generation, job = item
            with self._lock:
                cancelled_before_start = generation != self._generation
                if not cancelled_before_start:
                    self._current = job
                    self._current_cancellable = False
            if cancelled_before_start:
                try:
                    self.store.mark_cancelled(job.id)
                except Exception as exc:
                    self._logger.exception(
                        "stale job cancellation persistence failed job_id=%s", job.id
                    )
                    self._emit(
                        {
                            "type": "job_error",
                            "job_id": job.id,
                            "error": f"取消状态写入失败: {exc}",
                        }
                    )
                finally:
                    self._queue.task_done()
                    self._emit(
                        {
                            "type": "job_finished",
                            "job_id": job.id,
                            "status": "cancelled",
                            "steps": job.steps,
                        }
                    )
                    self._emit_queue()
                continue
            self._emit_queue()
            started_clock = time.perf_counter()
            duration = None
            status = "failed"
            try:
                started_at = datetime.now().astimezone()
                seed = job.seed if job.seed >= 0 else self._seed_source()
                running_job = self.store.mark_running(job.id, seed, started_at)
                self._logger.info(
                    "job started job_id=%s mode=%s seed=%d size=%dx%d steps=%d model=%s",
                    job.id,
                    running_job.mode.value,
                    seed,
                    running_job.width,
                    running_job.height,
                    running_job.steps,
                    running_job.model_path.name,
                )
                self._emit(
                    {
                        "type": "job_started",
                        "job_id": job.id,
                        "seed": seed,
                        "steps": running_job.steps,
                    }
                )

                def progress(step, total, metrics=None):
                    metrics = metrics or {}
                    self._logger.info(
                        "job sampling job_id=%s step=%d total=%d seconds_per_step=%s "
                        "steps_per_second=%s eta_seconds=%s",
                        job.id,
                        step,
                        total,
                        metrics.get("seconds_per_step"),
                        metrics.get("steps_per_second"),
                        metrics.get("eta_seconds"),
                    )
                    self._emit(
                        {
                            "type": "step_progress",
                            "job_id": job.id,
                            "step": step,
                            "total": total,
                            **metrics,
                        }
                    )

                def stage_progress(stage, total):
                    self._logger.info(
                        "job stage job_id=%s stage=%s total=%s", job.id, stage, total
                    )
                    self._emit(
                        {
                            "type": "stage_progress",
                            "job_id": job.id,
                            "stage": stage,
                            "total": total,
                        }
                    )

                with self._lock:
                    self._current_cancellable = True
                result = self.runtime.generate(
                    running_job,
                    progress,
                    stage_progress,
                    lambda event: self._emit(event),
                ) or {}
                self._logger.info(
                    "job worker result job_id=%s load_seconds=%s sampling_seconds=%s "
                    "vae_seconds=%s generation_seconds=%s",
                    job.id,
                    result.get("load_seconds"),
                    result.get("sampling_seconds"),
                    result.get("vae_seconds"),
                    result.get("generation_seconds"),
                )
                duration = time.perf_counter() - started_clock
                with self._lock:
                    self._current_cancellable = False
                    cancellation_requested = job.id in self._cancel_requested
                if cancellation_requested:
                    self.store.mark_cancelled(job.id, duration_seconds=duration)
                    status = "cancelled"
                else:
                    self.store.mark_completed(job.id, datetime.now().astimezone(), duration)
                    status = "completed"
            except GenerationCancelled as exc:
                duration = time.perf_counter() - started_clock
                status = "cancelled"
                with self._lock:
                    self._current_cancellable = False
                self._logger.warning("job cancelled job_id=%s reason=%s", job.id, exc)
                try:
                    self.store.mark_cancelled(job.id, duration_seconds=duration)
                except Exception as exc:
                    self._logger.exception(
                        "job cancellation persistence failed job_id=%s", job.id
                    )
                    self._emit(
                        {
                            "type": "job_error",
                            "job_id": job.id,
                            "error": f"取消状态写入失败: {exc}",
                        }
                    )
            except Exception as exc:
                duration = time.perf_counter() - started_clock
                with self._lock:
                    self._current_cancellable = False
                    cancellation_requested = job.id in self._cancel_requested
                if cancellation_requested:
                    status = "cancelled"
                    self._logger.warning(
                        "job cancelled during runtime failure job_id=%s reason=%s",
                        job.id,
                        exc,
                    )
                    try:
                        self.store.mark_cancelled(job.id, duration_seconds=duration)
                    except Exception as persistence_error:
                        self._logger.exception(
                            "job cancellation persistence failed job_id=%s", job.id
                        )
                        self._emit(
                            {
                                "type": "job_error",
                                "job_id": job.id,
                                "error": f"取消状态写入失败: {persistence_error}",
                            }
                        )
                else:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    self._logger.exception("job failed job_id=%s", job.id)
                    try:
                        self.store.mark_failed(
                            job.id,
                            datetime.now().astimezone(),
                            duration,
                            error,
                        )
                    except Exception as persistence_error:
                        self._logger.exception(
                            "job failure persistence failed job_id=%s", job.id
                        )
                        error += f"; 失败状态写入失败: {persistence_error}"
                    self._emit({"type": "job_error", "job_id": job.id, "error": error})
            finally:
                with self._lock:
                    self._cancel_requested.discard(job.id)
                    self._current_cancellable = False
                    self._current = None
                self._queue.task_done()
                self._emit(
                    {
                        "type": "job_finished",
                        "job_id": job.id,
                        "status": status,
                        "output_path": str(job.output_path),
                        "steps": job.steps,
                    }
                )
                log = {
                    "completed": self._logger.info,
                    "cancelled": self._logger.warning,
                    "failed": self._logger.error,
                }[status]
                log(
                    "job finished job_id=%s status=%s duration_seconds=%s output_path=%s",
                    job.id,
                    status,
                    round(duration, 3) if duration is not None else None,
                    job.output_path,
                )
                self._emit_queue()

    def _emit_queue(self) -> None:
        with self._lock:
            self._queue_event_sequence += 1
            event = {
                "type": "queue_progress",
                "queued": self._queue.qsize(),
                "running": self._current.id if self._current else None,
                "steps": self._current.steps if self._current else None,
                "sequence": self._queue_event_sequence,
            }
        self._emit(event)

    def _emit(self, event: dict) -> None:
        try:
            self._event_sink(event)
        except Exception:
            self._logger.exception(
                "event sink failed type=%s job_id=%s",
                event.get("type"),
                event.get("job_id"),
            )
