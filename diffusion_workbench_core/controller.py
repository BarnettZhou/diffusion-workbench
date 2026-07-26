import queue
import secrets
import threading
import time
from collections.abc import Callable
from datetime import datetime

from .domain import GenerationSettings, JobRecord
from .runtime import GenerationCancelled, GenerationRuntime
from .storage import JobStore


EventSink = Callable[[dict], None]


class GenerationController:
    def __init__(
        self,
        runtime: GenerationRuntime,
        store: JobStore,
        event_sink: EventSink | None = None,
        seed_source: Callable[[], int] | None = None,
    ):
        self.runtime = runtime
        self.store = store
        self._event_sink = event_sink or (lambda _event: None)
        self._seed_source = seed_source or (lambda: secrets.randbelow(2**63))
        self._queue: queue.Queue[tuple[int, JobRecord] | None] = queue.Queue()
        self._lock = threading.RLock()
        self._current: JobRecord | None = None
        self._cancel_requested: set[str] = set()
        self._generation = 0
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
                self._emit(
                    {
                        "type": "job_error",
                        "job_id": current.id,
                        "error": f"GPU Worker 取消失败: {exc}",
                    }
                )
        self._emit_queue()

    def wait_idle(self) -> None:
        self._queue.join()

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.stop()
        try:
            self.runtime.close()
        except Exception as exc:
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
            if cancelled_before_start:
                try:
                    self.store.mark_cancelled(job.id)
                except Exception as exc:
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
                        }
                    )
                    self._emit_queue()
                continue
            started_clock = time.perf_counter()
            status = "failed"
            try:
                started_at = datetime.now().astimezone()
                seed = job.seed if job.seed >= 0 else self._seed_source()
                running_job = self.store.mark_running(job.id, seed, started_at)
                self._emit({"type": "job_started", "job_id": job.id, "seed": seed})
                self.runtime.generate(
                    running_job,
                    lambda step, total: self._emit(
                        {
                            "type": "step_progress",
                            "job_id": job.id,
                            "step": step,
                            "total": total,
                        }
                    ),
                    lambda stage, total: self._emit(
                        {
                            "type": "stage_progress",
                            "job_id": job.id,
                            "stage": stage,
                            "total": total,
                        }
                    ),
                )
                duration = time.perf_counter() - started_clock
                if job.id in self._cancel_requested:
                    self.store.mark_cancelled(job.id, duration_seconds=duration)
                    status = "cancelled"
                else:
                    self.store.mark_completed(job.id, datetime.now().astimezone(), duration)
                    status = "completed"
            except GenerationCancelled:
                duration = time.perf_counter() - started_clock
                status = "cancelled"
                try:
                    self.store.mark_cancelled(job.id, duration_seconds=duration)
                except Exception as exc:
                    self._emit(
                        {
                            "type": "job_error",
                            "job_id": job.id,
                            "error": f"取消状态写入失败: {exc}",
                        }
                    )
            except Exception as exc:
                duration = time.perf_counter() - started_clock
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                try:
                    self.store.mark_failed(
                        job.id,
                        datetime.now().astimezone(),
                        duration,
                        error,
                    )
                except Exception as persistence_error:
                    error += f"; 失败状态写入失败: {persistence_error}"
                self._emit({"type": "job_error", "job_id": job.id, "error": error})
            finally:
                with self._lock:
                    self._cancel_requested.discard(job.id)
                    self._current = None
                self._queue.task_done()
                self._emit(
                    {
                        "type": "job_finished",
                        "job_id": job.id,
                        "status": status,
                        "output_path": str(job.output_path),
                    }
                )
                self._emit_queue()

    def _emit_queue(self) -> None:
        with self._lock:
            self._emit(
                {
                    "type": "queue_progress",
                    "queued": self._queue.qsize(),
                    "running": self._current.id if self._current else None,
                }
            )

    def _emit(self, event: dict) -> None:
        try:
            self._event_sink(event)
        except Exception:
            pass
