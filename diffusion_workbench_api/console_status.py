"""API 进程终端底部常驻任务进度状态栏。

仅在 TTY 终端启用（非 TTY 时整体 no-op，避免 ANSI 转义污染重定向的日志）。
可用环境变量 ``DWB_CONSOLE_STATUS=0`` 显式关闭。

状态更新与终端重绘解耦：``handle_event`` 由 Core 后台线程调用，只在锁内
更新共享状态；渲染由 rich.live.Live 的定时刷新完成，天然线程安全，也避免
高频 step_progress 事件直接刷屏。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass, field

ENV_FLAG = "DWB_CONSOLE_STATUS"

# 与 frontend/src/components/ProgressPanel.jsx 的阶段标签保持一致
STAGE_LABELS = {
    "starting_worker": "启动推理 Worker",
    "loading_model": "加载模型资源",
    "prompt": "编码提示词",
    "latent": "准备 latent",
    "sampling": "采样中",
    "vae": "VAE 解码",
    "saving": "保存图片",
    "saved": "图片已保存",
    "upscale_preparing": "放大准备",
    "upscale_sampling": "放大采样中",
    "upscale_decoding": "放大解码",
    "upscale_saving": "保存放大图",
    "upscale_saved": "放大图已保存",
}

_FINISHED_LABELS = {
    "completed": "已完成",
    "cancelled": "已取消",
    "failed": "失败",
}

_BAR_WIDTH = 14


@dataclass
class _JobProgress:
    job_id: str
    stage: str = "starting_worker"
    step: int = 0
    total: int | None = None
    steps_per_second: float | None = None
    seconds_per_step: float | None = None
    eta_seconds: float | None = None


@dataclass
class _StatusState:
    queued: int = 0
    current: _JobProgress | None = None
    last_finished: str | None = None


def _format_eta(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _format_speed(job: _JobProgress) -> str | None:
    if job.steps_per_second:
        return f"{job.steps_per_second:.2f} it/s"
    if job.seconds_per_step:
        return f"{job.seconds_per_step:.2f} s/it"
    return None


class StatusTracker:
    """Core 事件到状态栏快照的纯逻辑转换，不触碰终端，可独立测试。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = _StatusState()

    def handle_event(self, event: dict) -> None:
        event_type = event.get("type")
        with self._lock:
            state = self._state
            if event_type == "queue_progress":
                state.queued = int(event.get("queued") or 0)
            elif event_type == "job_started":
                state.current = _JobProgress(
                    job_id=str(event.get("job_id") or "?"),
                    total=event.get("steps"),
                )
                state.last_finished = None
            elif event_type == "stage_progress":
                if state.current is not None and event.get("job_id") == state.current.job_id:
                    state.current.stage = str(event.get("stage") or "")
                    total = event.get("total")
                    if total is not None:
                        state.current.total = int(total)
            elif event_type == "step_progress":
                if state.current is not None and event.get("job_id") == state.current.job_id:
                    current = state.current
                    current.step = int(event.get("step") or 0)
                    total = event.get("total")
                    if total is not None:
                        current.total = int(total)
                    current.stage = str(event.get("stage") or current.stage)
                    current.steps_per_second = event.get("steps_per_second")
                    current.seconds_per_step = event.get("seconds_per_step")
                    current.eta_seconds = event.get("eta_seconds")
            elif event_type == "job_finished":
                job_id = str(event.get("job_id") or "?")
                if state.current is None or state.current.job_id == job_id:
                    state.current = None
                label = _FINISHED_LABELS.get(event.get("status"), "已结束")
                state.last_finished = f"任务 {job_id[:8]} {label}"

    def render(self):
        """构建当前状态的单行渲染对象（rich.text.Text）。"""
        from rich.text import Text

        with self._lock:
            state = self._state
            line = Text("DWB ", style="bold cyan")
            current = state.current
            if current is None:
                line.append("空闲", style="dim")
                if state.queued:
                    line.append(f" │ 队列 {state.queued}", style="yellow")
                if state.last_finished:
                    line.append(f" │ {state.last_finished}", style="dim")
                return line

            label = STAGE_LABELS.get(current.stage, current.stage or "运行中")
            line.append(f"{current.job_id[:8]} {label}", style="bold")
            if current.total:
                percent = min(100, round(current.step / current.total * 100))
                filled = min(_BAR_WIDTH, round(current.step / current.total * _BAR_WIDTH))
                bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
                line.append(f" {current.step}/{current.total} {bar} {percent}%")
            speed = _format_speed(current)
            if speed:
                line.append(f" │ {speed}", style="green")
            eta = _format_eta(current.eta_seconds)
            if eta:
                line.append(f" │ ETA {eta}", style="yellow")
            if state.queued:
                line.append(f" │ 队列 {state.queued}", style="yellow")
            return line


def _enabled() -> bool:
    flag = os.environ.get(ENV_FLAG, "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    # dumb terminal 下 rich Live 不会绘制，直接视为不可用
    if os.environ.get("TERM", "").lower() in ("dumb", "unknown"):
        return False
    # 即使显式开启也要求 TTY：非 TTY 输出 ANSI 只会污染日志文件
    return sys.stderr.isatty()


class ConsoleStatusBar:
    """API 进程终端底部状态栏。非 TTY 或显式关闭时全部方法为 no-op。"""

    def __init__(self):
        self.tracker = StatusTracker()
        self._console = None
        self._live = None
        self._original_root_handlers: list | None = None
        self._original_uvicorn: dict[str, tuple[list, bool]] = {}

    @property
    def active(self) -> bool:
        return self._live is not None

    def handle_event(self, event: dict) -> None:
        self.tracker.handle_event(event)

    def start(self) -> bool:
        """启动状态栏；返回是否真正启用（仅 TTY 终端启用）。"""
        if not _enabled():
            return False
        from rich.console import Console
        from rich.live import Live

        # uvicorn 日志默认走 stderr，状态栏与重定向后的日志共用同一 console
        self._console = Console(stderr=True)
        # get_renderable 回调让自动刷新每次都拉取最新状态；
        # 直接传 tracker.render() 的结果会定格在启动帧
        self._live = Live(
            console=self._console,
            refresh_per_second=4,
            transient=True,
            get_renderable=self.tracker.render,
        )
        self._live.start()
        self._redirect_logging()
        return True

    def stop(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None
        self._restore_logging()

    def _redirect_logging(self) -> None:
        """把 root 与 uvicorn 日志切到 RichHandler，使日志从状态栏上方滚出。"""
        from rich.logging import RichHandler

        handler = RichHandler(console=self._console, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        self._original_root_handlers = root.handlers[:]
        root.handlers = [handler]
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            target = logging.getLogger(name)
            self._original_uvicorn[name] = (target.handlers[:], target.propagate)
            target.handlers = []
            target.propagate = True

    def _restore_logging(self) -> None:
        if self._original_root_handlers is not None:
            logging.getLogger().handlers = self._original_root_handlers
            self._original_root_handlers = None
        for name, (handlers, propagate) in self._original_uvicorn.items():
            target = logging.getLogger(name)
            target.handlers = handlers
            target.propagate = propagate
        self._original_uvicorn = {}
