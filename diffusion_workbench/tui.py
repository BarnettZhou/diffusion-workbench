from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Input, RichLog, Static

from .commands import CommandSession


class CoreEvent(Message):
    def __init__(self, event: dict):
        super().__init__()
        self.event = event


class WorkbenchApp(App):
    TITLE = "diffusion-workbench"
    STAGE_LABELS = {
        "preparing_job": "准备任务",
        "starting_worker": "启动推理 Worker",
        "loading_model": "加载模型资源",
        "prompt": "编码提示词",
        "latent": "准备 latent",
        "sampling": "采样中",
        "vae": "VAE 解码",
        "saving": "保存图片",
        "saved": "图片已保存",
        "failed": "生成失败",
        "cancelled": "任务已取消",
    }
    CSS = """
    Screen {
        layout: vertical;
    }
    #title {
        height: 1;
        padding: 0 1;
        background: $primary;
        color: $text;
        text-style: bold;
    }
    #log {
        height: 1fr;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    #progress {
        height: 4;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
        content-align: left middle;
    }
    #command {
        height: 3;
        border: tall $primary;
    }
    """

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.session = CommandSession(core)
        self.queue_waiting = 0
        self.queue_sequence = 0
        self.running_job = None
        self.step = 0
        self.total_steps = 0
        self.seconds_per_step = None
        self.steps_per_second = None
        self.eta_seconds = None
        self.stage = "idle"
        self.last_error = ""

    def compose(self) -> ComposeResult:
        yield Static("diffusion-workbench", id="title")
        yield RichLog(id="log", wrap=True, markup=False)
        yield Static("当前阶段: 等待任务 | 采样步数: 未开始/8", id="progress")
        yield Input(placeholder="/status", id="command")

    def on_mount(self) -> None:
        self.core.set_event_sink(self._receive_core_event)
        self.query_one("#command", Input).focus()
        self.query_one("#log", RichLog).write("ready: mode=zit, sampler=euler, scheduler=simple, cfg=1")
        self._render_progress()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        event.input.clear()
        if not command:
            return
        log = self.query_one("#log", RichLog)
        log.write(f"> {command}")
        response = self.session.handle(command)
        for line in response.lines:
            log.write(line)
        if response.exit_requested:
            self.exit()
        else:
            self._render_progress()

    def _receive_core_event(self, event: dict) -> None:
        if not self.is_running:
            return
        self.post_message(CoreEvent(event))

    def on_core_event(self, message: CoreEvent) -> None:
        self._apply_core_event(message.event)

    def _apply_core_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "preview_image":
            return
        if event_type == "queue_progress":
            sequence = int(event.get("sequence", self.queue_sequence + 1))
            if sequence <= self.queue_sequence:
                return
            self.queue_sequence = sequence
            self.queue_waiting = int(event.get("queued", 0))
            next_running = event.get("running")
            if next_running and next_running != self.running_job:
                self.step = 0
                self.total_steps = int(event.get("steps") or self.session.steps)
                self._reset_sampling_metrics()
                self.stage = "preparing_job"
                self.last_error = ""
            self.running_job = next_running
        elif event_type == "job_started":
            self.running_job = event.get("job_id")
            self.step = 0
            self.total_steps = int(event.get("steps", self.session.steps))
            self._reset_sampling_metrics()
            self.stage = "starting_worker"
            self.last_error = ""
        elif event_type == "stage_progress":
            self.stage = str(event.get("stage", "idle"))
            if event.get("total") is not None:
                self.total_steps = int(event["total"])
            if self.stage in {"vae", "saving", "saved"}:
                self.step = self.total_steps
        elif event_type == "step_progress":
            self.stage = "sampling"
            self.step = int(event.get("step", 0))
            self.total_steps = int(event.get("total", 0))
            self.seconds_per_step = self._optional_float(
                event.get("seconds_per_step")
            )
            self.steps_per_second = self._optional_float(
                event.get("steps_per_second")
            )
            self.eta_seconds = self._optional_float(event.get("eta_seconds"))
        elif event_type == "job_finished":
            status = event.get("status")
            if event.get("job_id") != self.running_job:
                self.step = 0
            output = f": {event.get('output_path')}" if status == "completed" else ""
            self.query_one("#log", RichLog).write(
                f"job {event.get('job_id')}: {status}{output}"
            )
            if event.get("steps") is not None:
                self.total_steps = int(event["steps"])
            self.running_job = None
            if status == "completed":
                self.stage = "saved"
            elif status == "cancelled":
                self.stage = "cancelled"
            else:
                self.stage = "failed"
        elif event_type == "job_error":
            self.stage = "failed"
            self.last_error = str(event.get("error", "生成失败")).splitlines()[0]
            self.query_one("#log", RichLog).write(f"生成错误: {event.get('error')}")
        self._render_progress()

    def _render_progress(self) -> None:
        total = self.total_steps or self.session.steps
        if self.stage == "sampling":
            sample = f"{self.step}/{total}"
        elif self.stage in {"vae", "saving", "saved"}:
            sample = f"{total}/{total}"
        elif self.stage in {"failed", "cancelled"} and self.step:
            sample = f"{self.step}/{total}"
        else:
            sample = f"未开始/{total}"

        if self.stage == "idle":
            phase = "准备任务" if self.running_job else "等待任务"
        else:
            phase = self.STAGE_LABELS.get(self.stage, f"未知阶段 ({self.stage})")
        if self.stage == "failed" and self.last_error:
            phase += f": {self.last_error[:80]}"

        status = self.core.runtime_status()
        queue_state = "运行中" if self.running_job else "空闲"
        text = (
            f"当前阶段: {phase}\n"
            f"采样步数: {sample}{self._sampling_metrics_text()} | "
            f"队列: {queue_state} | 等待: {self.queue_waiting}\n"
            f"Worker: {status.get('worker', 'stopped')} | "
            f"GPU: {status.get('gpu', '不可用')}"
        )
        self.query_one("#progress", Static).update(text)

    def _sampling_metrics_text(self) -> str:
        if self.stage != "sampling" or self.seconds_per_step is None:
            return ""
        if self.steps_per_second is not None and self.steps_per_second >= 1:
            speed = f"{self.steps_per_second:.2f} 步/秒"
        else:
            speed = f"{self.seconds_per_step:.2f} 秒/步"
        eta = max(self.eta_seconds or 0.0, 0.0)
        if eta >= 60:
            eta_text = f"{int(eta) // 60}:{int(eta) % 60:02d}"
        else:
            eta_text = f"{eta:.1f} 秒"
        return f" | 速度: {speed} | 剩余: {eta_text}"

    def _reset_sampling_metrics(self) -> None:
        self.seconds_per_step = None
        self.steps_per_second = None
        self.eta_seconds = None

    @staticmethod
    def _optional_float(value):
        return float(value) if value is not None else None

    def on_unmount(self) -> None:
        self.core.shutdown()
