from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Static

from .commands import CommandSession


class WorkbenchApp(App):
    TITLE = "diffusion-workbench"
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
        height: 2;
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
        self.running_job = None
        self.step = 0
        self.total_steps = 0
        self.stage = "idle"
        self.last_error = ""

    def compose(self) -> ComposeResult:
        yield Static("diffusion-workbench", id="title")
        yield RichLog(id="log", wrap=True, markup=False)
        yield Static("队列空闲 | Worker stopped", id="progress")
        yield Input(placeholder="/status", id="command")

    def on_mount(self) -> None:
        self.core.set_event_sink(self._receive_core_event)
        self.query_one("#command", Input).focus()
        self.query_one("#log", RichLog).write("ready: mode=zit, sampler=euler, scheduler=simple, cfg=1")

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

    def _receive_core_event(self, event: dict) -> None:
        if not self.is_running:
            return
        try:
            self.call_from_thread(self._apply_core_event, event)
        except RuntimeError:
            self._apply_core_event(event)

    def _apply_core_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "queue_progress":
            self.queue_waiting = int(event.get("queued", 0))
            self.running_job = event.get("running")
        elif event_type == "job_started":
            self.running_job = event.get("job_id")
            self.queue_waiting = max(0, self.queue_waiting - 1)
            self.step = 0
            self.total_steps = 0
            self.stage = "loading_model"
            self.last_error = ""
        elif event_type == "stage_progress":
            self.stage = str(event.get("stage", "idle"))
            if event.get("total") is not None:
                self.total_steps = int(event["total"])
        elif event_type == "step_progress":
            self.stage = "sampling"
            self.step = int(event.get("step", 0))
            self.total_steps = int(event.get("total", 0))
        elif event_type == "job_finished":
            status = event.get("status")
            output = f": {event.get('output_path')}" if status == "completed" else ""
            self.query_one("#log", RichLog).write(
                f"job {event.get('job_id')}: {status}{output}"
            )
            self.step = self.total_steps = 0
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
        if self.running_job:
            if self.stage == "prompt":
                detail = "处理 prompt"
            elif self.stage == "sampling":
                detail = f"采样步数 {self.step}/{self.total_steps}"
            elif self.stage == "vae":
                detail = "VAE 处理"
            elif self.stage == "saved":
                detail = "图片已保存"
            else:
                detail = "加载模型"
            text = f"{detail} | 等待 {self.queue_waiting}"
        elif self.queue_waiting:
            if self.stage == "saved":
                detail = "图片已保存"
            elif self.stage == "failed":
                detail = f"生成失败: {self.last_error}" if self.last_error else "生成失败"
            elif self.stage == "cancelled":
                detail = "任务已停止"
            else:
                detail = "等待启动"
            text = f"{detail} | 队列 {self.queue_waiting}"
        else:
            status = self.core.runtime_status()
            if self.stage == "saved":
                detail = "图片已保存"
            elif self.stage == "failed":
                detail = f"生成失败: {self.last_error}" if self.last_error else "生成失败"
            elif self.stage == "cancelled":
                detail = "任务已停止"
            else:
                detail = "队列空闲"
            text = f"{detail} | Worker {status.get('worker', 'stopped')} | GPU {status.get('gpu', '不可用')}"
        self.query_one("#progress", Static).update(text)

    def on_unmount(self) -> None:
        self.core.shutdown()
