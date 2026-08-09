import os
import sys
import unittest
from unittest import mock

from diffusion_workbench_api import console_status
from diffusion_workbench_api.console_status import ConsoleStatusBar, StatusTracker


class StatusTrackerTests(unittest.TestCase):
    def test_idle_render(self):
        tracker = StatusTracker()
        text = tracker.render().plain
        self.assertIn("DWB", text)
        self.assertIn("空闲", text)

    def test_sampling_progress_render(self):
        tracker = StatusTracker()
        tracker.handle_event(
            {"type": "queue_progress", "queued": 2, "running": None, "sequence": 1}
        )
        tracker.handle_event({"type": "job_started", "job_id": "abcdef123456", "steps": 8})
        tracker.handle_event(
            {"type": "stage_progress", "job_id": "abcdef123456", "stage": "sampling", "total": 8}
        )
        tracker.handle_event(
            {
                "type": "step_progress",
                "job_id": "abcdef123456",
                "step": 4,
                "total": 8,
                "stage": "sampling",
                "steps_per_second": 1.5,
                "eta_seconds": 2.7,
            }
        )
        text = tracker.render().plain
        self.assertIn("采样中", text)
        self.assertIn("4/8", text)
        self.assertIn("50%", text)
        self.assertIn("1.50 it/s", text)
        self.assertIn("ETA 3s", text)
        self.assertIn("队列 2", text)

    def test_job_finished_returns_to_idle_with_summary(self):
        tracker = StatusTracker()
        tracker.handle_event({"type": "job_started", "job_id": "abcdef123456", "steps": 8})
        tracker.handle_event(
            {"type": "job_finished", "job_id": "abcdef123456", "status": "completed"}
        )
        text = tracker.render().plain
        self.assertIn("空闲", text)
        self.assertIn("任务 abcdef12 已完成", text)

    def test_stale_job_events_do_not_clobber_current_job(self):
        tracker = StatusTracker()
        tracker.handle_event({"type": "job_started", "job_id": "current-job", "steps": 8})
        tracker.handle_event(
            {"type": "step_progress", "job_id": "other-job", "step": 9, "total": 9}
        )
        tracker.handle_event({"type": "job_finished", "job_id": "other-job", "status": "cancelled"})
        text = tracker.render().plain
        self.assertIn("current-", text)
        self.assertNotIn("9/9", text)

    def test_seconds_per_step_fallback_and_minute_eta(self):
        tracker = StatusTracker()
        tracker.handle_event({"type": "job_started", "job_id": "job-1", "steps": 20})
        tracker.handle_event(
            {
                "type": "step_progress",
                "job_id": "job-1",
                "step": 1,
                "total": 20,
                "seconds_per_step": 8.234,
                "eta_seconds": 155.6,
            }
        )
        text = tracker.render().plain
        self.assertIn("8.23 s/it", text)
        self.assertIn("ETA 2m36s", text)

    def test_unknown_events_ignored(self):
        tracker = StatusTracker()
        tracker.handle_event({"type": "preview_image", "job_id": "job-1", "data": "..."})
        tracker.handle_event({"type": "job_error", "job_id": "job-1", "error": "x"})
        self.assertIn("空闲", tracker.render().plain)


class ConsoleStatusBarToggleTests(unittest.TestCase):
    def test_non_tty_never_activates(self):
        with (
            mock.patch.object(sys.stderr, "isatty", return_value=False),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop(console_status.ENV_FLAG, None)
            bar = ConsoleStatusBar()
            self.assertFalse(bar.start())
            self.assertFalse(bar.active)
            bar.stop()  # no-op，不应抛异常

    def test_env_flag_disables_even_on_tty(self):
        with (
            mock.patch.object(sys.stderr, "isatty", return_value=True),
            mock.patch.dict(os.environ, {console_status.ENV_FLAG: "0"}),
        ):
            bar = ConsoleStatusBar()
            self.assertFalse(bar.start())


class LiveIntegrationTests(unittest.TestCase):
    def test_auto_refresh_pulls_fresh_renderable(self):
        """回归：Live 必须通过 get_renderable 回调拉取最新状态，
        不能定格在启动时的渲染帧（曾导致状态栏一直显示"空闲"）。"""
        import io
        import re
        import time

        from rich.console import Console
        from rich.live import Live

        buffer = io.StringIO()
        tracker = StatusTracker()
        with mock.patch.dict(os.environ, {"TERM": "xterm"}):
            # Console 必须在 TERM patch 内创建：is_interactive 在构造时固化
            console = Console(file=buffer, force_terminal=True, width=100)
            with Live(
                console=console,
                refresh_per_second=20,
                transient=True,
                get_renderable=tracker.render,
            ):
                tracker.handle_event(
                    {"type": "job_started", "job_id": "abcdef123456", "steps": 8}
                )
                tracker.handle_event(
                    {
                        "type": "step_progress",
                        "job_id": "abcdef123456",
                        "step": 3,
                        "total": 8,
                        "stage": "sampling",
                        "steps_per_second": 1.5,
                    }
                )
                # 不调用 live.update()，等自动刷新线程拉取
                time.sleep(0.2)
        content = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buffer.getvalue())
        self.assertIn("3/8", content)
        self.assertIn("采样中", content)


if __name__ == "__main__":
    unittest.main()
