import asyncio
import tempfile
import threading
import unittest
from pathlib import Path

from textual.widgets import Input, RichLog, Static

from diffusion_workbench.tui import WorkbenchApp
from diffusion_workbench_core.config import (
    ComfyConfig,
    ModeResources,
    WorkbenchConfig,
)
from diffusion_workbench_core.domain import Mode, ModelLoader


class TuiCore:
    def __init__(self, root: Path):
        self.config = WorkbenchConfig(
            path=root / "workbench.yaml",
            comfyui=ComfyConfig(root, root / "python.exe"),
            resources={
                Mode.ZIT: ModeResources(
                    (), (), (root / "zit-te.safetensors",), "stable_diffusion"
                ),
                Mode.KREA2: ModeResources(
                    (), (), (root / "krea-te.safetensors",), "krea2"
                ),
                Mode.ZIB: ModeResources(
                    (), (), (root / "zib-te.safetensors",), "stable_diffusion"
                ),
                Mode.SDXL: ModeResources((), (), (), None, ModelLoader.CHECKPOINT),
            },
            output_dir=root / "output",
            database=root / "jobs.sqlite3",
            worker_timeout_seconds=300,
        )
        self.sink = lambda _event: None
        self.closed = False
        self.shutdown_calls = 0
        self.shutdown_started = threading.Event()
        self.release_shutdown = threading.Event()
        self.release_shutdown.set()

    def set_event_sink(self, sink):
        self.sink = sink

    def runtime_status(self):
        return {
            "queue": 1,
            "running": "job-1",
            "worker": "ready",
            "pid": 123,
            "gpu": "12.00/15.92 GiB",
        }

    def shutdown(self):
        self.shutdown_calls += 1
        self.shutdown_started.set()
        self.release_shutdown.wait(timeout=2)
        self.closed = True


class WorkbenchTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_exit_shows_shutdown_message_before_waiting_for_core(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = TuiCore(Path(temp_dir))
            core.release_shutdown.clear()
            app = WorkbenchApp(core)

            async with app.run_test(size=(100, 30)) as pilot:
                command = app.query_one("#command", Input)
                command.value = "/exit"
                press = asyncio.create_task(pilot.press("enter"))

                started = await asyncio.to_thread(core.shutdown_started.wait, 1)
                self.assertTrue(started)
                await pilot.pause()

                log = app.query_one("#log", RichLog)
                self.assertIn(
                    "正在卸载资源", "\n".join(str(line) for line in log.lines)
                )
                self.assertTrue(command.disabled)
                self.assertTrue(app.is_running)

                app.on_unmount()
                self.assertEqual(core.shutdown_calls, 1)

                core.release_shutdown.set()
                await press
                await pilot.pause()

            self.assertTrue(core.closed)
            self.assertEqual(core.shutdown_calls, 1)

    async def test_all_generation_stages_keep_sampling_progress_visible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = TuiCore(Path(temp_dir))
            app = WorkbenchApp(core)

            async with app.run_test(size=(100, 30)) as pilot:
                progress = app.query_one("#progress", Static)
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "job_started",
                        "job_id": "job-1",
                        "seed": 42,
                        "steps": 12,
                    },
                )
                await pilot.pause()
                self.assertIn("启动推理 Worker", str(progress.render()))
                self.assertIn("未开始/12", str(progress.render()))

                for stage, label in (
                    ("loading_model", "加载模型资源"),
                    ("prompt", "编码提示词"),
                    ("latent", "准备 latent"),
                ):
                    await asyncio.to_thread(
                        core.sink,
                        {
                            "type": "stage_progress",
                            "job_id": "job-1",
                            "stage": stage,
                            "total": 12,
                        },
                    )
                    await pilot.pause()
                    self.assertIn(label, str(progress.render()))
                    self.assertIn("采样步数: 未开始/12", str(progress.render()))

                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "stage_progress",
                        "job_id": "job-1",
                        "stage": "sampling",
                        "total": 12,
                    },
                )
                await pilot.pause()
                self.assertIn("采样步数: 0/12", str(progress.render()))

                for stage, label in (
                    ("vae", "VAE 解码"),
                    ("saving", "保存图片"),
                    ("saved", "图片已保存"),
                ):
                    await asyncio.to_thread(
                        core.sink,
                        {
                            "type": "stage_progress",
                            "job_id": "job-1",
                            "stage": stage,
                            "total": 12,
                        },
                    )
                    await pilot.pause()
                    self.assertIn(label, str(progress.render()))
                    self.assertIn("采样步数: 12/12", str(progress.render()))

                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "job_started",
                        "job_id": "job-2",
                        "seed": 43,
                        "steps": 10,
                    },
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "job_finished",
                        "job_id": "job-2",
                        "status": "cancelled",
                        "steps": 10,
                    },
                )
                await pilot.pause()
                self.assertIn("任务已取消", str(progress.render()))
                self.assertIn("采样步数: 未开始/10", str(progress.render()))

    async def test_progress_from_worker_thread_does_not_block_command_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = TuiCore(Path(temp_dir))
            app = WorkbenchApp(core)

            async with app.run_test(size=(100, 30)) as pilot:
                progress = app.query_one("#progress", Static)
                self.assertIn("当前阶段: 等待任务", str(progress.render()))
                self.assertIn("采样步数: 未开始/8", str(progress.render()))

                command = app.query_one("#command", Input)
                command.value = "/prompt first prompt"
                await pilot.press("enter")
                self.assertEqual(app.session.prompt, "first prompt")

                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "queue_progress",
                        "queued": 1,
                        "running": "job-1",
                        "sequence": 2,
                    },
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "queue_progress",
                        "queued": 2,
                        "running": None,
                        "sequence": 1,
                    },
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "job_started",
                        "job_id": "job-1",
                        "seed": 42,
                        "steps": 8,
                    },
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "step_progress",
                        "job_id": "job-1",
                        "step": 3,
                        "total": 8,
                        "seconds_per_step": 2.0,
                        "steps_per_second": 0.5,
                        "eta_seconds": 10.0,
                    },
                )
                await pilot.pause()

                command.value = "/prompt changed while running"
                await pilot.press("enter")

                self.assertEqual(app.session.prompt, "changed while running")
                self.assertEqual(app.step, 3)
                self.assertEqual(app.total_steps, 8)
                self.assertEqual(app.queue_waiting, 1)
                self.assertIn(
                    "当前阶段: 采样中", str(progress.render())
                )
                self.assertIn(
                    "采样步数: 3/8", str(progress.render())
                )
                self.assertIn("速度: 2.00 秒/步", str(progress.render()))
                self.assertIn("剩余: 10.0 秒", str(progress.render()))

                await asyncio.to_thread(
                    core.sink,
                    {"type": "stage_progress", "job_id": "job-1", "stage": "vae"},
                )
                await pilot.pause()
                self.assertIn(
                    "当前阶段: VAE 解码", str(progress.render())
                )
                self.assertIn(
                    "采样步数: 8/8", str(progress.render())
                )

                await asyncio.to_thread(
                    core.sink,
                    {"type": "stage_progress", "job_id": "job-1", "stage": "saving"},
                )
                await pilot.pause()
                self.assertIn("当前阶段: 保存图片", str(progress.render()))
                self.assertIn("采样步数: 8/8", str(progress.render()))

                await asyncio.to_thread(
                    core.sink,
                    {"type": "job_error", "job_id": "job-1", "error": "failed"},
                )
                await asyncio.to_thread(
                    core.sink,
                    {"type": "job_finished", "job_id": "job-1", "status": "failed"},
                )
                await pilot.pause()
                command.value = "/prompt still responsive"
                await pilot.press("enter")
                self.assertEqual(app.session.prompt, "still responsive")
                self.assertIn(
                    "生成失败", str(progress.render())
                )

            self.assertTrue(core.closed)

    async def test_failure_before_job_started_does_not_reuse_previous_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = TuiCore(Path(temp_dir))
            app = WorkbenchApp(core)

            async with app.run_test(size=(100, 30)) as pilot:
                progress = app.query_one("#progress", Static)
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "job_started",
                        "job_id": "completed-job",
                        "seed": 42,
                        "steps": 8,
                    },
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "stage_progress",
                        "job_id": "completed-job",
                        "stage": "saved",
                        "total": 8,
                    },
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "job_finished",
                        "job_id": "completed-job",
                        "status": "completed",
                        "steps": 8,
                    },
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "queue_progress",
                        "queued": 0,
                        "running": "failed-job",
                        "steps": 12,
                        "sequence": 1,
                    },
                )
                await pilot.pause()
                self.assertIn("当前阶段: 准备任务", str(progress.render()))
                self.assertIn("采样步数: 未开始/12", str(progress.render()))

                await asyncio.to_thread(
                    core.sink,
                    {"type": "job_error", "job_id": "failed-job", "error": "seed failed"},
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "job_finished",
                        "job_id": "failed-job",
                        "status": "failed",
                        "steps": 12,
                    },
                )
                await pilot.pause()

                self.assertIn("当前阶段: 生成失败", str(progress.render()))
                self.assertIn("采样步数: 未开始/12", str(progress.render()))


if __name__ == "__main__":
    unittest.main()
