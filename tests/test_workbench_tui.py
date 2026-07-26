import asyncio
import tempfile
import unittest
from pathlib import Path

from textual.widgets import Input, Static

from diffusion_workbench.tui import WorkbenchApp
from diffusion_workbench_core.config import (
    ComfyConfig,
    ModeResources,
    WorkbenchConfig,
)
from diffusion_workbench_core.domain import Mode


class TuiCore:
    def __init__(self, root: Path):
        self.config = WorkbenchConfig(
            path=root / "workbench.yaml",
            comfyui=ComfyConfig(root, root / "python.exe"),
            resources={
                Mode.ZIT: ModeResources(
                    (), (), root / "zit-te.safetensors", "stable_diffusion"
                ),
                Mode.KREA2: ModeResources(
                    (), (), root / "krea-te.safetensors", "krea2"
                ),
            },
            output_dir=root / "output",
            database=root / "jobs.sqlite3",
            worker_timeout_seconds=300,
        )
        self.sink = lambda _event: None
        self.closed = False

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
        self.closed = True


class WorkbenchTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_from_worker_thread_does_not_block_command_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = TuiCore(Path(temp_dir))
            app = WorkbenchApp(core)

            async with app.run_test(size=(100, 30)) as pilot:
                command = app.query_one("#command", Input)
                command.value = "/prompt first prompt"
                await pilot.press("enter")
                self.assertEqual(app.session.prompt, "first prompt")

                await asyncio.to_thread(
                    core.sink,
                    {"type": "queue_progress", "queued": 2, "running": None},
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "job_started",
                        "job_id": "job-1",
                        "seed": 42,
                    },
                )
                await asyncio.to_thread(
                    core.sink,
                    {
                        "type": "step_progress",
                        "job_id": "job-1",
                        "step": 3,
                        "total": 8,
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
                    "采样步数 3/8", str(app.query_one("#progress", Static).render())
                )

                await asyncio.to_thread(
                    core.sink,
                    {"type": "stage_progress", "job_id": "job-1", "stage": "vae"},
                )
                await pilot.pause()
                self.assertIn(
                    "VAE 处理", str(app.query_one("#progress", Static).render())
                )

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
                    "生成失败", str(app.query_one("#progress", Static).render())
                )

            self.assertTrue(core.closed)


if __name__ == "__main__":
    unittest.main()
