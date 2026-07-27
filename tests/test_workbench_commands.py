import tempfile
import unittest
from pathlib import Path

from diffusion_workbench.commands import CommandSession
from diffusion_workbench_core.config import (
    ComfyConfig,
    ModeResources,
    WorkbenchConfig,
)
from diffusion_workbench_core.domain import Mode, ResourceItem, ResourceKind


class FakeCore:
    def __init__(self, root: Path):
        self.config = WorkbenchConfig(
            path=root / "workbench.yaml",
            comfyui=ComfyConfig(root, root / "python.exe"),
            resources={
                Mode.ZIT: ModeResources((), (), root / "zit-te.safetensors", "stable_diffusion"),
                Mode.KREA2: ModeResources((), (), root / "krea-te.safetensors", "krea2"),
            },
            output_dir=root / "output",
            database=root / "jobs.sqlite3",
            worker_timeout_seconds=300,
        )
        self.items = {
            (Mode.ZIT, ResourceKind.DIFFUSION): [ResourceItem(1, root / "zit.safetensors")],
            (Mode.ZIT, ResourceKind.VAE): [ResourceItem(1, root / "zit-vae.safetensors")],
            (Mode.KREA2, ResourceKind.DIFFUSION): [ResourceItem(1, root / "krea.safetensors")],
            (Mode.KREA2, ResourceKind.VAE): [ResourceItem(1, root / "krea-vae.safetensors")],
        }
        self.submitted = []
        self.stopped = False
        self.closed = False

    def list_resources(self, mode, kind):
        return self.items[(mode, kind)]

    def set_alias(self, mode, kind, path, alias):
        items = self.items[(mode, kind)]
        self.items[(mode, kind)] = [
            ResourceItem(item.index, item.path, alias if item.path == path else item.alias)
            for item in items
        ]

    def submit(self, settings, count):
        self.submitted.append((settings, count))
        return [f"job-{index}" for index in range(count)]

    def stop(self):
        self.stopped = True

    def runtime_status(self):
        return {"queue": 0, "running": None, "gpu": "1.0/16.0 GiB", "worker": "idle"}

    def shutdown(self):
        self.closed = True


class CommandSessionTests(unittest.TestCase):
    def test_start_submits_an_immutable_snapshot_and_mode_defaults_to_zit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = FakeCore(Path(temp_dir))
            session = CommandSession(core)

            self.assertIn("mode: zit", "\n".join(session.handle("/status").lines))
            session.handle("/model set 1")
            session.handle("/vae set 1")
            session.handle("/prompt twelve3456789")
            response = session.handle("/start 2")
            session.handle("/prompt changed")

            settings, count = core.submitted[0]
            self.assertEqual(count, 2)
            self.assertEqual(settings.prompt, "twelve3456789")
            self.assertEqual(settings.mode, Mode.ZIT)
            self.assertEqual(settings.steps, 8)
            self.assertEqual(settings.seed, -1)
            self.assertIn("已加入队列", "\n".join(response.lines))

    def test_start_defaults_to_one_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = FakeCore(Path(temp_dir))
            session = CommandSession(core)
            session.handle("/model set 1")
            session.handle("/vae set 1")
            session.handle("/prompt portrait")

            response = session.handle("/start")

            self.assertEqual(core.submitted[0][1], 1)
            self.assertIn("1 个任务", "\n".join(response.lines))

    def test_mode_specific_selection_alias_and_status_prompt_truncation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = FakeCore(Path(temp_dir))
            session = CommandSession(core)
            session.handle("/model set 1")
            session.handle("/model set-alias 1 portrait")
            session.handle("/prompt abcdefghijklmnop")

            status = "\n".join(session.handle("/status").lines)
            models = "\n".join(session.handle("/model list").lines)
            switched = session.handle("/mode")

            self.assertIn("abcdefghijkl...", status)
            self.assertIn("portrait", models)
            self.assertIn("krea2", "\n".join(switched.lines))
            self.assertIsNone(session.selected_model)

    def test_rejects_invalid_settings_and_exit_defers_core_shutdown_to_tui(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = FakeCore(Path(temp_dir))
            session = CommandSession(core)

            self.assertIn("8 到 20", "\n".join(session.handle("/steps 7").lines))
            self.assertIn("16 的倍数", "\n".join(session.handle("/size 575*576").lines))
            response = session.handle("/exit")

            self.assertTrue(response.exit_requested)
            self.assertIn("正在卸载资源", "\n".join(response.lines))
            self.assertFalse(core.closed)


if __name__ == "__main__":
    unittest.main()
