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
                Mode.ZIB: ModeResources((), (), root / "zib-te.safetensors", "stable_diffusion"),
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
            (Mode.ZIB, ResourceKind.DIFFUSION): [ResourceItem(1, root / "zib.safetensors")],
            (Mode.ZIB, ResourceKind.VAE): [ResourceItem(1, root / "zib-vae.safetensors")],
        }
        self.submitted = []
        self.stopped = False
        self.skipped_job_id = None
        self.skip_error = None
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

    def skip_current(self):
        if self.skip_error is not None:
            raise RuntimeError(self.skip_error)
        return self.skipped_job_id

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

    def test_mode_can_select_zib_and_list_its_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = CommandSession(FakeCore(Path(temp_dir)))

            selected = session.handle("/mode zib")
            models = session.handle("/model list")

            self.assertIn("zib", "\n".join(selected.lines))
            self.assertIn("zib.safetensors", "\n".join(models.lines))

    def test_sampling_commands_update_submitted_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = FakeCore(Path(temp_dir))
            session = CommandSession(core)
            session.handle("/mode zib")
            session.handle("/model set 1")
            session.handle("/vae set 1")
            session.handle("/prompt portrait")
            session.handle("/negative blurry, watermark")

            self.assertIn("dpmpp_2m_sde", "\n".join(session.handle("/sampler list").lines))
            self.assertIn("sgm_uniform", "\n".join(session.handle("/scheduler list").lines))
            session.handle("/cfg 4")
            session.handle("/steps 40")
            session.handle("/sampler set 2")
            session.handle("/scheduler set sgm_uniform")
            session.handle("/start")

            settings, _count = core.submitted[0]
            self.assertEqual(settings.mode, Mode.ZIB)
            self.assertEqual(settings.cfg, 4.0)
            self.assertEqual(settings.steps, 40)
            self.assertEqual(settings.sampler, "dpmpp_2m_sde")
            self.assertEqual(settings.scheduler, "sgm_uniform")
            self.assertEqual(settings.negative_prompt, "blurry, watermark")
            status = "\n".join(session.handle("/status").lines)
            self.assertIn("cfg: 4", status)
            self.assertIn("sampler: dpmpp_2m_sde", status)
            self.assertIn("scheduler: sgm_uniform", status)
            self.assertIn("negative: blurry, wate...", status)

    def test_sampling_commands_accept_names_and_reject_invalid_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = CommandSession(FakeCore(Path(temp_dir)))

            self.assertIn("ddim", "\n".join(session.handle("/sampler list").lines))
            self.assertIn("karras", "\n".join(session.handle("/scheduler list").lines))
            self.assertIn("ddim", "\n".join(session.handle("/sampler set ddim").lines))
            self.assertIn("karras", "\n".join(session.handle("/scheduler set karras").lines))
            self.assertIn("大于 0", "\n".join(session.handle("/cfg 0").lines))
            self.assertIn("找不到 sampler", "\n".join(session.handle("/sampler set nope").lines))

    def test_rejects_invalid_settings_and_exit_defers_core_shutdown_to_tui(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = FakeCore(Path(temp_dir))
            session = CommandSession(core)

            self.assertIn("1 到 100", "\n".join(session.handle("/steps 0").lines))
            self.assertIn("16 的倍数", "\n".join(session.handle("/size 575*576").lines))
            response = session.handle("/exit")

            self.assertTrue(response.exit_requested)
            self.assertIn("正在卸载资源", "\n".join(response.lines))
            self.assertFalse(core.closed)

    def test_skip_reports_running_job_or_idle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            core = FakeCore(Path(temp_dir))
            session = CommandSession(core)

            idle = session.handle("/skip")
            core.skipped_job_id = "job-123"
            skipped = session.handle("/skip")

            self.assertIn("没有可跳过的任务", "\n".join(idle.lines))
            self.assertIn("job-123", "\n".join(skipped.lines))
            self.assertIn("/skip", "\n".join(session.handle("/help").lines))

            core.skip_error = "跳过任务失败"
            failed = session.handle("/skip")
            self.assertIn("错误: 跳过任务失败", "\n".join(failed.lines))


if __name__ == "__main__":
    unittest.main()
