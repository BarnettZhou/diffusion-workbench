import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
import threading
from pathlib import Path

from diffusion_workbench_core.catalog import ResourceCatalog
from diffusion_workbench_core.config import load_config
from diffusion_workbench_core.domain import Mode, ResourceKind
from diffusion_workbench_core.domain import GenerationSettings, ResourceItem
from diffusion_workbench_core.storage import JobStore
from diffusion_workbench_core.controller import GenerationController
from diffusion_workbench_core.runtime import GenerationCancelled
from diffusion_workbench_core.instance_lock import InstanceLock
from diffusion_workbench_core.core import WorkbenchCore
from diffusion_workbench_core.config import ComfyConfig, ModeResources, WorkbenchConfig


class ResourceCatalogTests(unittest.TestCase):
    def test_yaml_catalog_supports_multiple_paths_and_persistent_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "models-a"
            second = root / "models-b"
            vae = root / "vae"
            first.mkdir()
            second.mkdir()
            vae.mkdir()
            (first / "zeta.safetensors").touch()
            (second / "alpha.safetensors").touch()
            (second / "ignore.txt").touch()
            (vae / "ae.safetensors").touch()
            text_encoder = root / "te.safetensors"
            text_encoder.touch()
            config_path = root / "workbench.yaml"
            config_path.write_text(
                f"""
comfyui:
  root: {root.as_posix()}
  python: {text_encoder.as_posix()}
resources:
  zit:
    diffusion: [{first.as_posix()}, {second.as_posix()}]
    vae: [{vae.as_posix()}]
    text_encoder: {text_encoder.as_posix()}
    clip_type: stable_diffusion
  krea2:
    diffusion: [{first.as_posix()}]
    vae: [{vae.as_posix()}]
    text_encoder: {text_encoder.as_posix()}
    clip_type: krea2
output_dir: output
database: jobs.sqlite3
worker_timeout_seconds: 300
""",
                encoding="utf-8",
            )

            config = load_config(config_path)
            store = JobStore(config.database, config.output_dir)
            catalog = ResourceCatalog(config, store)
            models = catalog.list(Mode.ZIT, ResourceKind.DIFFUSION)
            catalog.set_alias(Mode.ZIT, ResourceKind.DIFFUSION, models[0].path, "fast")
            reloaded = ResourceCatalog(config, JobStore(config.database, config.output_dir))

            self.assertEqual([item.path.name for item in models], ["alpha.safetensors", "zeta.safetensors"])
            self.assertEqual(reloaded.list(Mode.ZIT, ResourceKind.DIFFUSION)[0].alias, "fast")
            self.assertEqual(config.output_dir, root / "output")
            self.assertEqual(config.database, root / "jobs.sqlite3")

    def test_alias_must_be_unique_within_mode_and_kind(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            store.set_alias(
                Mode.ZIT, ResourceKind.DIFFUSION, root / "one.safetensors", "fast"
            )

            with self.assertRaisesRegex(ValueError, "已被"):
                store.set_alias(
                    Mode.ZIT,
                    ResourceKind.DIFFUSION,
                    root / "two.safetensors",
                    "fast",
                )


class JobStoreTests(unittest.TestCase):
    def test_batch_reserves_daily_mode_paths_and_persists_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            model = root / "model.safetensors"
            vae = root / "vae.safetensors"
            settings = GenerationSettings(
                mode=Mode.KREA2,
                model=ResourceItem(1, model, "photo"),
                vae=ResourceItem(1, vae, "default"),
                text_encoder=root / "te.safetensors",
                clip_type="krea2",
                prompt="portrait",
                width=576,
                height=576,
                steps=8,
                seed=-1,
            )
            submitted = datetime(2026, 7, 26, 12, 30, tzinfo=timezone.utc)

            first_batch = store.create_jobs(settings, 2, submitted_at=submitted)
            next_batch = store.create_jobs(settings, 1, submitted_at=submitted)
            persisted = store.get_job(first_batch[0].id)

            self.assertEqual(first_batch[0].batch_id, first_batch[1].batch_id)
            self.assertIsNotNone(first_batch[0].batch_id)
            self.assertIsNone(next_batch[0].batch_id)
            self.assertEqual(first_batch[0].output_path, root / "output" / "2026-07-26" / "krea2-00001.png")
            self.assertEqual(first_batch[1].output_path.name, "krea2-00002.png")
            self.assertEqual(next_batch[0].output_path.name, "krea2-00003.png")
            self.assertEqual(persisted.prompt, "portrait")
            self.assertEqual(persisted.model_path, model)
            self.assertEqual(persisted.status, "queued")

    def test_existing_output_files_are_never_reused_after_database_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_date = "2026-07-26"
            output_directory = root / "output" / output_date
            output_directory.mkdir(parents=True)
            (output_directory / "krea2-00007.png").write_bytes(b"existing")
            store = JobStore(root / "jobs.sqlite3", root / "output")
            settings = GenerationSettings(
                mode=Mode.KREA2,
                model=ResourceItem(1, root / "model.safetensors"),
                vae=ResourceItem(1, root / "vae.safetensors"),
                text_encoder=root / "te.safetensors",
                clip_type="krea2",
                prompt="portrait",
            )

            job = store.create_jobs(
                settings,
                1,
                submitted_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
            )[0]

            self.assertEqual(job.output_path.name, "krea2-00008.png")

    def test_reopen_marks_incomplete_jobs_as_cancelled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            settings = GenerationSettings(
                mode=Mode.ZIT,
                model=ResourceItem(1, root / "model.safetensors"),
                vae=ResourceItem(1, root / "vae.safetensors"),
                text_encoder=root / "te.safetensors",
                clip_type="stable_diffusion",
                prompt="portrait",
            )
            jobs = store.create_jobs(settings, 2)
            store.mark_running(jobs[0].id, 42, datetime.now(timezone.utc))

            reopened = JobStore(root / "jobs.sqlite3", root / "output")
            reopened.recover_incomplete_jobs()

            self.assertEqual(reopened.get_job(jobs[0].id).status, "cancelled")
            self.assertEqual(reopened.get_job(jobs[1].id).status, "cancelled")


class RecordingRuntime:
    def __init__(self):
        self.jobs = []
        self.active = 0
        self.max_active = 0
        self.closed = False

    def generate(self, job, progress, stage, preview=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.jobs.append(job)
        stage("sampling", job.steps)
        progress(
            1,
            job.steps,
            {
                "elapsed_seconds": 0.5,
                "seconds_per_step": 0.5,
                "steps_per_second": 2.0,
                "eta_seconds": 3.5,
            },
        )
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        job.output_path.write_bytes(b"png")
        self.active -= 1
        return {"generation_seconds": 0.01}

    def cancel(self):
        pass

    def close(self):
        self.closed = True

    def status(self):
        return {"worker": "ready", "gpu": "0/16 GiB"}

    def set_preview_enabled(self, enabled):
        self.preview_enabled = enabled


class BlockingRuntime(RecordingRuntime):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def generate(self, job, progress, stage, preview=None):
        self.started.set()
        self.cancelled.wait(5)
        raise GenerationCancelled("cancelled")

    def cancel(self):
        self.cancelled.set()


class CancelErrorRuntime(BlockingRuntime):
    def cancel(self):
        raise RuntimeError("cancel failed")

    def close(self):
        self.closed = True
        self.cancelled.set()


class GenerationControllerTests(unittest.TestCase):
    def make_settings(self, root: Path) -> GenerationSettings:
        return GenerationSettings(
            mode=Mode.ZIT,
            model=ResourceItem(1, root / "model.safetensors"),
            vae=ResourceItem(1, root / "vae.safetensors"),
            text_encoder=root / "te.safetensors",
            clip_type="stable_diffusion",
            prompt="portrait",
        )

    def test_runs_queue_sequentially_and_resolves_random_seed_at_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = RecordingRuntime()
            seeds = iter([101, 102])
            events = []
            controller = GenerationController(
                runtime,
                store,
                event_sink=events.append,
                seed_source=lambda: next(seeds),
            )

            jobs = controller.submit(self.make_settings(root), 2)
            controller.wait_idle()
            persisted = [store.get_job(job.id) for job in jobs]
            controller.shutdown()

            self.assertEqual(runtime.max_active, 1)
            self.assertEqual([job.seed for job in persisted], [101, 102])
            self.assertEqual([job.status for job in persisted], ["completed", "completed"])
            self.assertTrue(all(job.duration_seconds is not None for job in persisted))
            self.assertTrue(runtime.closed)
            started = [event for event in events if event["type"] == "job_started"]
            self.assertEqual([event["steps"] for event in started], [8, 8])
            step = next(event for event in events if event["type"] == "step_progress")
            self.assertEqual(step["steps_per_second"], 2.0)
            self.assertEqual(step["eta_seconds"], 3.5)

    def test_stop_cancels_running_and_queued_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = BlockingRuntime()
            controller = GenerationController(runtime, store, seed_source=lambda: 42)

            jobs = controller.submit(self.make_settings(root), 2)
            self.assertTrue(runtime.started.wait(2))
            controller.stop()
            controller.wait_idle()
            statuses = [store.get_job(job.id).status for job in jobs]
            controller.shutdown()

            self.assertEqual(statuses, ["cancelled", "cancelled"])

    def test_cancelled_before_start_event_keeps_job_step_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = RecordingRuntime()
            events = []
            controller = GenerationController(runtime, store, event_sink=events.append)
            controller.stop()
            settings = replace(self.make_settings(root), steps=12)
            job = store.create_jobs(settings, 1)[0]

            controller._queue.put((0, job))
            controller.wait_idle()
            controller.shutdown()

            finished = [
                event
                for event in events
                if event["type"] == "job_finished" and event["job_id"] == job.id
            ]
            self.assertEqual(finished[0]["status"], "cancelled")
            self.assertEqual(finished[0]["steps"], 12)

    def test_seed_failure_does_not_kill_queue_thread_or_block_shutdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = RecordingRuntime()
            controller = GenerationController(
                runtime,
                store,
                seed_source=lambda: (_ for _ in ()).throw(RuntimeError("seed failed")),
            )

            failed = controller.submit(self.make_settings(root), 1)[0]
            controller.wait_idle()
            fixed_seed_settings = replace(self.make_settings(root), seed=42)
            completed = controller.submit(fixed_seed_settings, 1)[0]
            controller.wait_idle()
            controller.shutdown()

            self.assertEqual(store.get_job(failed.id).status, "failed")
            self.assertEqual(store.get_job(completed.id).status, "completed")

    def test_shutdown_closes_runtime_when_cancel_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = CancelErrorRuntime()
            controller = GenerationController(runtime, store, seed_source=lambda: 42)
            job = controller.submit(self.make_settings(root), 1)[0]
            self.assertTrue(runtime.started.wait(2))

            controller.shutdown()

            self.assertTrue(runtime.closed)
            self.assertEqual(store.get_job(job.id).status, "cancelled")


class WorkbenchCoreBoundaryTests(unittest.TestCase):
    def test_rejects_non_configured_text_encoder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configured_encoder = root / "fixed.safetensors"
            model_directory = root / "models"
            vae_directory = root / "vaes"
            model_directory.mkdir()
            vae_directory.mkdir()
            configured_model = model_directory / "model.safetensors"
            configured_vae = vae_directory / "vae.safetensors"
            configured_model.touch()
            configured_vae.touch()
            config = WorkbenchConfig(
                path=root / "workbench.yaml",
                comfyui=ComfyConfig(root, root / "python.exe"),
                resources={
                    Mode.ZIT: ModeResources(
                        (model_directory,),
                        (vae_directory,),
                        configured_encoder,
                        "stable_diffusion",
                    )
                },
                output_dir=root / "output",
                database=root / "jobs.sqlite3",
                worker_timeout_seconds=300,
            )
            core = object.__new__(WorkbenchCore)
            core.config = config
            core.store = JobStore(config.database, config.output_dir)
            core.catalog = ResourceCatalog(config, core.store)
            core.controller = RecordingController()
            settings = GenerationSettings(
                mode=Mode.ZIT,
                model=ResourceItem(1, root / "model.safetensors"),
                vae=ResourceItem(1, root / "vae.safetensors"),
                text_encoder=root / "other.safetensors",
                clip_type="stable_diffusion",
                prompt="portrait",
            )

            with self.assertRaisesRegex(ValueError, "text encoder 固定"):
                core.submit(settings, 1)

            wrong_model = GenerationSettings(
                mode=Mode.ZIT,
                model=ResourceItem(1, root / "outside.safetensors"),
                vae=ResourceItem(1, configured_vae),
                text_encoder=configured_encoder,
                clip_type="stable_diffusion",
                prompt="portrait",
            )
            with self.assertRaisesRegex(ValueError, "diffusion 目录"):
                core.submit(wrong_model, 1)


class RecordingController:
    def submit(self, settings, count):
        return [(settings, count)]


class InstanceLockTests(unittest.TestCase):
    def test_only_one_owner_can_hold_database_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "jobs.sqlite3.lock"
            first = InstanceLock(lock_path)
            try:
                with self.assertRaisesRegex(RuntimeError, "已有 diffusion-workbench"):
                    InstanceLock(lock_path)
            finally:
                first.close()

            second = InstanceLock(lock_path)
            second.close()


if __name__ == "__main__":
    unittest.main()
