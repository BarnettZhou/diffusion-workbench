import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
import threading
from pathlib import Path
from unittest.mock import patch

from diffusion_workbench_core.catalog import ResourceCatalog
from diffusion_workbench_core.config import load_config
from diffusion_workbench_core.domain import Mode, ResourceKind
from diffusion_workbench_core.domain import (
    GenerationSettings,
    ResourceItem,
    UpscaleMethod,
    UpscaleSettings,
)
from diffusion_workbench_core.storage import JobStore
from diffusion_workbench_core.controller import GenerationController
from diffusion_workbench_core.runtime import GenerationCancelled
from diffusion_workbench_core.instance_lock import InstanceLock
from diffusion_workbench_core.core import WorkbenchCore
from diffusion_workbench_core.config import ComfyConfig, ModeResources, WorkbenchConfig


class ResourceCatalogTests(unittest.TestCase):
    def test_video_diffusion_catalog_can_include_gguf_without_exposing_gguf_vae(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("high.gguf", "low.safetensors", "ignore.txt"):
                (root / name).touch()

            models = ResourceCatalog._list_model_files((root,), include_gguf=True)
            vaes = ResourceCatalog._list_model_files((root,))

            self.assertEqual(
                [item.path.name for item in models],
                ["high.gguf", "low.safetensors"],
            )
            self.assertEqual([item.path.name for item in vaes], ["low.safetensors"])

    def test_config_can_omit_modes_that_are_not_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "workbench.yaml"
            config_path.write_text(
                f"""
comfyui:
  root: {root.as_posix()}
  python: {(root / 'python.exe').as_posix()}
resources:
  zit:
    diffusion: [{root.as_posix()}]
    vae: [{root.as_posix()}]
    text_encoder: {(root / 'te.safetensors').as_posix()}
    clip_type: stable_diffusion
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(tuple(config.resources), (Mode.ZIT,))

    def test_yaml_catalog_supports_multiple_paths_and_persistent_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "models-a"
            second = root / "models-b"
            vae = root / "vae"
            upscale_models = root / "upscale-models"
            first.mkdir()
            second.mkdir()
            vae.mkdir()
            upscale_models.mkdir()
            (first / "zeta.safetensors").touch()
            (second / "alpha.safetensors").touch()
            (second / "ignore.txt").touch()
            (vae / "ae.safetensors").touch()
            (upscale_models / "4x-UltraSharp.pth").touch()
            (upscale_models / "realesrgan.safetensors").touch()
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
  zib:
    diffusion: [{first.as_posix()}]
    vae: [{vae.as_posix()}]
    text_encoder: {text_encoder.as_posix()}
    clip_type: stable_diffusion
  sdxl:
    model_loader: checkpoint
    diffusion: [{(first / 'zeta.safetensors').as_posix()}]
upscaling:
  models: [{upscale_models.as_posix()}]
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
            self.assertEqual(
                [item.path.name for item in catalog.list(Mode.ZIB, ResourceKind.DIFFUSION)],
                ["zeta.safetensors"],
            )
            self.assertEqual(
                [item.path.name for item in catalog.list(Mode.SDXL, ResourceKind.DIFFUSION)],
                ["zeta.safetensors"],
            )
            self.assertEqual(
                [item.path.name for item in catalog.list_upscale_models()],
                ["4x-UltraSharp.pth", "realesrgan.safetensors"],
            )

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

    def test_list_prunes_alias_of_deleted_model_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = root / "models"
            vae = root / "vae"
            models.mkdir()
            vae.mkdir()
            (models / "alpha.safetensors").touch()
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
    diffusion: [{models.as_posix()}]
    vae: [{vae.as_posix()}]
    text_encoder: {text_encoder.as_posix()}
    clip_type: stable_diffusion
output_dir: output
database: jobs.sqlite3
worker_timeout_seconds: 300
""",
                encoding="utf-8",
            )

            config = load_config(config_path)
            store = JobStore(config.database, config.output_dir)
            catalog = ResourceCatalog(config, store)
            items = catalog.list(Mode.ZIT, ResourceKind.DIFFUSION)
            catalog.set_alias(Mode.ZIT, ResourceKind.DIFFUSION, items[0].path, "fast")
            self.assertEqual(
                store.get_aliases(Mode.ZIT, ResourceKind.DIFFUSION),
                {items[0].path: "fast"},
            )

            # 模型文件从磁盘删除后,再次列表自动清理索引中对应的别名记录
            items[0].path.unlink()
            self.assertEqual(catalog.list(Mode.ZIT, ResourceKind.DIFFUSION), [])
            self.assertEqual(store.get_aliases(Mode.ZIT, ResourceKind.DIFFUSION), {})

            # 清理后同一别名可重新分配给新文件
            new_model = models / "beta.safetensors"
            new_model.touch()
            store.set_alias(Mode.ZIT, ResourceKind.DIFFUSION, new_model, "fast")
            self.assertEqual(
                store.get_aliases(Mode.ZIT, ResourceKind.DIFFUSION),
                {new_model: "fast"},
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
                negative_prompt="blurry",
                width=576,
                height=576,
                steps=8,
                seed=-1,
                upscale=UpscaleSettings(
                    enabled=True,
                    method=UpscaleMethod.LATENT_HIRES,
                    scale=2.0,
                    interpolation="bislerp",
                    steps=9,
                    start_step=4,
                ),
            )
            submitted = datetime(2026, 7, 26, 12, 30, tzinfo=timezone.utc)

            first_batch = store.create_jobs(settings, 2, submitted_at=submitted)
            next_batch = store.create_jobs(settings, 1, submitted_at=submitted)
            persisted = store.get_job(first_batch[0].id)

            self.assertEqual(first_batch[0].batch_id, first_batch[1].batch_id)
            self.assertIsNotNone(first_batch[0].batch_id)
            self.assertIsNone(next_batch[0].batch_id)
            self.assertEqual(first_batch[0].output_path, root / "output" / "2026-07-26" / "krea2-00001.png")
            self.assertEqual(
                first_batch[0].upscaled_output_path,
                root / "output" / "2026-07-26" / "krea2-00001-upscale.png",
            )
            self.assertEqual(first_batch[1].output_path.name, "krea2-00002.png")
            self.assertEqual(next_batch[0].output_path.name, "krea2-00003.png")
            self.assertEqual(persisted.prompt, "portrait")
            self.assertEqual(persisted.negative_prompt, "blurry")
            self.assertEqual(persisted.model_path, model)
            self.assertTrue(persisted.upscale.enabled)
            self.assertEqual(persisted.upscale.method, UpscaleMethod.LATENT_HIRES)
            self.assertEqual(persisted.upscale.steps, 9)
            self.assertEqual(persisted.upscale.start_step, 4)
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


class SkippableRuntime(RecordingRuntime):
    def __init__(self):
        super().__init__()
        self.first_started = threading.Event()
        self.first_cancelled = threading.Event()
        self.cancel_calls = 0

    def generate(self, job, progress, stage, preview=None):
        if not self.jobs:
            self.jobs.append(job)
            self.first_started.set()
            self.first_cancelled.wait(5)
            raise GenerationCancelled("skipped")
        return super().generate(job, progress, stage, preview)

    def cancel(self):
        self.cancel_calls += 1
        self.first_cancelled.set()


class CancelFailureRuntime(RecordingRuntime):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, job, progress, stage, preview=None):
        self.started.set()
        self.release.wait(5)
        return super().generate(job, progress, stage, preview)

    def cancel(self):
        raise RuntimeError("cancel failed")


class InterleavedCancelFailureRuntime(CancelFailureRuntime):
    def __init__(self):
        super().__init__()
        self.first_cancel_started = threading.Event()
        self.release_first_cancel = threading.Event()
        self._cancel_calls = 0
        self._cancel_lock = threading.Lock()

    def cancel(self):
        with self._cancel_lock:
            self._cancel_calls += 1
            call = self._cancel_calls
        if call == 1:
            self.first_cancel_started.set()
            self.release_first_cancel.wait(5)
            raise RuntimeError("stop cancel failed")
        raise RuntimeError("skip cancel failed")


class BlockingCompletionStore(JobStore):
    def __init__(self, database, output_dir):
        super().__init__(database, output_dir)
        self.completion_started = threading.Event()
        self.release_completion = threading.Event()

    def mark_completed(self, job_id, completed_at, duration_seconds):
        self.completion_started.set()
        self.release_completion.wait(5)
        super().mark_completed(job_id, completed_at, duration_seconds)


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

    def test_default_random_seed_is_safe_for_javascript_numbers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = RecordingRuntime()
            with patch(
                "diffusion_workbench_core.controller.secrets.randbelow",
                return_value=123,
            ) as randbelow:
                controller = GenerationController(runtime, store)
                job = controller.submit(self.make_settings(root), 1)[0]
                controller.wait_idle()
                controller.shutdown()

            randbelow.assert_called_once_with(2**53)
            self.assertEqual(store.get_job(job.id).seed, 123)

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

    def test_skip_current_cancels_running_job_and_runs_next_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = SkippableRuntime()
            events = []
            controller = GenerationController(
                runtime, store, event_sink=events.append, seed_source=lambda: 42
            )

            jobs = controller.submit(self.make_settings(root), 2)
            self.assertTrue(runtime.first_started.wait(2))
            skipped_job_id = controller.skip_current()
            controller.wait_idle()
            statuses = [store.get_job(job.id).status for job in jobs]
            controller.shutdown()

            self.assertEqual(skipped_job_id, jobs[0].id)
            self.assertEqual(statuses, ["cancelled", "completed"])
            self.assertEqual([job.id for job in runtime.jobs], [job.id for job in jobs])
            self.assertEqual(runtime.cancel_calls, 1)
            finished = [
                (event["job_id"], event["status"])
                for event in events
                if event["type"] == "job_finished"
            ]
            self.assertEqual(
                finished,
                [(jobs[0].id, "cancelled"), (jobs[1].id, "completed")],
            )

    def test_skip_current_with_no_waiting_job_leaves_controller_idle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = SkippableRuntime()
            controller = GenerationController(runtime, store, seed_source=lambda: 42)

            job = controller.submit(self.make_settings(root), 1)[0]
            self.assertTrue(runtime.first_started.wait(2))
            self.assertEqual(controller.skip_current(), job.id)
            controller.wait_idle()
            status = controller.status()
            persisted = store.get_job(job.id)
            controller.shutdown()

            self.assertEqual(persisted.status, "cancelled")
            self.assertEqual(status["queue"], 0)
            self.assertIsNone(status["running"])
            self.assertEqual(runtime.cancel_calls, 1)

    def test_skip_current_is_noop_when_idle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = SkippableRuntime()
            controller = GenerationController(runtime, store)

            self.assertIsNone(controller.skip_current())
            controller.shutdown()

            self.assertEqual(runtime.cancel_calls, 0)

    def test_skip_current_reports_cancel_failure_and_job_can_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = CancelFailureRuntime()
            events = []
            controller = GenerationController(
                runtime, store, event_sink=events.append, seed_source=lambda: 42
            )
            job = controller.submit(self.make_settings(root), 1)[0]
            self.assertTrue(runtime.started.wait(2))

            with self.assertRaisesRegex(RuntimeError, "跳过任务失败"):
                controller.skip_current()
            runtime.release.set()
            controller.wait_idle()
            persisted = store.get_job(job.id)
            controller.shutdown()

            self.assertEqual(persisted.status, "completed")
            errors = [event for event in events if event["type"] == "job_error"]
            self.assertIn("跳过失败", errors[0]["error"])

    def test_skip_current_is_rejected_after_completion_is_claimed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = BlockingCompletionStore(root / "jobs.sqlite3", root / "output")
            runtime = RecordingRuntime()
            controller = GenerationController(runtime, store, seed_source=lambda: 42)
            job = controller.submit(self.make_settings(root), 1)[0]
            self.assertTrue(store.completion_started.wait(2))

            self.assertIsNone(controller.skip_current())
            store.release_completion.set()
            controller.wait_idle()
            persisted = store.get_job(job.id)
            controller.shutdown()

            self.assertEqual(persisted.status, "completed")

    def test_failed_skip_does_not_remove_concurrent_stop_intent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            runtime = InterleavedCancelFailureRuntime()
            controller = GenerationController(runtime, store, seed_source=lambda: 42)
            job = controller.submit(self.make_settings(root), 1)[0]
            self.assertTrue(runtime.started.wait(2))
            stop_thread = threading.Thread(target=controller.stop)
            stop_thread.start()
            self.assertTrue(runtime.first_cancel_started.wait(2))

            with self.assertRaisesRegex(RuntimeError, "skip cancel failed"):
                controller.skip_current()
            runtime.release_first_cancel.set()
            stop_thread.join(2)
            runtime.release.set()
            controller.wait_idle()
            persisted = store.get_job(job.id)
            controller.shutdown()

            self.assertFalse(stop_thread.is_alive())
            self.assertEqual(persisted.status, "cancelled")

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
