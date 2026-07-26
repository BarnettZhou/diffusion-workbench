import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from diffusion_workbench_core.config import ComfyConfig, ModeResources, WorkbenchConfig
from diffusion_workbench_core.domain import JobRecord, Mode
from diffusion_workbench_core.persistent_runtime import PersistentComfyRuntime


FAKE_WORKER = r'''
import argparse, json, sys
parser = argparse.ArgumentParser(); parser.add_argument("--comfy-root"); parser.parse_args()
def emit(payload): print("DWB_EVENT=" + json.dumps(payload), flush=True)
emit({"type": "ready"})
loaded = None
for line in sys.stdin:
    command = json.loads(line)
    if command["type"] == "shutdown":
        emit({"type": "stopped"}); break
    loaded_now = loaded != command["model_path"]
    loaded = command["model_path"]
    emit({"type": "stage_progress", "job_id": command["job_id"], "stage": "sampling", "total": command["steps"]})
    emit({"type": "step_progress", "job_id": command["job_id"], "step": 1, "total": command["steps"]})
    emit({"type": "result", "job_id": command["job_id"], "loaded_model": loaded_now, "output_path": command["output_path"]})
'''


class PersistentRuntimeTests(unittest.TestCase):
    def test_reuses_worker_and_loaded_model_for_consecutive_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            worker_script = root / "fake_worker.py"
            worker_script.write_text(FAKE_WORKER, encoding="utf-8")
            config = WorkbenchConfig(
                path=root / "config.yaml",
                comfyui=ComfyConfig(root, Path(sys.executable)),
                resources={
                    Mode.ZIT: ModeResources(
                        (), (), root / "te.safetensors", "stable_diffusion"
                    )
                },
                output_dir=root / "output",
                database=root / "jobs.sqlite3",
                worker_timeout_seconds=10,
            )
            runtime = PersistentComfyRuntime(config, worker_script=worker_script)
            progress = []
            stages = []

            first = runtime.generate(
                self.make_job(root, "one"),
                lambda step, total: progress.append((step, total)),
                lambda stage, total: stages.append((stage, total)),
            )
            first_pid = runtime.status()["pid"]
            second = runtime.generate(
                self.make_job(root, "two"),
                lambda _step, _total: None,
                lambda _stage, _total: None,
            )
            second_pid = runtime.status()["pid"]
            runtime.close()

            self.assertEqual(first_pid, second_pid)
            self.assertTrue(first["loaded_model"])
            self.assertFalse(second["loaded_model"])
            self.assertEqual(progress, [(1, 8)])
            self.assertEqual(stages, [("sampling", 8)])
            self.assertEqual(runtime.status()["worker"], "stopped")

    def test_status_does_not_block_on_gpu_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            started = threading.Event()
            release = threading.Event()

            def slow_probe():
                started.set()
                release.wait(2)
                return "1.00/16.00 GiB"

            config = WorkbenchConfig(
                path=root / "config.yaml",
                comfyui=ComfyConfig(root, Path(sys.executable)),
                resources={},
                output_dir=root / "output",
                database=root / "jobs.sqlite3",
                worker_timeout_seconds=10,
            )
            with patch.object(PersistentComfyRuntime, "_gpu_memory", side_effect=slow_probe):
                runtime = PersistentComfyRuntime(config)
                self.assertTrue(started.wait(1))
                before = time.perf_counter()
                status = runtime.status()
                elapsed = time.perf_counter() - before
                release.set()
                runtime.close()

            self.assertLess(elapsed, 0.1)
            self.assertEqual(status["gpu"], "查询中")

    @staticmethod
    def make_job(root: Path, job_id: str) -> JobRecord:
        return JobRecord(
            id=job_id,
            batch_id=None,
            status="running",
            submitted_at=datetime.now(timezone.utc),
            output_path=root / f"{job_id}.png",
            mode=Mode.ZIT,
            prompt="portrait",
            model_path=root / "model.safetensors",
            vae_path=root / "vae.safetensors",
            text_encoder_path=root / "te.safetensors",
            sampler="euler",
            scheduler="simple",
            width=576,
            height=576,
            steps=8,
            seed=42,
            cfg=1.0,
        )


if __name__ == "__main__":
    unittest.main()
