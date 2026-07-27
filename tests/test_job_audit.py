import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from diffusion_workbench_core.domain import GenerationSettings, Mode, ResourceItem
from diffusion_workbench_core.storage import JobStore


class JobAuditTests(unittest.TestCase):
    def test_completed_job_remains_readable_when_output_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3", root / "output")
            model_root = Path("models")
            settings = GenerationSettings(
                mode=Mode.ZIT,
                model=ResourceItem(1, model_root / "model.safetensors"),
                vae=ResourceItem(1, model_root / "vae.safetensors"),
                text_encoder=model_root / "te.safetensors",
                clip_type="stable_diffusion",
                prompt="a prompt",
            )
            job = store.create_jobs(settings, 1)[0]
            completed_at = datetime(2026, 7, 27, 10, 1, 0).astimezone()
            store.mark_completed(job.id, completed_at, 12.5)

            self.assertFalse(job.output_path.exists())
            loaded = store.get_job(job.id)

            self.assertEqual(loaded.status, "completed")
            self.assertEqual(loaded.prompt, "a prompt")
            self.assertEqual(loaded.seed, job.seed)
            self.assertEqual(loaded.model_path, job.model_path)
            self.assertEqual(loaded.output_path, job.output_path)


if __name__ == "__main__":
    unittest.main()
