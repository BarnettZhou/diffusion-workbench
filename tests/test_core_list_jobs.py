import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from diffusion_workbench_core.domain import (
    GenerationSettings,
    Mode,
    ResourceItem,
)
from diffusion_workbench_core.storage import JobStore


def _settings(mode: Mode = Mode.ZIT) -> GenerationSettings:
    root = Path("models")
    return GenerationSettings(
        mode=mode,
        model=ResourceItem(1, root / "model.safetensors"),
        vae=ResourceItem(1, root / "vae.safetensors"),
        text_encoder=root / "te.safetensors",
        clip_type="stable_diffusion",
        prompt="a prompt",
    )


class JobStoreListJobsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.store = JobStore(root / "jobs.sqlite3", root / "output")

    def tearDown(self):
        self._tmp.cleanup()

    def _create(self, count: int, mode: Mode = Mode.ZIT, start=None):
        start = start or datetime(2026, 7, 27, 10, 0, 0).astimezone()
        return self.store.create_jobs(_settings(mode), count, submitted_at=start)

    def test_paginates_with_stable_cursor(self):
        base = datetime(2026, 7, 27, 10, 0, 0).astimezone()
        created = []
        for offset in range(3):
            created.extend(self._create(1, start=base + timedelta(minutes=offset)))

        first_page, cursor = self.store.list_jobs(limit=2)
        self.assertEqual([job.id for job in first_page], [created[2].id, created[1].id])
        self.assertIsNotNone(cursor)

        second_page, next_cursor = self.store.list_jobs(limit=2, cursor=cursor)
        self.assertEqual([job.id for job in second_page], [created[0].id])
        self.assertIsNone(next_cursor)

    def test_filters_by_status_and_mode(self):
        zit_jobs = self._create(2)
        self._create(1, mode=Mode.KREA2)
        self.store.mark_cancelled(zit_jobs[0].id)

        completed, _ = self.store.list_jobs(status="queued")
        self.assertEqual({job.mode for job in completed}, {Mode.ZIT, Mode.KREA2})
        self.assertEqual(len(completed), 2)

        cancelled, _ = self.store.list_jobs(status="cancelled")
        self.assertEqual([job.id for job in cancelled], [zit_jobs[0].id])

        krea2, _ = self.store.list_jobs(mode=Mode.KREA2)
        self.assertEqual(len(krea2), 1)
        self.assertEqual(krea2[0].mode, Mode.KREA2)

    def test_rejects_invalid_cursor_and_limit(self):
        with self.assertRaises(ValueError):
            self.store.list_jobs(cursor="no-separator")
        with self.assertRaises(ValueError):
            self.store.list_jobs(limit=0)

if __name__ == "__main__":
    unittest.main()
