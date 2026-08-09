import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from diffusion_workbench_api import create_app
from fake_api_core import FakeApiCore


class LifespanTests(unittest.TestCase):
    def test_lifespan_creates_one_core_enables_preview_and_shuts_down(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            created = []

            def factory():
                core = FakeApiCore(Path(temp_dir))
                created.append(core)
                return core

            app = create_app(core_factory=factory)
            with TestClient(app) as client:
                self.assertEqual(len(created), 1)
                core = created[0]
                self.assertIs(app.state.core, core)
                self.assertIsNotNone(core.sink)
                self.assertTrue(core.preview_enabled)
                response = client.get("/api/v1/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": "ok"})
                self.assertFalse(core.closed)
            self.assertTrue(core.closed)

    def test_core_is_created_once_across_requests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            created = []

            def factory():
                core = FakeApiCore(Path(temp_dir))
                created.append(core)
                return core

            with TestClient(create_app(core_factory=factory)) as client:
                client.get("/api/v1/status")
                client.get("/api/v1/resources/zit/diffusion")
                self.assertEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
