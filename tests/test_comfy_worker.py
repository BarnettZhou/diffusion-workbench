import unittest

from diffusion_workbench_core.comfy_worker import ComfyWorker


class FakeInferenceMode:
    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        self.torch.enabled = True

    def __exit__(self, _exc_type, _exc, _traceback):
        self.torch.enabled = False


class FakeTorch:
    def __init__(self):
        self.enabled = False

    def inference_mode(self):
        return FakeInferenceMode(self)


class ComfyWorkerTests(unittest.TestCase):
    def test_generation_matches_comfy_executor_inference_mode_boundary(self):
        worker = object.__new__(ComfyWorker)
        worker.torch = FakeTorch()
        worker._validate = lambda _command: None

        def generate_inside_mode(_command):
            self.assertTrue(worker.torch.enabled)
            return {"type": "result"}

        worker._generate = generate_inside_mode

        self.assertEqual(worker.generate({}), {"type": "result"})
        self.assertFalse(worker.torch.enabled)


if __name__ == "__main__":
    unittest.main()
