import unittest
import base64

from PIL import Image

from diffusion_workbench_core.comfy_worker import (
    ComfyWorker,
    encode_preview_image,
    sampling_progress_payload,
)


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
    def test_zib_sampling_settings_are_valid(self):
        ComfyWorker._validate(
            {
                "mode": "zib",
                "width": 1024,
                "height": 1024,
                "steps": 40,
                "cfg": 4,
                "sampler": "dpmpp_2m_sde",
                "scheduler": "sgm_uniform",
            }
        )

    def test_rejects_unknown_sampling_options(self):
        command = {
            "mode": "zib",
            "width": 1024,
            "height": 1024,
            "steps": 10,
            "cfg": 1,
            "sampler": "unknown",
            "scheduler": "simple",
        }

        with self.assertRaisesRegex(ValueError, "sampler"):
            ComfyWorker._validate(command)

    def test_sampling_progress_contains_speed_and_eta(self):
        progress = sampling_progress_payload(
            step=2,
            total_steps=8,
            started_at=10.0,
            previous_step_at=13.5,
            now=16.0,
        )

        self.assertEqual(progress["step"], 3)
        self.assertEqual(progress["elapsed_seconds"], 6.0)
        self.assertEqual(progress["step_seconds"], 2.5)
        self.assertEqual(progress["seconds_per_step"], 2.0)
        self.assertEqual(progress["steps_per_second"], 0.5)
        self.assertEqual(progress["eta_seconds"], 10.0)

    def test_preview_is_a_base64_jpeg_event_payload(self):
        class Previewer:
            @staticmethod
            def decode_latent_to_preview(_x0):
                return Image.new("RGB", (12, 8), "red")

        payload = encode_preview_image(Previewer(), object())

        self.assertEqual(payload["mime_type"], "image/jpeg")
        self.assertEqual((payload["width"], payload["height"]), (12, 8))
        self.assertTrue(base64.b64decode(payload["data"]).startswith(b"\xff\xd8"))

    def test_generation_matches_comfy_executor_inference_mode_boundary(self):
        worker = object.__new__(ComfyWorker)
        worker.torch = FakeTorch()
        worker._validate = lambda _command: None
        worker.available_samplers = {"euler"}
        worker.available_schedulers = {"simple"}

        def generate_inside_mode(_command):
            self.assertTrue(worker.torch.enabled)
            return {"type": "result"}

        worker._generate = generate_inside_mode

        self.assertEqual(
            worker.generate({"sampler": "euler", "scheduler": "simple"}),
            {"type": "result"},
        )
        self.assertFalse(worker.torch.enabled)


if __name__ == "__main__":
    unittest.main()
