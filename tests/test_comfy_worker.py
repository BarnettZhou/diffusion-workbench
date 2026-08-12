import unittest
import base64
from pathlib import Path

from PIL import Image

from diffusion_workbench_core.comfy_worker import (
    ComfyWorker,
    encode_preview_image,
    resolve_upscale_settings,
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
    def test_gguf_diffusion_uses_installed_custom_loader(self):
        calls = []

        class BuiltinLoader:
            def load_unet(self, name, dtype):
                calls.append(("builtin", name, dtype))
                return ("builtin-model",)

        class GgufLoader:
            def load_unet(self, name):
                calls.append(("gguf", name))
                return ("gguf-model",)

        worker = object.__new__(ComfyWorker)
        worker.nodes = type("Nodes", (), {"UNETLoader": BuiltinLoader})
        worker._register_exact = lambda _category, path: path.name
        worker._ensure_gguf_unet_loader = lambda: GgufLoader

        model = worker._load_diffusion_model(Path("wan-high.gguf"))

        self.assertEqual(model, "gguf-model")
        self.assertEqual(calls, [("gguf", "wan-high.gguf")])

    def test_safetensors_diffusion_keeps_builtin_loader(self):
        calls = []

        class BuiltinLoader:
            def load_unet(self, name, dtype):
                calls.append((name, dtype))
                return ("builtin-model",)

        worker = object.__new__(ComfyWorker)
        worker.nodes = type("Nodes", (), {"UNETLoader": BuiltinLoader})
        worker._register_exact = lambda _category, path: path.name

        model = worker._load_diffusion_model(Path("wan-high.safetensors"))

        self.assertEqual(model, "builtin-model")
        self.assertEqual(calls, [("wan-high.safetensors", "default")])

    def test_wan_video_command_validates_length_and_fixed_denoise(self):
        command = {
            "video_model": "wan2.2-ti2v-5b",
            "clip_type": "wan",
            "width": 704,
            "height": 960,
            "duration_seconds": 5,
            "fps": 24,
            "length": 121,
            "steps": 20,
            "cfg": 5,
            "sampler": "uni_pc",
            "scheduler": "simple",
            "denoise": 1,
        }

        ComfyWorker._validate_video(None, command)
        command["length"] = 120
        with self.assertRaisesRegex(ValueError, "length"):
            ComfyWorker._validate_video(None, command)

    def test_h3_video_command_validates_native_av_contract(self):
        command = {
            "video_model": "minimax-h3",
            "clip_type": "minimax",
            "audio_vae_path": "audio.safetensors",
            "width": 608,
            "height": 352,
            "duration_seconds": 5,
            "fps": 24,
            "length": 124,
            "steps": 8,
            "cfg": 1,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "denoise": 1,
        }

        ComfyWorker._validate_video(None, command)
        command["fps"] = 25
        with self.assertRaisesRegex(ValueError, "帧率固定"):
            ComfyWorker._validate_video(None, command)

    def test_save_video_forwards_native_audio(self):
        captured = {}

        class Video:
            def save_to(self, path, **kwargs):
                captured["path"] = path
                captured.update(kwargs)
                Path(path).touch()

        class Input:
            @staticmethod
            def VideoFromComponents(components, bit_depth):
                captured["components"] = components
                captured["bit_depth"] = bit_depth
                return Video()

        class Components:
            def __init__(self, images, frame_rate, audio):
                self.images = images
                self.frame_rate = frame_rate
                self.audio = audio

        class Types:
            VideoComponents = Components
            VideoContainer = type("VideoContainer", (), {"MP4": "mp4"})
            VideoCodec = type("VideoCodec", (), {"H264": "h264"})

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            worker = object.__new__(ComfyWorker)
            worker.Types = Types
            worker.InputImpl = Input
            output = Path(temp_dir) / "result.mp4"
            worker._save_video("frames", output, 24, {}, "job", audio="stereo")

        self.assertEqual(captured["components"].audio, "stereo")

    def test_second_video_sampling_stage_uses_zero_noise(self):
        class Samples:
            dtype = "dtype"
            layout = "layout"

            @staticmethod
            def size():
                return (1, 16, 2, 2, 2)

        class TorchStub:
            @staticmethod
            def zeros(size, dtype, layout, device):
                return ("zero", size, dtype, layout, device)

        class SamplingStub:
            noise = None

            @staticmethod
            def fix_empty_latent_channels(_model, samples, *_ratios):
                return samples

            @staticmethod
            def prepare_noise(*_args):
                raise AssertionError("禁用噪声时不应生成随机噪声")

            def sample(self, _model, noise, *args, **_kwargs):
                self.noise = noise
                return args[6]

        sampling = SamplingStub()
        worker = object.__new__(ComfyWorker)
        worker.comfy_sample = sampling
        worker.torch = TorchStub()
        worker.model = object()
        command = {
            "seed": 1,
            "steps": 20,
            "cfg": 3.5,
            "sampler": "euler",
            "scheduler": "simple",
        }

        worker._sample(
            command,
            {"samples": Samples()},
            [],
            [],
            lambda *_args: None,
            disable_noise=True,
        )

        self.assertEqual(sampling.noise[0], "zero")

    def test_latent_upscale_inherits_sampling_values_and_reports_actual_steps(self):
        resolved = resolve_upscale_settings(
            {
                "cfg": 1.5,
                "sampler": "euler",
                "scheduler": "simple",
                "seed": 42,
                "upscale": {
                    "enabled": True,
                    "method": "latent_hires",
                    "scale": 2,
                    "interpolation": "bislerp",
                    "steps": 9,
                    "start_step": 4,
                },
            }
        )

        self.assertEqual(resolved["cfg"], 1.5)
        self.assertEqual(resolved["sampler"], "euler")
        self.assertEqual(resolved["scheduler"], "simple")
        self.assertEqual(resolved["seed"], 42)
        self.assertTrue(resolved["seed_inherited"])
        self.assertEqual(resolved["steps"] - resolved["start_step"], 5)

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

    def test_sdxl_checkpoint_command_rejects_external_components(self):
        command = {
            "mode": "sdxl",
            "model_loader": "checkpoint",
            "vae_path": "vae.safetensors",
            "text_encoder_path": None,
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg": 7,
            "sampler": "euler",
            "scheduler": "simple",
        }

        with self.assertRaisesRegex(ValueError, "checkpoint loader"):
            ComfyWorker._validate(command)

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
