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


class FakeModelManagement:
    """generate() 按单回收所需的最小 model_management 替身(可记录调用顺序)。"""

    def __init__(self, calls=None):
        self.calls = calls

    def cleanup_models_gc(self):
        if self.calls is not None:
            self.calls.append("cleanup_models_gc")

    def cleanup_models(self):
        if self.calls is not None:
            self.calls.append("cleanup_models")

    def soft_empty_cache(self, force=False):
        if self.calls is not None:
            self.calls.append("soft_empty_cache")

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
            "video_model": "minimax-h3-fl2va",
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

    def test_h3_ref2va_requires_matching_model_and_reference(self):
        command = {
            "video_model": "minimax-h3-ref2va",
            "clip_type": "minimax",
            "model_path": "minimax_h3_ref2va_int4.safetensors",
            "reference_image_paths": ["reference.png"],
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
        command["reference_image_paths"] = []
        with self.assertRaisesRegex(ValueError, "Ref2VA"):
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

    def _rebalance_command(self, **overrides):
        command = {
            "mode": "krea2-rebalance",
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "cfg": 1,
            "sampler": "euler",
            "scheduler": "simple",
            "reference_image_paths": ["ref.png"],
        }
        command.update(overrides)
        return command

    def test_rebalance_command_with_reference_images_is_valid(self):
        ComfyWorker._validate(self._rebalance_command())

    def test_rebalance_command_requires_reference_images(self):
        with self.assertRaisesRegex(ValueError, "参考图"):
            ComfyWorker._validate(self._rebalance_command(reference_image_paths=[]))

    def test_rebalance_command_rejects_too_many_reference_images(self):
        with self.assertRaisesRegex(ValueError, "参考图"):
            ComfyWorker._validate(
                self._rebalance_command(reference_image_paths=["a", "b", "c", "d", "e"])
            )

    def test_rebalance_command_validates_token_tiers(self):
        # 数量不一致
        with self.assertRaisesRegex(ValueError, "数量"):
            ComfyWorker._validate(
                self._rebalance_command(
                    reference_image_paths=["a", "b"],
                    reference_image_tokens=["normal"],
                )
            )
        # 非法档位
        with self.assertRaisesRegex(ValueError, "档位"):
            ComfyWorker._validate(
                self._rebalance_command(reference_image_tokens=["ultra"])
            )
        # 合法档位全部通过
        ComfyWorker._validate(
            self._rebalance_command(
                reference_image_paths=["a", "b", "c", "d"],
                reference_image_tokens=["low", "normal", "high", "max"],
            )
        )

    def test_non_rebalance_command_rejects_reference_images(self):
        with self.assertRaisesRegex(ValueError, "krea2-rebalance"):
            ComfyWorker._validate(
                self._rebalance_command(mode="krea2")
            )

    def _krea2_lora_command(self, **overrides):
        command = {
            "mode": "krea2",
            "width": 576,
            "height": 576,
            "steps": 8,
            "cfg": 1,
            "sampler": "euler",
            "scheduler": "simple",
            "loras": [{"path": "loras/style-a.safetensors", "strength": 0.8}],
        }
        command.update(overrides)
        return command

    def test_krea2_command_with_loras_is_valid(self):
        ComfyWorker._validate(self._krea2_lora_command())
        ComfyWorker._validate(
            self._krea2_lora_command(
                loras=[{"path": f"loras/style-{i}.safetensors"} for i in range(3)]
            )
        )

    def test_zit_command_with_loras_is_valid(self):
        ComfyWorker._validate(self._krea2_lora_command(mode="zit"))

    def test_command_rejects_loras_in_other_modes(self):
        with self.assertRaisesRegex(ValueError, "krea2 / zit"):
            ComfyWorker._validate(self._krea2_lora_command(mode="zib"))

    def test_krea2_command_rejects_more_than_three_loras(self):
        with self.assertRaisesRegex(ValueError, "最多支持 3 个 LoRA"):
            ComfyWorker._validate(
                self._krea2_lora_command(
                    loras=[{"path": f"loras/style-{i}.safetensors"} for i in range(4)]
                )
            )

    def test_krea2_command_validates_lora_shape_and_strength(self):
        with self.assertRaisesRegex(ValueError, "path"):
            ComfyWorker._validate(self._krea2_lora_command(loras=[{"strength": 1.0}]))
        with self.assertRaisesRegex(ValueError, "strength"):
            ComfyWorker._validate(
                self._krea2_lora_command(
                    loras=[{"path": "loras/a.safetensors", "strength": 2.5}]
                )
            )
        with self.assertRaisesRegex(ValueError, "strength"):
            ComfyWorker._validate(
                self._krea2_lora_command(
                    loras=[{"path": "loras/a.safetensors", "strength": float("nan")}]
                )
            )

    def _edit_command(self, **overrides):
        command = {
            "mode": "edit-krea2",
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "cfg": 1,
            "sampler": "euler",
            "scheduler": "simple",
            "input_image_path": "scene.png",
            "edit_lora_path": "edit.safetensors",
            "grounding_px": 768,
            "ref_boost": 1.0,
        }
        command.update(overrides)
        return command

    def test_edit_command_with_secondary_image_is_valid(self):
        # 双图编辑:secondary_input_image_path 可选,存在时合法
        ComfyWorker._validate(
            self._edit_command(secondary_input_image_path="subject.png")
        )
        ComfyWorker._validate(self._edit_command())

    def test_non_edit_command_rejects_secondary_input_image(self):
        with self.assertRaisesRegex(ValueError, "第二输入图片"):
            ComfyWorker._validate(
                self._edit_command(
                    mode="krea2",
                    input_image_path=None,
                    edit_lora_path=None,
                    secondary_input_image_path="s.png",
                )
            )

    def test_edit_command_rejects_latent_hires_upscale(self):
        # latent_hires 二段采样与 in-context patch 冲突,编辑模式拒绝
        command = self._edit_command(
            upscale={"enabled": True, "method": "latent_hires", "scale": 2},
            upscaled_output_path="out-upscale.png",
        )
        with self.assertRaisesRegex(ValueError, "放大"):
            ComfyWorker._validate(command)

    def test_edit_command_allows_postprocess_upscale(self):
        # resize / upscale_model 是纯后处理,编辑模式放行
        for method in ("resize", "upscale_model"):
            upscale = {"enabled": True, "method": method, "scale": 2, "interpolation": "lanczos"}
            if method == "upscale_model":
                upscale["model_path"] = "4x.pth"
            ComfyWorker._validate(
                self._edit_command(
                    upscale=upscale, upscaled_output_path="out-upscale.png"
                )
            )

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
        worker._comfy_execution_cleanup = lambda: None
        worker.model_management = FakeModelManagement()
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

    def test_generation_runs_comfy_execution_cleanup_on_success_and_failure(self):
        worker = object.__new__(ComfyWorker)
        worker.torch = FakeTorch()
        worker._validate = lambda _command: None
        worker.available_samplers = {"euler"}
        worker.available_schedulers = {"simple"}
        calls = []
        worker._comfy_execution_cleanup = lambda: calls.append("cleanup")
        worker.model_management = FakeModelManagement(calls)

        worker._generate = lambda _command: {"type": "result"}
        worker.generate({"sampler": "euler", "scheduler": "simple"})

        def failing_generate(_command):
            raise RuntimeError("boom")

        worker._generate = failing_generate
        with self.assertRaises(RuntimeError):
            worker.generate({"sampler": "euler", "scheduler": "simple"})

        # 每次 generate(成功或失败)都按序执行:全局收尾 -> 孤儿模型回收
        expected = ["cleanup", "cleanup_models_gc", "cleanup_models", "soft_empty_cache"]
        self.assertEqual(calls, expected + expected)

    def test_release_runs_comfy_execution_cleanup(self):
        calls = []

        class FakeModelManagement:
            def unload_all_models(self):
                calls.append("unload_all_models")

            def soft_empty_cache(self, force=False):
                calls.append("soft_empty_cache")

            def reset_cast_buffers(self):
                calls.append("reset_cast_buffers")

            def cleanup_models_gc(self):
                calls.append("cleanup_models_gc")

            def cleanup_models(self):
                calls.append("cleanup_models")

        class FakePrefetch:
            def cleanup_prefetch_queues(self):
                calls.append("cleanup_prefetch_queues")

        class FakeModelVbar:
            def vbars_reset_watermark_limits(self):
                calls.append("vbars_reset_watermark_limits")

        worker = object.__new__(ComfyWorker)
        worker.model_management = FakeModelManagement()
        worker.model_prefetch = FakePrefetch()
        worker.model_vbar = FakeModelVbar()
        worker._conditioning_cache = {}
        worker.release()

        for expected in (
            "reset_cast_buffers",
            "cleanup_prefetch_queues",
            "vbars_reset_watermark_limits",
        ):
            self.assertIn(expected, calls)
        # 全局收尾必须先于模型回收（cleanup_models*），否则引用仍挂住旧模型
        self.assertLess(calls.index("reset_cast_buffers"), calls.index("cleanup_models_gc"))

    def test_release_tolerates_missing_vbar_module(self):
        class FakeModelManagement:
            def unload_all_models(self):
                pass

            def soft_empty_cache(self, force=False):
                pass

            def reset_cast_buffers(self):
                pass

            def cleanup_models_gc(self):
                pass

            def cleanup_models(self):
                pass

        class FakePrefetch:
            def cleanup_prefetch_queues(self):
                pass

        worker = object.__new__(ComfyWorker)
        worker.model_management = FakeModelManagement()
        worker.model_prefetch = FakePrefetch()
        worker.model_vbar = None
        worker._conditioning_cache = {}
        worker.release()  # 不抛异常即可

    def test_ensure_vae_unloads_only_replaced_vae(self):
        class FakeVae:
            def __init__(self, patcher):
                self.patcher = patcher

        class FakeVaeLoader:
            def load_vae(self, name):
                return (FakeVae(f"new:{name}"),)

        calls = []

        class FakeMm:
            def unload_model_and_clones(self, patcher):
                calls.append(("unload_model_and_clones", patcher))

            def unload_all_models(self):
                calls.append(("unload_all_models",))

        worker = object.__new__(ComfyWorker)
        worker.vae = FakeVae("old-vae-patcher")
        worker.vae_path = Path("old.safetensors")
        worker.upscale_model = "upscale-keep"
        worker.model_management = FakeMm()
        worker.nodes = type("Nodes", (), {"VAELoader": FakeVaeLoader})
        worker._register_exact = lambda _category, path: path.name
        worker._conditioning_cache = {}

        self.assertTrue(worker._ensure_vae(Path("new.safetensors")))

        # 只定向卸载旧 VAE，不做全量 unload_all_models
        self.assertEqual(calls, [("unload_model_and_clones", "old-vae-patcher")])
        self.assertEqual(worker.vae.patcher, "new:new.safetensors")
        self.assertEqual(worker.vae_path, Path("new.safetensors"))
        # VAE 切换不再连坐丢弃 upscale_model 缓存
        self.assertEqual(worker.upscale_model, "upscale-keep")

        self.assertFalse(worker._ensure_vae(Path("new.safetensors")))
        self.assertEqual(len(calls), 1)

    def test_ensure_model_unloads_only_replaced_unet(self):
        calls = []

        class FakeMm:
            def unload_model_and_clones(self, patcher):
                calls.append(("unload_model_and_clones", patcher))

            def unload_all_models(self):
                calls.append(("unload_all_models",))

        worker = object.__new__(ComfyWorker)
        worker.model = "old-unet-patcher"
        worker.model_path = Path("old.safetensors")
        worker.model_management = FakeMm()
        worker._load_diffusion_model = lambda path: f"new:{path.name}"
        worker._conditioning_cache = {}

        self.assertTrue(worker._ensure_model(Path("new.safetensors")))

        self.assertEqual(calls, [("unload_model_and_clones", "old-unet-patcher")])
        self.assertEqual(worker.model, "new:new.safetensors")
        self.assertEqual(worker.model_path, Path("new.safetensors"))

    def test_ensure_clip_unloads_only_replaced_clip(self):
        class FakeClip:
            patcher = "old-clip-patcher"

        class FakeClipLoader:
            def load_clip(self, name, clip_type):
                return (FakeClip(),)

        calls = []

        class FakeMm:
            def unload_model_and_clones(self, patcher):
                calls.append(("unload_model_and_clones", patcher))

            def unload_all_models(self):
                calls.append(("unload_all_models",))

        worker = object.__new__(ComfyWorker)
        worker.clip = FakeClip()
        worker.clip_path = Path("old.safetensors")
        worker.clip_type = "wan"
        worker.model_management = FakeMm()
        worker.nodes = type("Nodes", (), {"CLIPLoader": FakeClipLoader})
        worker._register_exact = lambda _category, path: path.name
        worker._conditioning_cache = {}

        self.assertTrue(worker._ensure_clip_local(Path("new.safetensors"), "wan"))

        self.assertEqual(calls, [("unload_model_and_clones", "old-clip-patcher")])
        self.assertEqual(worker.clip_path, Path("new.safetensors"))


if __name__ == "__main__":
    unittest.main()
