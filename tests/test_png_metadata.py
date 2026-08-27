import tempfile
import unittest
from pathlib import Path

from PIL import Image

from diffusion_workbench_core.png_metadata import (
    PNG_METADATA_KEY,
    ResourceFingerprintCache,
    build_generation_metadata,
    create_png_info,
    read_generation_metadata,
)


class PngMetadataTests(unittest.TestCase):
    def test_checkpoint_metadata_records_embedded_resources_as_null(self):
        command = {
            "job_id": "sdxl-job",
            "mode": "sdxl",
            "model_loader": "checkpoint",
            "prompt": "portrait",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "seed": 42,
            "cfg": 7,
            "sampler": "euler",
            "scheduler": "simple",
            "model_path": Path("sdxl.safetensors"),
            "vae_path": None,
            "text_encoder_path": None,
            "clip_type": None,
        }

        metadata = build_generation_metadata(command, {})

        self.assertEqual(metadata["resources"]["model_loader"], "checkpoint")
        self.assertIsNone(metadata["resources"]["vae"])
        self.assertIsNone(metadata["resources"]["text_encoder"])

    def test_round_trips_all_generation_settings_as_unicode_itxt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "generated.png"
            command = {
                "job_id": "job-1",
                "batch_id": "batch-1",
                "workbench_version": "0.1.0",
                "mode": "krea2",
                "prompt": "一幅雨夜肖像",
                "negative_prompt": "模糊，水印",
                "width": 960,
                "height": 1280,
                "steps": 8,
                "seed": 42,
                "cfg": 1.0,
                "sampler": "euler",
                "scheduler": "simple",
                "model_path": root / "model.safetensors",
                "vae_path": root / "vae.safetensors",
                "text_encoder_path": root / "te.safetensors",
                "clip_type": "krea2",
            }
            for filename in ("model.safetensors", "vae.safetensors", "te.safetensors"):
                (root / filename).write_bytes(filename.encode("ascii"))
            fingerprints = ResourceFingerprintCache()
            resources = {
                "diffusion_model": fingerprints.describe(command["model_path"]),
                "vae": fingerprints.describe(command["vae_path"]),
                "text_encoder": fingerprints.describe(command["text_encoder_path"]),
            }
            metadata = build_generation_metadata(
                command,
                {"comfyui": "0.28.0", "pytorch": "2.8.0", "cuda": "12.8"},
                {"sampling_seconds": 6.25},
                resources,
            )
            Image.new("RGB", (8, 8)).save(
                output, format="PNG", pnginfo=create_png_info(metadata)
            )

            loaded = read_generation_metadata(output)

            self.assertEqual(loaded, metadata)
            self.assertEqual(loaded["parameters"]["prompt"], "一幅雨夜肖像")
            self.assertEqual(loaded["parameters"]["negative_prompt"], "模糊，水印")
            self.assertEqual(loaded["parameters"]["seed"], 42)
            self.assertEqual(loaded["schema_version"], 4)
            self.assertEqual(loaded["artifact"]["kind"], "original")
            self.assertFalse(loaded["parameters"]["upscale"]["enabled"])
            self.assertEqual(
                loaded["resources"]["diffusion_model"]["filename"],
                "model.safetensors",
            )
            self.assertEqual(len(loaded["resources"]["diffusion_model"]["sha256"]), 64)
            with Image.open(output) as image:
                self.assertIn(PNG_METADATA_KEY, image.info)

    def test_returns_none_for_an_unannotated_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "plain.png"
            Image.new("RGB", (8, 8)).save(output)

            self.assertIsNone(read_generation_metadata(output))

    def test_reads_legacy_v1_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "legacy.png"
            legacy = {"schema_version": 1, "parameters": {"prompt": "portrait"}}
            Image.new("RGB", (8, 8)).save(
                output, format="PNG", pnginfo=create_png_info(legacy)
            )

            self.assertEqual(read_generation_metadata(output), legacy)

    def test_resource_fingerprint_cache_invalidates_replaced_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource = Path(temp_dir) / "model.safetensors"
            resource.write_bytes(b"first")
            fingerprints = ResourceFingerprintCache()

            first = fingerprints.describe(resource)
            cached = fingerprints.describe(resource)
            resource.write_bytes(b"replacement")
            replaced = fingerprints.describe(resource)

            self.assertEqual(first, cached)
            self.assertNotEqual(first["sha256"], replaced["sha256"])
            self.assertEqual(replaced["size_bytes"], len(b"replacement"))

    def test_edit_krea2_metadata_records_edit_fields_and_lora(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            command = {
                "job_id": "edit-job",
                "batch_id": None,
                "workbench_version": "0.1.0",
                "mode": "edit-krea2",
                "prompt": "turn the cat into a tiger",
                "negative_prompt": "",
                "width": 576,
                "height": 576,
                "steps": 8,
                "seed": 7,
                "cfg": 1.0,
                "sampler": "euler",
                "scheduler": "simple",
                "model_path": root / "krea2.safetensors",
                "vae_path": root / "vae.safetensors",
                "text_encoder_path": root / "te.safetensors",
                "clip_type": "krea2",
                "input_image_path": str((root / "source.png").resolve()),
                "grounding_px": 512,
                "ref_boost": 2.0,
                "edit_lora_path": str((root / "edit_lora.safetensors").resolve()),
            }
            for filename in ("krea2.safetensors", "vae.safetensors", "te.safetensors", "edit_lora.safetensors"):
                (root / filename).write_bytes(filename.encode("ascii"))
            (root / "source.png").write_bytes(b"source")

            metadata = build_generation_metadata(command, {})

            self.assertEqual(metadata["parameters"]["mode"], "edit-krea2")
            self.assertEqual(metadata["parameters"]["grounding_px"], 512)
            self.assertEqual(metadata["parameters"]["ref_boost"], 2.0)
            # CFG 1 + 编辑模式 + 空 negative → positive_reused
            self.assertEqual(
                metadata["parameters"]["negative_conditioning"], "positive_reused"
            )
            self.assertEqual(
                metadata["resources"]["edit_lora"]["filename"], "edit_lora.safetensors"
            )

    def test_edit_krea2_metadata_marks_conditioning_zero_out_when_cfg_not_one(self):
        command = {
            "job_id": "edit-job",
            "mode": "edit-krea2",
            "prompt": "p",
            "negative_prompt": "",
            "width": 576,
            "height": 576,
            "steps": 8,
            "seed": 1,
            "cfg": 2.0,
            "sampler": "euler",
            "scheduler": "simple",
            "model_path": "m",
            "clip_type": "krea2",
        }

        metadata = build_generation_metadata(command, {})

        self.assertEqual(
            metadata["parameters"]["negative_conditioning"], "conditioning_zero_out"
        )

    def test_non_edit_metadata_omits_edit_fields(self):
        command = {
            "job_id": "krea2-job",
            "mode": "krea2",
            "prompt": "p",
            "negative_prompt": "",
            "width": 576,
            "height": 576,
            "steps": 8,
            "seed": 1,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "model_path": "m",
            "clip_type": "krea2",
            # 即使错误地携带 edit 字段也应被忽略
            "input_image_path": "/tmp/source.png",
            "grounding_px": 512,
            "ref_boost": 2.0,
            "edit_lora_path": "/tmp/edit_lora.safetensors",
        }

        metadata = build_generation_metadata(command, {})

        self.assertNotIn("grounding_px", metadata["parameters"])
        self.assertNotIn("ref_boost", metadata["parameters"])
        self.assertNotIn("edit_lora", metadata["resources"])

    def test_rebalance_metadata_records_reference_fields(self):
        command = {
            "job_id": "rebalance-job",
            "batch_id": None,
            "workbench_version": "0.1.0",
            "mode": "krea2-rebalance",
            "prompt": "参考构图画一只猫",
            "negative_prompt": "",
            "width": 576,
            "height": 576,
            "steps": 8,
            "seed": 7,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "model_path": "m",
            "clip_type": "krea2",
            "reference_image_paths": ["/tmp/ref-a.png", "/tmp/ref-b.png"],
            "reference_image_tokens": ["low", "max"],
        }

        metadata = build_generation_metadata(command, {})

        self.assertEqual(metadata["parameters"]["mode"], "krea2-rebalance")
        self.assertEqual(
            metadata["parameters"]["reference_images"], ["ref-a.png", "ref-b.png"]
        )
        self.assertEqual(
            metadata["parameters"]["reference_image_tokens"], ["low", "max"]
        )
        # CFG 1 时与普通模式一致:negative 复用 positive
        self.assertEqual(
            metadata["parameters"]["negative_conditioning"], "positive_reused"
        )

    def test_non_rebalance_metadata_omits_reference_fields(self):
        command = {
            "job_id": "krea2-job",
            "mode": "krea2",
            "prompt": "p",
            "negative_prompt": "",
            "width": 576,
            "height": 576,
            "steps": 8,
            "seed": 1,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "model_path": "m",
            "clip_type": "krea2",
            # 即使错误地携带 reference 字段也应被忽略
            "reference_image_paths": ["/tmp/ref.png"],
            "reference_image_tokens": ["normal"],
        }

        metadata = build_generation_metadata(command, {})

        self.assertNotIn("reference_images", metadata["parameters"])
        self.assertNotIn("reference_image_tokens", metadata["parameters"])


if __name__ == "__main__":
    unittest.main()
