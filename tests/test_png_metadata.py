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


if __name__ == "__main__":
    unittest.main()
