import unittest
from dataclasses import replace
from pathlib import Path

from diffusion_workbench_core.domain import GenerationSettings, Mode, ResourceItem
from test_api_jobs import ApiTestCase


class ImageTests(ApiTestCase):
    def _completed_job(self, output_path: Path):
        settings = GenerationSettings(
            mode=Mode.ZIT,
            model=ResourceItem(1, self.core.root / "zit.safetensors"),
            vae=ResourceItem(1, self.core.root / "zit-vae.safetensors"),
            text_encoder=self.core.root / "zit-te.safetensors",
            clip_type="stable_diffusion",
            prompt="p",
        )
        job = self.core.make_job(settings, status="completed")
        job = replace(job, output_path=output_path)
        self.core.jobs[job.id] = job
        return job

    def test_completed_image_is_served_as_png(self):
        output_dir = self.core.config.output_dir / "2026-07-27"
        output_dir.mkdir(parents=True)
        png = output_dir / "zit-00001.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        job = self._completed_job(png)

        response = self.client.get(f"/api/v1/images/{job.id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content, b"\x89PNG\r\n\x1a\nfake")

    def test_unfinished_job_returns_404(self):
        job = self._completed_job(self.core.config.output_dir / "x.png")
        self.core.jobs[job.id] = replace(job, status="queued")
        response = self.client.get(f"/api/v1/images/{job.id}")
        self.assertEqual(response.status_code, 404)

    def test_path_outside_output_dir_is_rejected(self):
        outside = self.core.root / "stolen.png"
        outside.write_bytes(b"secret")
        job = self._completed_job(outside)

        response = self.client.get(f"/api/v1/images/{job.id}")
        self.assertEqual(response.status_code, 500)
        self.assertNotEqual(response.content, b"secret")

    def test_missing_file_returns_410(self):
        job = self._completed_job(self.core.config.output_dir / "gone.png")
        response = self.client.get(f"/api/v1/images/{job.id}")
        self.assertEqual(response.status_code, 410)

    def test_unknown_job_returns_404(self):
        response = self.client.get("/api/v1/images/nope")
        self.assertEqual(response.status_code, 404)


class ImageMetadataTests(ApiTestCase):
    def _metadata_command(self):
        return {
            "job_id": "job-x",
            "batch_id": None,
            "mode": "zit",
            "prompt": "一只猫",
            "negative_prompt": "模糊, 水印",
            "width": 576,
            "height": 576,
            "steps": 8,
            "seed": 42,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "model_path": str(self.core.root / "models" / "m.safetensors"),
            "vae_path": str(self.core.root / "vae" / "v.safetensors"),
            "text_encoder_path": str(self.core.root / "te.safetensors"),
            "clip_type": "stable_diffusion",
        }

    def _png_with_metadata(self, name="zit-00009.png", metadata=None):
        from PIL import Image

        from diffusion_workbench_core.png_metadata import (
            build_generation_metadata,
            create_png_info,
        )

        if metadata is None:
            metadata = build_generation_metadata(
                self._metadata_command(), {"comfyui": "0.1"}
            )
        output_dir = self.core.config.output_dir / "2026-07-27"
        output_dir.mkdir(parents=True, exist_ok=True)
        png = output_dir / name
        Image.new("RGB", (8, 8)).save(png, pnginfo=create_png_info(metadata))
        return png

    def _completed_job(self, output_path: Path):
        settings = GenerationSettings(
            mode=Mode.ZIT,
            model=ResourceItem(1, self.core.root / "zit.safetensors"),
            vae=ResourceItem(1, self.core.root / "zit-vae.safetensors"),
            text_encoder=self.core.root / "zit-te.safetensors",
            clip_type="stable_diffusion",
            prompt="p",
        )
        job = self.core.make_job(settings, status="completed")
        job = replace(job, output_path=output_path)
        self.core.jobs[job.id] = job
        return job

    def test_metadata_comes_from_png_and_is_sanitized(self):
        job = self._completed_job(self._png_with_metadata())

        response = self.client.get(f"/api/v1/images/{job.id}/metadata")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["prompt"], "一只猫")
        self.assertEqual(body["model_name"], "m.safetensors")
        self.assertEqual(body["vae_name"], "v.safetensors")
        self.assertEqual(body["steps"], 8)
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["sampler"], "euler")
        self.assertEqual(body["scheduler"], "simple")
        self.assertEqual(body["negative_prompt"], "模糊, 水印")
        self.assertEqual(body["negative_conditioning"], "positive_reused")
        # 不泄露绝对路径和 SHA-256
        self.assertNotIn(str(self.core.root), str(body))
        self.assertNotIn("sha256", str(body))

    def test_metadata_exposes_loras_sanitized(self):
        from diffusion_workbench_core.png_metadata import build_generation_metadata

        command = self._metadata_command()
        command["loras"] = [
            {
                "path": str(self.core.root / "loras" / "style-a.safetensors"),
                "strength": 0.8,
            },
            {
                "path": str(self.core.root / "loras" / "style-b.safetensors"),
                "strength": 1.0,
            },
        ]
        metadata = build_generation_metadata(command, {"comfyui": "0.1"})
        job = self._completed_job(self._png_with_metadata("zit-00011.png", metadata))

        response = self.client.get(f"/api/v1/images/{job.id}/metadata")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["loras"],
            [
                {"name": "style-a.safetensors", "strength": 0.8},
                {"name": "style-b.safetensors", "strength": 1.0},
            ],
        )
        # 不泄露 LoRA 的服务器绝对路径
        self.assertNotIn(str(self.core.root), str(body))

    def test_v1_metadata_returns_compat_defaults(self):
        from diffusion_workbench_core.png_metadata import build_generation_metadata

        metadata = build_generation_metadata(
            self._metadata_command(), {"comfyui": "0.1"}
        )
        metadata["schema_version"] = 1
        for key in ("negative_prompt", "negative_conditioning"):
            metadata["parameters"].pop(key, None)
        job = self._completed_job(self._png_with_metadata("zit-00010.png", metadata))

        response = self.client.get(f"/api/v1/images/{job.id}/metadata")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["prompt"], "一只猫")
        self.assertEqual(body["negative_prompt"], "")
        self.assertIsNone(body["negative_conditioning"])

    def test_png_without_metadata_returns_404(self):
        from PIL import Image

        output_dir = self.core.config.output_dir / "2026-07-27"
        output_dir.mkdir(parents=True, exist_ok=True)
        png = output_dir / "plain.png"
        Image.new("RGB", (8, 8)).save(png)
        job = self._completed_job(png)

        response = self.client.get(f"/api/v1/images/{job.id}/metadata")
        self.assertEqual(response.status_code, 404)

    def test_metadata_requires_completed_job(self):
        job = self._completed_job(self.core.config.output_dir / "x.png")
        self.core.jobs[job.id] = replace(job, status="queued")
        response = self.client.get(f"/api/v1/images/{job.id}/metadata")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
