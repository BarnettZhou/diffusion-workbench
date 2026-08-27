import unittest
from dataclasses import replace
from pathlib import Path

from diffusion_workbench_api.schemas import public_event
from diffusion_workbench_core.domain import (
    GenerationSettings,
    Mode,
    ResourceItem,
    UpscaleMethod,
    UpscaleSettings,
)
from test_api_jobs import ApiTestCase


class UpscaleOptionsTests(ApiTestCase):
    def test_upscale_options_endpoint(self):
        response = self.client.get("/api/v1/upscale-options")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["methods"], ["resize", "upscale_model", "latent_hires"])
        self.assertIn("lanczos", body["image_interpolations"])
        self.assertIn("bislerp", body["latent_interpolations"])
        self.assertNotIn("bislerp", body["image_interpolations"])
        self.assertEqual(len(body["samplers"]), 44)
        self.assertEqual(len(body["schedulers"]), 9)
        self.assertEqual(body["defaults"]["method"], "latent_hires")
        self.assertEqual(body["defaults"]["steps"], 9)
        self.assertEqual(body["defaults"]["start_step"], 4)

    def test_upscale_models_endpoint_hides_paths(self):
        response = self.client.get("/api/v1/upscale-models")

        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["index"], 1)
        self.assertEqual(models[0]["name"], "4x-UltraSharp.pth")
        self.assertNotIn(str(self.core.root), str(models))


class UpscaleSubmitTests(ApiTestCase):
    def _payload(self, upscale=None, **overrides):
        payload = {
            "mode": "krea2",
            "model_index": 1,
            "vae_index": 1,
            "text_encoder_index": 1,
            "prompt": "an adult studio portrait",
        }
        if upscale is not None:
            payload["upscale"] = upscale
        payload.update(overrides)
        return payload

    def test_omitted_upscale_defaults_to_disabled(self):
        response = self.client.post("/api/v1/jobs", json=self._payload())

        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertFalse(job["upscale"]["enabled"])
        self.assertIsNone(job["upscaled_output_name"])
        self.assertIsNone(job["upscaled_image_url"])

        settings, _ = self.core.submitted[0]
        self.assertFalse(settings.upscale.enabled)

    def test_latent_hires_maps_to_upscale_settings(self):
        response = self.client.post(
            "/api/v1/jobs",
            json=self._payload(
                {
                    "enabled": True,
                    "method": "latent_hires",
                    "scale": 2.0,
                    "interpolation": "bislerp",
                    "steps": 9,
                    "start_step": 4,
                    "cfg": None,
                    "sampler": None,
                    "scheduler": None,
                    "seed": None,
                }
            ),
        )

        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertTrue(job["upscale"]["enabled"])
        self.assertEqual(job["upscale"]["method"], "latent_hires")
        self.assertEqual(job["upscale"]["steps"], 9)
        self.assertEqual(job["upscale"]["start_step"], 4)
        self.assertTrue(job["upscaled_output_name"].endswith("-upscale.png"))

        settings, _ = self.core.submitted[0]
        self.assertTrue(settings.upscale.enabled)
        self.assertEqual(settings.upscale.method, UpscaleMethod.LATENT_HIRES)
        self.assertEqual(settings.upscale.executed_steps, 5)
        self.assertIsNone(settings.upscale.cfg)
        self.assertIsNone(settings.upscale.sampler)

    def test_resize_method_accepted(self):
        response = self.client.post(
            "/api/v1/jobs",
            json=self._payload(
                {"enabled": True, "method": "resize", "scale": 2.0, "interpolation": "lanczos"}
            ),
        )

        self.assertEqual(response.status_code, 202)
        settings, _ = self.core.submitted[0]
        self.assertEqual(settings.upscale.method, UpscaleMethod.RESIZE)
        self.assertEqual(settings.upscale.interpolation, "lanczos")

    def test_upscale_model_resolves_by_index(self):
        response = self.client.post(
            "/api/v1/jobs",
            json=self._payload(
                {
                    "enabled": True,
                    "method": "upscale_model",
                    "model_index": 1,
                    "scale": 2.0,
                    "interpolation": "lanczos",
                    "tile": 512,
                    "overlap": 32,
                }
            ),
        )

        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["upscale"]["model_name"], "4x-UltraSharp.pth")

        settings, _ = self.core.submitted[0]
        self.assertEqual(settings.upscale.method, UpscaleMethod.UPSCALE_MODEL)
        self.assertEqual(settings.upscale.model.path.name, "4x-UltraSharp.pth")

    def test_unknown_upscale_model_index_returns_404(self):
        response = self.client.post(
            "/api/v1/jobs",
            json=self._payload(
                {"enabled": True, "method": "upscale_model", "model_index": 99}
            ),
        )
        self.assertEqual(response.status_code, 404)

    def test_upscale_model_without_index_returns_422(self):
        response = self.client.post(
            "/api/v1/jobs",
            json=self._payload({"enabled": True, "method": "upscale_model"}),
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_method_combination_returns_422(self):
        # latent_hires 不支持 lanczos 插值
        response = self.client.post(
            "/api/v1/jobs",
            json=self._payload(
                {"enabled": True, "method": "latent_hires", "interpolation": "lanczos"}
            ),
        )
        self.assertEqual(response.status_code, 422)

    def test_start_step_must_be_less_than_steps(self):
        response = self.client.post(
            "/api/v1/jobs",
            json=self._payload(
                {"enabled": True, "method": "latent_hires", "steps": 9, "start_step": 9}
            ),
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_scale_returns_422(self):
        response = self.client.post(
            "/api/v1/jobs",
            json=self._payload({"enabled": True, "method": "resize", "scale": 5.0}),
        )
        self.assertEqual(response.status_code, 422)


class UpscaledImageTests(ApiTestCase):
    def _job_with_upscale(self, status="completed", write_files=()):
        output_dir = self.core.config.output_dir / "2026-07-27"
        output_dir.mkdir(parents=True, exist_ok=True)
        settings = GenerationSettings(
            mode=Mode.ZIT,
            model=ResourceItem(1, self.core.root / "zit.safetensors"),
            vae=ResourceItem(1, self.core.root / "zit-vae.safetensors"),
            text_encoder=self.core.root / "zit-te.safetensors",
            clip_type="stable_diffusion",
            prompt="p",
            upscale=UpscaleSettings(
                enabled=True, method=UpscaleMethod.RESIZE, interpolation="lanczos"
            ),
        )
        job = self.core.make_job(settings, status=status)
        self.core.jobs[job.id] = job
        paths = {"original": job.output_path, "upscaled": job.upscaled_output_path}
        for kind in write_files:
            paths[kind].write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return job

    def test_upscaled_image_is_served(self):
        job = self._job_with_upscale(write_files=("original", "upscaled"))

        response = self.client.get(f"/api/v1/images/{job.id}/upscaled")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

        detail = self.client.get(f"/api/v1/jobs/{job.id}").json()
        self.assertEqual(
            detail["upscaled_image_url"], f"/api/v1/images/{job.id}/upscaled"
        )
        self.assertTrue(detail["upscaled_output_name"].endswith("-upscale.png"))

    def test_upscale_not_enabled_returns_404(self):
        settings = GenerationSettings(
            mode=Mode.ZIT,
            model=ResourceItem(1, self.core.root / "zit.safetensors"),
            vae=ResourceItem(1, self.core.root / "zit-vae.safetensors"),
            text_encoder=self.core.root / "zit-te.safetensors",
            clip_type="stable_diffusion",
            prompt="p",
        )
        job = self.core.make_job(settings, status="completed")
        self.core.jobs[job.id] = job

        response = self.client.get(f"/api/v1/images/{job.id}/upscaled")
        self.assertEqual(response.status_code, 404)

    def test_missing_upscaled_file_returns_410(self):
        job = self._job_with_upscale()
        response = self.client.get(f"/api/v1/images/{job.id}/upscaled")
        self.assertEqual(response.status_code, 410)

    def test_running_job_upscaled_returns_404(self):
        job = self._job_with_upscale(status="running")
        response = self.client.get(f"/api/v1/images/{job.id}/upscaled")
        self.assertEqual(response.status_code, 404)

    def test_original_image_available_when_upscale_failed(self):
        # 放大阶段失败:任务 failed,但已保存的原图仍可下载
        job = self._job_with_upscale(status="failed", write_files=("original",))

        response = self.client.get(f"/api/v1/images/{job.id}")
        self.assertEqual(response.status_code, 200)

        detail = self.client.get(f"/api/v1/jobs/{job.id}").json()
        self.assertEqual(detail["image_url"], f"/api/v1/images/{job.id}")
        self.assertIsNone(detail["upscaled_image_url"])

    def test_upscaled_metadata_exposes_artifact_and_sanitized_upscale(self):
        from PIL import Image

        from diffusion_workbench_core.png_metadata import (
            build_generation_metadata,
            create_png_info,
        )

        job = self._job_with_upscale()
        command = {
            "job_id": job.id,
            "batch_id": None,
            "mode": "zit",
            "prompt": "一只猫",
            "negative_prompt": "",
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
            "upscale": job.upscale.to_dict(),
        }
        metadata = build_generation_metadata(
            command,
            {"comfyui": "0.1"},
            artifact_kind="upscaled",
            artifact_size=(1152, 1152),
        )
        Image.new("RGB", (8, 8)).save(
            job.upscaled_output_path, pnginfo=create_png_info(metadata)
        )

        response = self.client.get(f"/api/v1/images/{job.id}/upscaled/metadata")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["artifact"], {"kind": "upscaled", "width": 1152, "height": 1152})
        self.assertEqual(body["width"], 576)
        self.assertTrue(body["upscale"]["enabled"])
        self.assertEqual(body["upscale"]["method"], "resize")
        self.assertEqual(body["upscale"]["interpolation"], "lanczos")
        # 不泄露绝对路径和 SHA-256
        self.assertNotIn(str(self.core.root), str(body))
        self.assertNotIn("model_path", str(body))


class UpscaleEventTests(unittest.TestCase):
    def test_job_finished_event_includes_upscaled_url_without_path(self):
        event = {
            "type": "job_finished",
            "job_id": "job-1",
            "status": "completed",
            "output_path": "D:\\output\\zit-00001.png",
            "upscaled_output_path": "D:\\output\\zit-00001-upscale.png",
            "steps": 8,
        }
        public = public_event(event, 7)

        self.assertEqual(public["image_url"], "/api/v1/images/job-1")
        self.assertEqual(public["upscaled_image_url"], "/api/v1/images/job-1/upscaled")
        self.assertNotIn("upscaled_output_path", public)
        self.assertNotIn("D:\\", str(public))

    def test_failed_event_has_no_upscaled_url(self):
        event = {
            "type": "job_finished",
            "job_id": "job-1",
            "status": "failed",
            "output_path": "D:\\output\\zit-00001.png",
            "upscaled_output_path": "D:\\output\\zit-00001-upscale.png",
            "steps": 8,
        }
        public = public_event(event, 7)
        self.assertNotIn("upscaled_image_url", public)


if __name__ == "__main__":
    unittest.main()
