import io
import unittest

from PIL import Image

from diffusion_workbench_core.domain import Mode, UpscaleMethod
from test_api_jobs import ApiTestCase


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 64, 32)).save(buffer, format="PNG")
    return buffer.getvalue()


class EditInfoTests(ApiTestCase):
    def test_info_reports_enabled_and_defaults(self):
        response = self.client.get("/api/v1/edit/info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "enabled": True,
                "defaults": {"grounding_px": 768, "ref_boost": 1.0},
                "rebalance": {
                    "enabled": True,
                    "defaults": {
                        "token_tier": "normal",
                        "token_tiers": ["low", "normal", "high", "max"],
                        "max_reference_images": 4,
                    },
                },
            },
        )

    def test_info_reports_disabled_when_edit_lora_missing(self):
        # 把 krea2 的 edit_lora 置空,模拟服务端未配置编辑 LoRA 的情况
        from dataclasses import replace
        resources = self.core.config.resources[Mode.KREA2]
        new_resources = dict(self.core.config.resources)
        new_resources[Mode.KREA2] = replace(resources, edit_lora=None)
        object.__setattr__(self.core.config, "resources", new_resources)
        response = self.client.get("/api/v1/edit/info")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["enabled"])
        self.assertEqual(body["defaults"], {"grounding_px": 768, "ref_boost": 1.0})


class EditInputImageUploadTests(ApiTestCase):
    def test_upload_returns_id_and_url(self):
        response = self.client.post(
            "/api/v1/edit/input-images",
            content=_png_bytes(),
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        image_id = body["id"]
        self.assertRegex(image_id, r"^[0-9a-f]{32}\.png$")
        self.assertEqual(body["url"], f"/api/v1/edit/input-images/{image_id}")
        # 落盘文件可按 url 读取,内容与上传一致
        served = self.client.get(body["url"])
        self.assertEqual(served.status_code, 200)
        self.assertEqual(served.headers["content-type"], "image/png")
        self.assertEqual(served.content, _png_bytes())

    def test_upload_rejects_unsupported_media_type(self):
        response = self.client.post(
            "/api/v1/edit/input-images",
            content=_png_bytes(),
            headers={"Content-Type": "image/gif"},
        )
        self.assertEqual(response.status_code, 422)

    def test_upload_rejects_empty_body(self):
        response = self.client.post(
            "/api/v1/edit/input-images",
            content=b"",
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status_code, 422)

    def test_upload_rejects_invalid_image_bytes(self):
        response = self.client.post(
            "/api/v1/edit/input-images",
            content=b"not an image",
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status_code, 422)

    def test_get_unknown_input_image_returns_404(self):
        self.assertEqual(
            self.client.get(f"/api/v1/edit/input-images/{'0' * 32}.png").status_code,
            404,
        )
        # 非法 id 含目录穿越尝试同样 404
        self.assertEqual(
            self.client.get("/api/v1/edit/input-images/..%2F..%2Fjobs.sqlite3").status_code,
            404,
        )


class EditInputImageFromJobTests(ApiTestCase):
    def _submit_job(self) -> str:
        response = self.client.post(
            "/api/v1/jobs",
            json={
                "mode": "krea2",
                "model_index": 1,
                "vae_index": 1,
                "text_encoder_index": 1,
                "prompt": "an adult studio portrait",
            },
        )
        self.assertEqual(response.status_code, 202)
        return response.json()["jobs"][0]["id"]

    def _write_output(self, job_id: str) -> None:
        job = self.core.jobs[job_id]
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        job.output_path.write_bytes(_png_bytes())

    def test_import_from_job_roundtrip(self):
        job_id = self._submit_job()
        self._write_output(job_id)
        response = self.client.post(
            "/api/v1/edit/input-images/from-job", json={"job_id": job_id}
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertRegex(body["id"], r"^[0-9a-f]{32}\.png$")
        # 落盘文件可按 url 读取,内容与任务输出一致
        served = self.client.get(body["url"])
        self.assertEqual(served.status_code, 200)
        self.assertEqual(served.content, _png_bytes())

    def test_import_unknown_job_returns_404(self):
        response = self.client.post(
            "/api/v1/edit/input-images/from-job", json={"job_id": "nope"}
        )
        self.assertEqual(response.status_code, 404)

    def test_import_job_without_output_returns_404(self):
        # 任务存在但输出文件未落盘(未完成/已丢失)
        job_id = self._submit_job()
        response = self.client.post(
            "/api/v1/edit/input-images/from-job", json={"job_id": job_id}
        )
        self.assertEqual(response.status_code, 404)


class EditInputImageFromAlbumTests(ApiTestCase):
    def _write_album_png(self, relpath="2026-08-09/zit-00001.png"):
        path = self.core.config.output_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png_bytes())
        return relpath

    def test_import_from_album_roundtrip(self):
        relpath = self._write_album_png()
        response = self.client.post(
            "/api/v1/edit/input-images/from-album",
            json={"id": relpath, "dir": "output"},
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        image_id = body["id"]
        self.assertRegex(image_id, r"^[0-9a-f]{32}\.png$")
        served = self.client.get(body["url"])
        self.assertEqual(served.status_code, 200)

    def test_import_missing_file_returns_404(self):
        response = self.client.post(
            "/api/v1/edit/input-images/from-album",
            json={"id": "2026-08-09/nope.png", "dir": "output"},
        )
        self.assertEqual(response.status_code, 404)

    def test_import_rejects_non_image(self):
        path = self.core.config.output_dir / "2026-08-09" / "krea2-00001.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not an image")
        response = self.client.post(
            "/api/v1/edit/input-images/from-album",
            json={"id": "2026-08-09/krea2-00001.txt", "dir": "output"},
        )
        self.assertEqual(response.status_code, 422)

    def test_import_rejects_path_traversal(self):
        response = self.client.post(
            "/api/v1/edit/input-images/from-album",
            json={"id": "../jobs.sqlite3", "dir": "output"},
        )
        # 相册 resolve 对越界路径返回 400
        self.assertEqual(response.status_code, 400)


class EditJobsTests(ApiTestCase):
    def _upload_image(self):
        response = self.client.post(
            "/api/v1/edit/input-images",
            content=_png_bytes(),
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def _payload(self, **overrides):
        payload = {
            "model_index": 1,
            "vae_index": 1,
            "text_encoder_index": 1,
            "prompt": "把人物改成夜晚氛围",
            "input_image_id": self._upload_image(),
        }
        payload.update(overrides)
        return payload

    def test_submit_maps_request_to_edit_settings(self):
        response = self.client.post("/api/v1/edit/jobs", json=self._payload(count=2))
        self.assertEqual(response.status_code, 202)
        body = response.json()["jobs"]
        self.assertEqual(len(body), 2)
        job = body[0]
        self.assertEqual(job["mode"], "edit-krea2")
        self.assertEqual(job["status"], "queued")
        self.assertIsNotNone(job["batch_id"])
        self.assertEqual(job["batch_id"], body[1]["batch_id"])
        self.assertEqual(job["model_name"], "krea.safetensors")
        self.assertEqual(job["vae_name"], "krea-vae.safetensors")
        # input_image_url 指向受控目录的 id,grounding_px/ref_boost 透传
        image_id = job["input_image_url"].rsplit("/", 1)[-1]
        self.assertRegex(image_id, r"^[0-9a-f]{32}\.png$")
        self.assertEqual(job["grounding_px"], 768)
        self.assertEqual(job["ref_boost"], 1.0)

        settings, count = self.core.submitted[-1]
        self.assertEqual(count, 2)
        self.assertEqual(settings.mode, Mode.KREA2_EDIT)
        self.assertEqual(settings.model.path.name, "krea.safetensors")
        self.assertEqual(settings.vae.path.name, "krea-vae.safetensors")
        self.assertEqual(
            settings.text_encoder,
            self.core.root / "krea-te.safetensors",
        )
        self.assertEqual(settings.clip_type, "krea2")
        # input_image 是服务端解析后的受控路径(不是客户端提交的内容)
        self.assertIsNotNone(settings.input_image)
        self.assertTrue(str(settings.input_image).endswith(image_id))
        self.assertEqual(settings.grounding_px, 768)
        self.assertEqual(settings.ref_boost, 1.0)

    def test_dual_submit_maps_secondary_image(self):
        # 双图编辑:secondary_input_image_id 透传到 settings 并回显受控 URL
        secondary_id = self._upload_image()
        response = self.client.post(
            "/api/v1/edit/jobs",
            json=self._payload(secondary_input_image_id=secondary_id),
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["mode"], "edit-krea2")
        self.assertTrue(job["secondary_input_image_url"].endswith(secondary_id))

        settings, _ = self.core.submitted[-1]
        self.assertEqual(settings.mode, Mode.KREA2_EDIT)
        self.assertIsNotNone(settings.secondary_input_image)
        self.assertTrue(str(settings.secondary_input_image).endswith(secondary_id))

    def test_submit_without_secondary_image_keeps_field_null(self):
        response = self.client.post("/api/v1/edit/jobs", json=self._payload())
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertIsNone(job["secondary_input_image_url"])
        settings, _ = self.core.submitted[-1]
        self.assertIsNone(settings.secondary_input_image)

    def test_unknown_secondary_input_image_id_returns_404(self):
        response = self.client.post(
            "/api/v1/edit/jobs",
            json=self._payload(secondary_input_image_id="f" * 32 + ".png"),
        )
        self.assertEqual(response.status_code, 404)

    def test_edit_submit_with_upscale_model(self):
        # 编辑模式支持纯后处理放大:upscale_model 透传到 settings 并回显
        response = self.client.post(
            "/api/v1/edit/jobs",
            json=self._payload(
                upscale={
                    "enabled": True,
                    "method": "upscale_model",
                    "scale": 2,
                    "interpolation": "lanczos",
                    "model_index": 1,
                }
            ),
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertTrue(job["upscale"]["enabled"])
        self.assertEqual(job["upscale"]["method"], "upscale_model")
        self.assertEqual(job["upscale"]["model_name"], "4x-UltraSharp.pth")

        settings, _ = self.core.submitted[-1]
        self.assertTrue(settings.upscale.enabled)
        self.assertEqual(settings.upscale.method, UpscaleMethod.UPSCALE_MODEL)

    def test_edit_submit_rejects_latent_hires_upscale(self):
        # latent_hires 与 in-context patch 冲突,domain 校验拒绝
        response = self.client.post(
            "/api/v1/edit/jobs",
            json=self._payload(
                upscale={"enabled": True, "method": "latent_hires", "scale": 2}
            ),
        )
        self.assertEqual(response.status_code, 422)

    def test_single_job_has_no_batch_id(self):
        response = self.client.post("/api/v1/edit/jobs", json=self._payload())
        self.assertEqual(response.status_code, 202)
        self.assertIsNone(response.json()["jobs"][0]["batch_id"])

    def test_custom_grounding_and_ref_boost_propagate(self):
        response = self.client.post(
            "/api/v1/edit/jobs",
            json=self._payload(grounding_px=0, ref_boost=3.5),
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["grounding_px"], 0)
        self.assertEqual(job["ref_boost"], 3.5)
        settings, _ = self.core.submitted[-1]
        self.assertEqual(settings.grounding_px, 0)
        self.assertEqual(settings.ref_boost, 3.5)

    def test_non_edit_job_keeps_edit_fields_null(self):
        response = self.client.post(
            "/api/v1/jobs",
            json={
                "mode": "krea2",
                "model_index": 1,
                "vae_index": 1,
                "text_encoder_index": 1,
                "prompt": "p",
            },
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["mode"], "krea2")
        self.assertIsNone(job["input_image_url"])
        self.assertIsNone(job["grounding_px"])
        self.assertIsNone(job["ref_boost"])

    def test_get_edit_job_returns_new_fields(self):
        created = self.client.post("/api/v1/edit/jobs", json=self._payload()).json()["jobs"][0]
        response = self.client.get(f"/api/v1/jobs/{created['id']}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "edit-krea2")
        self.assertIsNotNone(body["input_image_url"])
        self.assertEqual(body["grounding_px"], 768)
        self.assertEqual(body["ref_boost"], 1.0)

    def test_missing_input_image_id_returns_422(self):
        response = self.client.post(
            "/api/v1/edit/jobs",
            json={
                "model_index": 1,
                "vae_index": 1,
                "text_encoder_index": 1,
                "prompt": "p",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_unknown_input_image_id_returns_404(self):
        response = self.client.post(
            "/api/v1/edit/jobs",
            json={
                "model_index": 1,
                "vae_index": 1,
                "text_encoder_index": 1,
                "prompt": "p",
                "input_image_id": "f" * 32 + ".png",
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_malformed_input_image_id_returns_404(self):
        response = self.client.post(
            "/api/v1/edit/jobs",
            json={
                "model_index": 1,
                "vae_index": 1,
                "text_encoder_index": 1,
                "prompt": "p",
                "input_image_id": "../../jobs.sqlite3",
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_grounding_px_out_of_range_returns_422(self):
        response = self.client.post(
            "/api/v1/edit/jobs", json=self._payload(grounding_px=4097)
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.post(
            "/api/v1/edit/jobs", json=self._payload(grounding_px=-1)
        )
        self.assertEqual(response.status_code, 422)

    def test_ref_boost_out_of_range_returns_422(self):
        response = self.client.post(
            "/api/v1/edit/jobs", json=self._payload(ref_boost=1001)
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.post(
            "/api/v1/edit/jobs", json=self._payload(ref_boost=-0.1)
        )
        self.assertEqual(response.status_code, 422)

    def test_unknown_resource_index_returns_404(self):
        response = self.client.post(
            "/api/v1/edit/jobs", json=self._payload(model_index=99)
        )
        self.assertEqual(response.status_code, 404)
        response = self.client.post(
            "/api/v1/edit/jobs", json=self._payload(vae_index=99)
        )
        self.assertEqual(response.status_code, 404)

    def test_invalid_size_returns_422(self):
        # 577 不是 16 的倍数,core domain 校验拒绝
        response = self.client.post(
            "/api/v1/edit/jobs", json=self._payload(width=577)
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_sampler_returns_422(self):
        response = self.client.post(
            "/api/v1/edit/jobs", json=self._payload(sampler="not-a-sampler")
        )
        self.assertEqual(response.status_code, 422)

    def test_submit_when_edit_disabled_returns_404(self):
        # 把 krea2 的 edit_lora 置空,模拟服务端未配置编辑能力
        from dataclasses import replace
        resources = self.core.config.resources[Mode.KREA2]
        new_resources = dict(self.core.config.resources)
        new_resources[Mode.KREA2] = replace(resources, edit_lora=None)
        object.__setattr__(self.core.config, "resources", new_resources)

        response = self.client.post("/api/v1/edit/jobs", json=self._payload())
        self.assertEqual(response.status_code, 404)
        self.assertIn("未配置 krea2 图像编辑", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()