import io
import unittest

from PIL import Image

from diffusion_workbench_core.domain import Mode
from test_api_jobs import ApiTestCase


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 64, 32)).save(buffer, format="PNG")
    return buffer.getvalue()


class RebalanceInfoTests(ApiTestCase):
    def test_info_reports_rebalance_enabled_and_defaults(self):
        response = self.client.get("/api/v1/edit/info")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["rebalance"],
            {
                "enabled": True,
                "defaults": {
                    "token_tier": "normal",
                    "token_tiers": ["low", "normal", "high", "max"],
                    "max_reference_images": 4,
                },
            },
        )

    def test_info_reports_rebalance_disabled_when_krea2_missing(self):
        # 移除 krea2 资源,模拟服务端未配置 krea2 的情况
        new_resources = dict(self.core.config.resources)
        del new_resources[Mode.KREA2]
        object.__setattr__(self.core.config, "resources", new_resources)
        response = self.client.get("/api/v1/edit/info")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["rebalance"]["enabled"])


class RebalanceJobsTests(ApiTestCase):
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
            "prompt": "参考这张图的构图画一只猫",
            "reference_image_ids": [self._upload_image()],
        }
        payload.update(overrides)
        return payload

    def test_submit_maps_request_to_rebalance_settings(self):
        response = self.client.post("/api/v1/edit/rebalance-jobs", json=self._payload(count=2))
        self.assertEqual(response.status_code, 202)
        body = response.json()["jobs"]
        self.assertEqual(len(body), 2)
        job = body[0]
        self.assertEqual(job["mode"], "krea2-rebalance")
        self.assertEqual(job["status"], "queued")
        self.assertIsNotNone(job["batch_id"])
        self.assertEqual(job["model_name"], "krea.safetensors")
        self.assertEqual(job["vae_name"], "krea-vae.safetensors")
        # reference_image_urls 指向受控目录的 id;tokens 缺省时为空列表(全部按 normal)
        self.assertEqual(len(job["reference_image_urls"]), 1)
        image_id = job["reference_image_urls"][0].rsplit("/", 1)[-1]
        self.assertRegex(image_id, r"^[0-9a-f]{32}\.png$")
        self.assertEqual(job["reference_image_tokens"], [])
        # 编辑字段保持 null
        self.assertIsNone(job["input_image_url"])
        self.assertIsNone(job["grounding_px"])
        self.assertIsNone(job["ref_boost"])

        settings, count = self.core.submitted[-1]
        self.assertEqual(count, 2)
        self.assertEqual(settings.mode, Mode.KREA2_REBALANCE)
        self.assertEqual(settings.model.path.name, "krea.safetensors")
        self.assertEqual(settings.vae.path.name, "krea-vae.safetensors")
        self.assertEqual(
            settings.text_encoder,
            self.core.root / "krea-te.safetensors",
        )
        self.assertEqual(settings.clip_type, "krea2")
        # reference_images 是服务端解析后的受控路径(不是客户端提交的内容)
        self.assertEqual(len(settings.reference_images), 1)
        self.assertTrue(str(settings.reference_images[0]).endswith(image_id))
        self.assertEqual(settings.reference_image_tokens, ())

    def test_multiple_references_with_tokens_propagate(self):
        ids = [self._upload_image(), self._upload_image()]
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs",
            json=self._payload(
                reference_image_ids=ids,
                reference_image_tokens=["low", "max"],
            ),
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(len(job["reference_image_urls"]), 2)
        self.assertEqual(job["reference_image_tokens"], ["low", "max"])
        settings, _ = self.core.submitted[-1]
        self.assertEqual(len(settings.reference_images), 2)
        self.assertEqual(settings.reference_image_tokens, ("low", "max"))

    def test_get_rebalance_job_returns_reference_fields(self):
        created = self.client.post(
            "/api/v1/edit/rebalance-jobs", json=self._payload()
        ).json()["jobs"][0]
        response = self.client.get(f"/api/v1/jobs/{created['id']}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "krea2-rebalance")
        self.assertEqual(len(body["reference_image_urls"]), 1)

    def test_non_rebalance_job_keeps_reference_fields_empty(self):
        response = self.client.post(
            "/api/v1/jobs",
            json={"mode": "krea2", "model_index": 1, "vae_index": 1, "text_encoder_index": 1, "prompt": "p"},
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["reference_image_urls"], [])
        self.assertEqual(job["reference_image_tokens"], [])

    def test_missing_reference_image_ids_returns_422(self):
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs",
            json={
                "model_index": 1,
                "vae_index": 1,
                "text_encoder_index": 1,
                "prompt": "p",
            },
        )
        self.assertEqual(response.status_code, 422)
        # 空列表同样 422(至少 1 张参考图)
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs",
            json={
                "model_index": 1,
                "vae_index": 1,
                "text_encoder_index": 1,
                "prompt": "p",
                "reference_image_ids": [],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_too_many_reference_images_returns_422(self):
        ids = [self._upload_image() for _ in range(5)]
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs",
            json=self._payload(reference_image_ids=ids),
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_token_tier_returns_422(self):
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs",
            json=self._payload(reference_image_tokens=["ultra"]),
        )
        self.assertEqual(response.status_code, 422)

    def test_token_count_mismatch_returns_422(self):
        ids = [self._upload_image(), self._upload_image()]
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs",
            json=self._payload(
                reference_image_ids=ids,
                reference_image_tokens=["normal"],
            ),
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("数量", response.json()["detail"])

    def test_unknown_reference_image_id_returns_404(self):
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs",
            json=self._payload(reference_image_ids=["f" * 32 + ".png"]),
        )
        self.assertEqual(response.status_code, 404)

    def test_unknown_resource_index_returns_404(self):
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs", json=self._payload(model_index=99)
        )
        self.assertEqual(response.status_code, 404)
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs", json=self._payload(vae_index=99)
        )
        self.assertEqual(response.status_code, 404)

    def test_invalid_size_returns_422(self):
        # 577 不是 16 的倍数,core domain 校验拒绝
        response = self.client.post(
            "/api/v1/edit/rebalance-jobs", json=self._payload(width=577)
        )
        self.assertEqual(response.status_code, 422)

    def test_submit_when_rebalance_disabled_returns_404(self):
        # 移除 krea2 资源,模拟服务端未配置 krea2 的情况
        new_resources = dict(self.core.config.resources)
        del new_resources[Mode.KREA2]
        object.__setattr__(self.core.config, "resources", new_resources)

        response = self.client.post(
            "/api/v1/edit/rebalance-jobs", json=self._payload()
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("未配置 krea2 资源", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
