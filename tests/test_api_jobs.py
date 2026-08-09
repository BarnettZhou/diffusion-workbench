import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from diffusion_workbench_api import create_app
from diffusion_workbench_core.domain import Mode
from fake_api_core import FakeApiCore


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.core = FakeApiCore(Path(self._tmp.name))
        self._client_ctx = TestClient(create_app(core_factory=lambda: self.core))
        self.client = self._client_ctx.__enter__()

    def tearDown(self):
        self._client_ctx.__exit__(None, None, None)
        self._tmp.cleanup()


class CreateJobsTests(ApiTestCase):
    def test_sdxl_checkpoint_submit_does_not_require_external_vae(self):
        response = self.client.post(
            "/api/v1/jobs",
            json={
                "mode": "sdxl",
                "model_index": 1,
                "prompt": "a studio portrait",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "cfg": 7,
            },
        )

        self.assertEqual(response.status_code, 202)
        self.assertIsNone(response.json()["jobs"][0]["vae_name"])
        settings, _count = self.core.submitted[-1]
        self.assertEqual(settings.mode, Mode.SDXL)
        self.assertIsNone(settings.vae)
        self.assertIsNone(settings.text_encoder)
        self.assertEqual(settings.model_loader.value, "checkpoint")

    def test_sdxl_checkpoint_rejects_external_vae_index(self):
        response = self.client.post(
            "/api/v1/jobs",
            json={
                "mode": "sdxl",
                "model_index": 1,
                "vae_index": 1,
                "prompt": "portrait",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("内嵌 VAE", response.json()["detail"])

    def _payload(self, **overrides):
        payload = {
            "mode": "krea2",
            "model_index": 1,
            "vae_index": 1,
            "prompt": "an adult studio portrait",
        }
        payload.update(overrides)
        return payload

    def test_post_maps_request_to_fixed_generation_settings(self):
        response = self.client.post("/api/v1/jobs", json=self._payload(count=2))

        self.assertEqual(response.status_code, 202)
        body = response.json()["jobs"]
        self.assertEqual(len(body), 2)
        self.assertIsNotNone(body[0]["batch_id"])
        self.assertEqual(body[0]["batch_id"], body[1]["batch_id"])
        self.assertEqual(body[0]["status"], "queued")
        self.assertEqual(body[0]["seed"], -1)
        self.assertIsNone(body[0]["image_url"])

        settings, count = self.core.submitted[0]
        self.assertEqual(count, 2)
        self.assertEqual(settings.mode, Mode.KREA2)
        self.assertEqual(settings.prompt, "an adult studio portrait")
        self.assertEqual(
            settings.text_encoder,
            self.core.config.resources[Mode.KREA2].text_encoder,
        )
        self.assertEqual(settings.clip_type, "krea2")
        self.assertEqual(settings.model.path.name, "krea.safetensors")
        self.assertEqual(settings.vae.path.name, "krea-vae.safetensors")

    def test_single_job_has_no_batch_id(self):
        response = self.client.post("/api/v1/jobs", json=self._payload())
        self.assertEqual(response.status_code, 202)
        self.assertIsNone(response.json()["jobs"][0]["batch_id"])

    def test_unknown_resource_index_returns_404(self):
        response = self.client.post("/api/v1/jobs", json=self._payload(model_index=99))
        self.assertEqual(response.status_code, 404)

    def test_invalid_size_returns_422(self):
        response = self.client.post("/api/v1/jobs", json=self._payload(width=577))
        self.assertEqual(response.status_code, 422)

    def test_pydantic_rejects_out_of_range_steps(self):
        response = self.client.post("/api/v1/jobs", json=self._payload(steps=101))
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/v1/jobs", json=self._payload(steps=0))
        self.assertEqual(response.status_code, 422)

    def test_zib_submit_maps_new_sampling_parameters(self):
        payload = {
            "mode": "zib",
            "model_index": 1,
            "vae_index": 1,
            "prompt": "a cinematic portrait",
            "negative_prompt": "blurry, watermark",
            "steps": 40,
            "cfg": 4.0,
            "sampler": "dpmpp_2m_sde",
            "scheduler": "sgm_uniform",
        }
        response = self.client.post("/api/v1/jobs", json=payload)

        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["mode"], "zib")
        self.assertEqual(job["negative_prompt"], "blurry, watermark")
        self.assertEqual(job["steps"], 40)
        self.assertEqual(job["cfg"], 4.0)
        self.assertEqual(job["sampler"], "dpmpp_2m_sde")
        self.assertEqual(job["scheduler"], "sgm_uniform")

        settings, _ = self.core.submitted[0]
        self.assertEqual(settings.mode, Mode.ZIB)
        self.assertEqual(settings.negative_prompt, "blurry, watermark")
        self.assertEqual(settings.steps, 40)
        self.assertEqual(settings.cfg, 4.0)
        self.assertEqual(settings.sampler, "dpmpp_2m_sde")
        self.assertEqual(settings.scheduler, "sgm_uniform")
        self.assertEqual(
            settings.text_encoder,
            self.core.config.resources[Mode.ZIB].text_encoder,
        )

    def test_omitted_new_fields_keep_legacy_defaults(self):
        response = self.client.post("/api/v1/jobs", json=self._payload())

        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["negative_prompt"], "")
        self.assertEqual(job["cfg"], 1.0)
        self.assertEqual(job["sampler"], "euler")
        self.assertEqual(job["scheduler"], "simple")

        settings, _ = self.core.submitted[0]
        self.assertEqual(settings.negative_prompt, "")
        self.assertEqual(settings.steps, 8)
        self.assertEqual(settings.cfg, 1.0)
        self.assertEqual(settings.sampler, "euler")
        self.assertEqual(settings.scheduler, "simple")

    def test_invalid_cfg_sampler_scheduler_returns_422(self):
        self.assertEqual(
            self.client.post("/api/v1/jobs", json=self._payload(cfg=0)).status_code, 422
        )
        self.assertEqual(
            self.client.post("/api/v1/jobs", json=self._payload(cfg=-1.5)).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/jobs", json=self._payload(sampler="not_a_sampler")
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/jobs", json=self._payload(scheduler="not_a_scheduler")
            ).status_code,
            422,
        )

    def test_full_sampler_scheduler_list_accepted(self):
        # core 开放的全部采样器/调度器都应能通过 API 校验
        response = self.client.post(
            "/api/v1/jobs", json=self._payload(sampler="ddim", scheduler="karras")
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["sampler"], "ddim")
        self.assertEqual(job["scheduler"], "karras")

    def test_sampling_options_endpoint(self):
        response = self.client.get("/api/v1/sampling-options")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["samplers"]), 44)
        self.assertEqual(len(body["schedulers"]), 9)
        self.assertEqual(body["defaults"], {"sampler": "euler", "scheduler": "simple"})

    def test_count_above_admission_limit_returns_422(self):
        response = self.client.post("/api/v1/jobs", json=self._payload(count=33))
        self.assertEqual(response.status_code, 422)


class GetJobTests(ApiTestCase):
    def test_get_job_returns_persisted_record(self):
        created = self.client.post(
            "/api/v1/jobs",
            json={"mode": "zit", "model_index": 1, "vae_index": 1, "prompt": "p"},
        ).json()["jobs"][0]

        response = self.client.get(f"/api/v1/jobs/{created['id']}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["id"], created["id"])
        self.assertEqual(body["mode"], "zit")
        self.assertEqual(body["negative_prompt"], "")
        self.assertNotIn("output_path", body)
        self.assertNotIn("model_path", body)

    def test_get_zib_job_returns_negative_prompt_and_sampling_params(self):
        created = self.client.post(
            "/api/v1/jobs",
            json={
                "mode": "zib",
                "model_index": 1,
                "vae_index": 1,
                "prompt": "p",
                "negative_prompt": "blurry",
                "steps": 40,
                "cfg": 4.0,
                "sampler": "dpmpp_2m_sde",
                "scheduler": "beta",
            },
        ).json()["jobs"][0]

        response = self.client.get(f"/api/v1/jobs/{created['id']}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "zib")
        self.assertEqual(body["negative_prompt"], "blurry")
        self.assertEqual(body["steps"], 40)
        self.assertEqual(body["cfg"], 4.0)
        self.assertEqual(body["sampler"], "dpmpp_2m_sde")
        self.assertEqual(body["scheduler"], "beta")

    def test_get_unknown_job_returns_404(self):
        response = self.client.get("/api/v1/jobs/does-not-exist")
        self.assertEqual(response.status_code, 404)


class ListJobsTests(ApiTestCase):
    def test_list_jobs_passes_filters_and_cursor(self):
        self.client.post(
            "/api/v1/jobs",
            json={"mode": "zit", "model_index": 1, "vae_index": 1, "prompt": "p"},
        )
        response = self.client.get(
            "/api/v1/jobs",
            params={"status": "queued", "mode": "zit", "limit": 10},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["jobs"]), 1)
        self.assertIn("next_cursor", body)

        call = self.core.list_jobs_calls[-1]
        self.assertEqual(call["status"], "queued")
        self.assertEqual(call["mode"], Mode.ZIT)
        self.assertEqual(call["limit"], 10)

    def test_mode_filter_accepts_zib(self):
        self.client.post(
            "/api/v1/jobs",
            json={"mode": "zib", "model_index": 1, "vae_index": 1, "prompt": "p"},
        )
        response = self.client.get("/api/v1/jobs", params={"mode": "zib"})
        self.assertEqual(response.status_code, 200)
        jobs = response.json()["jobs"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["mode"], "zib")
        self.assertIn("negative_prompt", jobs[0])
        self.assertEqual(self.core.list_jobs_calls[-1]["mode"], Mode.ZIB)

    def test_invalid_cursor_returns_422(self):
        response = self.client.get("/api/v1/jobs", params={"cursor": "bad-cursor"})
        self.assertEqual(response.status_code, 422)

    def test_invalid_status_returns_422(self):
        response = self.client.get("/api/v1/jobs", params={"status": "bogus"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
