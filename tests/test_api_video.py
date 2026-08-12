import os
import unittest

from diffusion_workbench_api.schemas import public_event
from diffusion_workbench_core.domain import VideoModel
from test_api_jobs import ApiTestCase

WAN = VideoModel.WAN22_TI2V_5B.value
WAN_14B = VideoModel.WAN22_I2V_14B.value
H3 = VideoModel.MINIMAX_H3.value


class VideoModelsTests(ApiTestCase):
    def test_list_video_models(self):
        response = self.client.get("/api/v1/video/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "video_models": [
                    {
                        "video_model": WAN,
                        "label": "Wan 2.2 TI2V-5B",
                        "generation_types": ["t2v", "i2v"],
                        "requires_input_image": False,
                    },
                    {
                        "video_model": WAN_14B,
                        "label": "Wan 2.2 I2V-14B",
                        "generation_types": ["i2v"],
                        "requires_input_image": True,
                    },
                    {
                        "video_model": H3,
                        "label": "MiniMax H3",
                        "generation_types": ["t2v", "i2v", "r2v"],
                        "requires_input_image": False,
                    },
                ]
            },
        )

    def test_list_video_resources(self):
        response = self.client.get(f"/api/v1/video/models/{WAN}/resources")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            [m["name"] for m in body["models"]], ["wan2.2-ti2v-5b.safetensors"]
        )
        self.assertEqual([v["name"] for v in body["vaes"]], ["wan-vae.safetensors"])
        self.assertEqual(body["models"][0]["index"], 1)
        # 视频资源没有 alias 机制
        self.assertIsNone(body["models"][0]["alias"])

    def test_list_14b_fp8_resources(self):
        response = self.client.get(f"/api/v1/video/models/{WAN_14B}/resources")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            [model["name"] for model in body["models"]],
            [
                "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            ],
        )
        self.assertEqual(
            [vae["name"] for vae in body["vaes"]],
            ["wan_2.1_vae.safetensors"],
        )

    def test_unconfigured_video_model_returns_404(self):
        self.core.config.video_resources.clear()
        response = self.client.get(f"/api/v1/video/models/{WAN}/resources")
        self.assertEqual(response.status_code, 404)

    def test_status_hides_h3_audio_vae_path(self):
        self.core.loaded_resources = {
            "workload": H3,
            "model": r"C:\models\h3.safetensors",
            "vae": r"C:\models\video-vae.safetensors",
            "audio_vae": r"C:\models\audio-vae.safetensors",
            "text_encoder": r"C:\models\qwen.safetensors",
            "clip_type": "minimax",
        }
        response = self.client.get("/api/v1/status")
        self.assertEqual(
            response.json()["loaded_resources"]["audio_vae"],
            "audio-vae.safetensors",
        )

    def test_unknown_video_model_value_returns_422(self):
        response = self.client.get("/api/v1/video/models/nope/resources")
        self.assertEqual(response.status_code, 422)


class CreateVideoJobsTests(ApiTestCase):
    def _payload(self, **overrides):
        payload = {
            "video_model": WAN,
            "model_index": 1,
            "vae_index": 1,
            "prompt": "一只猫在雪地里奔跑",
        }
        payload.update(overrides)
        return payload

    def test_submit_maps_request_and_injects_fixed_text_encoder(self):
        response = self.client.post(
            "/api/v1/video/jobs",
            json=self._payload(count=2, duration_seconds=3, fps=16, seed=42),
        )
        self.assertEqual(response.status_code, 202)
        body = response.json()["jobs"]
        self.assertEqual(len(body), 2)
        self.assertIsNotNone(body[0]["batch_id"])
        self.assertEqual(body[0]["batch_id"], body[1]["batch_id"])
        self.assertEqual(body[0]["status"], "queued")
        self.assertEqual(body[0]["video_model"], WAN)
        self.assertEqual(body[0]["generation_type"], "t2v")
        self.assertEqual(body[0]["duration_seconds"], 3)
        self.assertEqual(body[0]["fps"], 16)
        self.assertEqual(body[0]["length"], 3 * 16 + 1)
        self.assertEqual(body[0]["seed"], 42)
        self.assertIsNone(body[0]["video_url"])
        self.assertEqual(body[0]["model_name"], "wan2.2-ti2v-5b.safetensors")
        self.assertEqual(body[0]["vae_name"], "wan-vae.safetensors")

        settings, count = self.core.video_submitted[0]
        self.assertEqual(count, 2)
        self.assertEqual(settings.video_model, VideoModel.WAN22_TI2V_5B)
        self.assertEqual(settings.prompt, "一只猫在雪地里奔跑")
        self.assertEqual(settings.sampler, "uni_pc")
        self.assertEqual(settings.scheduler, "simple")
        self.assertEqual(settings.cfg, 5.0)
        self.assertIsNone(settings.input_image)
        # text encoder 固定注入服务端配置,不接受客户端指定
        self.assertEqual(
            settings.text_encoder,
            self.core.config.video_resources[VideoModel.WAN22_TI2V_5B].text_encoder,
        )

    def test_single_job_has_no_batch_id(self):
        response = self.client.post("/api/v1/video/jobs", json=self._payload())
        self.assertEqual(response.status_code, 202)
        self.assertIsNone(response.json()["jobs"][0]["batch_id"])

    def test_h3_uses_fixed_audio_vae_and_model_defaults(self):
        response = self.client.post(
            "/api/v1/video/jobs",
            json=self._payload(video_model=H3),
        )

        self.assertEqual(response.status_code, 202)
        body = response.json()["jobs"][0]
        self.assertEqual(body["video_model"], H3)
        self.assertEqual(body["fps"], 24)
        self.assertEqual(body["length"], 124)
        self.assertEqual(body["cfg"], 1.0)
        self.assertEqual(body["shift"], 12.0)
        self.assertEqual(body["sampler"], "res_multistep")
        settings, _ = self.core.video_submitted[-1]
        self.assertEqual(
            settings.audio_vae,
            self.core.config.video_resources[VideoModel.MINIMAX_H3].audio_vae,
        )

    def test_h3_ref2va_accepts_single_reference_image(self):
        image_id = self.client.post(
            "/api/v1/video/input-images",
            content=_png_bytes(),
            headers={"Content-Type": "image/png"},
        ).json()["id"]
        response = self.client.post(
            "/api/v1/video/jobs",
            json=self._payload(
                video_model=H3,
                model_index=2,
                reference_image_id=image_id,
            ),
        )
        self.assertEqual(response.status_code, 202)
        body = response.json()["jobs"][0]
        self.assertEqual(body["generation_type"], "r2v")
        self.assertEqual(body["reference_image_url"], f"/api/v1/video/input-images/{image_id}")
        settings, _ = self.core.video_submitted[-1]
        self.assertEqual(settings.reference_image.name, image_id)
        self.assertIsNone(settings.input_image)

    def test_h3_rejects_reference_image_with_fl2va_or_first_frame(self):
        image_id = self.client.post(
            "/api/v1/video/input-images",
            content=_png_bytes(),
            headers={"Content-Type": "image/png"},
        ).json()["id"]
        response = self.client.post(
            "/api/v1/video/jobs",
            json=self._payload(video_model=H3, model_index=1, reference_image_id=image_id),
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.post(
            "/api/v1/video/jobs",
            json=self._payload(video_model=H3, model_index=2, reference_image_id=image_id, input_image_id=image_id),
        )
        self.assertEqual(response.status_code, 422)

    def test_wan_rejects_reference_image(self):
        image_id = self.client.post(
            "/api/v1/video/input-images",
            content=_png_bytes(),
            headers={"Content-Type": "image/png"},
        ).json()["id"]
        response = self.client.post(
            "/api/v1/video/jobs",
            json=self._payload(video_model=WAN, reference_image_id=image_id),
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("只支持 MiniMax H3 Ref2VA", response.json()["detail"])

    def test_unknown_resource_index_returns_404(self):
        response = self.client.post(
            "/api/v1/video/jobs", json=self._payload(model_index=99)
        )
        self.assertEqual(response.status_code, 404)
        response = self.client.post("/api/v1/video/jobs", json=self._payload(vae_index=99))
        self.assertEqual(response.status_code, 404)

    def test_invalid_sampler_returns_422(self):
        response = self.client.post(
            "/api/v1/video/jobs", json=self._payload(sampler="not-a-sampler")
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_size_returns_422(self):
        response = self.client.post("/api/v1/video/jobs", json=self._payload(width=705))
        self.assertEqual(response.status_code, 422)

    def test_pydantic_rejects_out_of_range_fps(self):
        response = self.client.post("/api/v1/video/jobs", json=self._payload(fps=0))
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/v1/video/jobs", json=self._payload(fps=121))
        self.assertEqual(response.status_code, 422)

    def test_submit_maps_shift_and_defaults_to_8(self):
        # 默认 shift=8
        response = self.client.post("/api/v1/video/jobs", json=self._payload())
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["jobs"][0]["shift"], 8.0)
        settings, _ = self.core.video_submitted[-1]
        self.assertEqual(settings.shift, 8.0)
        # 显式指定
        response = self.client.post(
            "/api/v1/video/jobs", json=self._payload(shift=10.5)
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["jobs"][0]["shift"], 10.5)
        settings, _ = self.core.video_submitted[-1]
        self.assertEqual(settings.shift, 10.5)

    def test_pydantic_rejects_out_of_range_shift(self):
        self.assertEqual(
            self.client.post("/api/v1/video/jobs", json=self._payload(shift=-1)).status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/api/v1/video/jobs", json=self._payload(shift=101)).status_code,
            422,
        )

    def test_unknown_video_model_value_returns_422(self):
        response = self.client.post(
            "/api/v1/video/jobs", json=self._payload(video_model="nope")
        )
        self.assertEqual(response.status_code, 422)


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 64, 32)).save(buffer, format="PNG")
    return buffer.getvalue()


class VideoInputImageTests(ApiTestCase):
    def _upload(self, data=None, content_type="image/png"):
        return self.client.post(
            "/api/v1/video/input-images",
            content=data if data is not None else _png_bytes(),
            headers={"Content-Type": content_type},
        )

    def test_upload_and_get_roundtrip(self):
        response = self._upload()
        self.assertEqual(response.status_code, 201)
        image_id = response.json()["id"]
        self.assertRegex(image_id, r"^[0-9a-f]{32}\.png$")
        # 文件落在受控目录,客户端只拿到 id 与受控 URL
        served = self.client.get(response.json()["url"])
        self.assertEqual(served.status_code, 200)
        self.assertEqual(served.headers["content-type"], "image/png")
        self.assertEqual(served.content, _png_bytes())

    def test_upload_rejects_unsupported_media_type(self):
        response = self._upload(content_type="image/gif")
        self.assertEqual(response.status_code, 422)

    def test_upload_rejects_empty_and_invalid_content(self):
        self.assertEqual(self._upload(data=b"").status_code, 422)
        self.assertEqual(self._upload(data=b"not an image").status_code, 422)

    def test_get_unknown_input_image_returns_404(self):
        self.assertEqual(
            self.client.get(
                f"/api/v1/video/input-images/{'0' * 32}.png"
            ).status_code,
            404,
        )
        # 非法 id(含目录穿越尝试)同样 404,不会拼出任何路径
        self.assertEqual(
            self.client.get("/api/v1/video/input-images/..%2F..%2Fjobs.sqlite3").status_code,
            404,
        )

    def test_submit_with_input_image_becomes_i2v(self):
        image_id = self._upload().json()["id"]
        response = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN,
                "model_index": 1,
                "vae_index": 1,
                "prompt": "让画面动起来",
                "input_image_id": image_id,
            },
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["generation_type"], "i2v")
        self.assertEqual(job["input_image_url"], f"/api/v1/video/input-images/{image_id}")
        # Core 收到的是服务端解析出的受控路径,而非客户端提交的内容
        settings, _ = self.core.video_submitted[-1]
        self.assertEqual(settings.input_image.name, image_id)

    def test_submit_t2v_has_no_input_image_url(self):
        response = self.client.post(
            "/api/v1/video/jobs",
            json={"video_model": WAN, "model_index": 1, "vae_index": 1, "prompt": "猫"},
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["generation_type"], "t2v")
        self.assertIsNone(job["input_image_url"])

    def test_14b_requires_input_image(self):
        response = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN_14B,
                "model_index": 1,
                "vae_index": 1,
                "prompt": "让人物自然转头",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("必须提供输入图片", response.json()["detail"])

    def test_submit_14b_fp8_i2v(self):
        image_id = self._upload().json()["id"]
        response = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN_14B,
                "model_index": 1,
                "vae_index": 1,
                "prompt": "让人物自然转头",
                "input_image_id": image_id,
                "width": 512,
                "height": 512,
            },
        )
        self.assertEqual(response.status_code, 202)
        job = response.json()["jobs"][0]
        self.assertEqual(job["video_model"], WAN_14B)
        self.assertEqual(job["generation_type"], "i2v")
        self.assertEqual(
            job["model_name"],
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        )
        self.assertEqual(job["vae_name"], "wan_2.1_vae.safetensors")
        settings, _ = self.core.video_submitted[-1]
        self.assertEqual(settings.video_model, VideoModel.WAN22_I2V_14B)
        self.assertEqual(settings.sampler, "euler")
        self.assertEqual(
            settings.text_encoder,
            self.core.config.video_resources[VideoModel.WAN22_I2V_14B].text_encoder,
        )

    def test_submit_14b_with_explicit_high_and_low_models(self):
        image_id = self._upload().json()["id"]

        response = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN_14B,
                "high_model_index": 1,
                "low_model_index": 2,
                "vae_index": 1,
                "prompt": "让人物自然转头",
                "input_image_id": image_id,
            },
        )

        self.assertEqual(response.status_code, 202)
        settings, _ = self.core.video_submitted[-1]
        self.assertEqual(
            settings.model.path.name,
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        )

    def test_submit_14b_rejects_incomplete_or_mismatched_explicit_pair(self):
        image_id = self._upload().json()["id"]
        base = {
            "video_model": WAN_14B,
            "vae_index": 1,
            "prompt": "让人物自然转头",
            "input_image_id": image_id,
        }

        incomplete = self.client.post(
            "/api/v1/video/jobs", json={**base, "high_model_index": 1}
        )
        reversed_pair = self.client.post(
            "/api/v1/video/jobs",
            json={**base, "high_model_index": 2, "low_model_index": 1},
        )

        self.assertEqual(incomplete.status_code, 422)
        self.assertIn("同时选择", incomplete.json()["detail"])
        self.assertEqual(reversed_pair.status_code, 422)
        self.assertIn("high_model_index", reversed_pair.json()["detail"])

    def test_submit_unknown_input_image_id_returns_404(self):
        response = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN,
                "model_index": 1,
                "vae_index": 1,
                "prompt": "猫",
                "input_image_id": "f" * 32 + ".png",
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_submit_rejects_malformed_input_image_id(self):
        response = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN,
                "model_index": 1,
                "vae_index": 1,
                "prompt": "猫",
                "input_image_id": "../../jobs.sqlite3",
            },
        )
        self.assertEqual(response.status_code, 404)


class VideoInputFromAlbumTests(ApiTestCase):
    """相册图片直接导入为 I2V 输入图片(服务端本地复制)。"""

    def _write_album_png(self, relpath="2026-08-09/zit-00001.png"):
        path = self.core.config.output_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png_bytes())
        return relpath

    def test_import_from_album_roundtrip(self):
        relpath = self._write_album_png()
        response = self.client.post(
            "/api/v1/video/input-images/from-album",
            json={"id": relpath, "dir": "output"},
        )
        self.assertEqual(response.status_code, 201)
        image_id = response.json()["id"]
        self.assertRegex(image_id, r"^[0-9a-f]{32}\.png$")
        # 导入的图片按同一受控 URL 可读,可直接用于提交 I2V 任务
        served = self.client.get(response.json()["url"])
        self.assertEqual(served.status_code, 200)
        submit = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN,
                "model_index": 1,
                "vae_index": 1,
                "prompt": "让画面动起来",
                "input_image_id": image_id,
            },
        )
        self.assertEqual(submit.status_code, 202)
        self.assertEqual(submit.json()["jobs"][0]["generation_type"], "i2v")

    def test_import_missing_file_returns_404(self):
        response = self.client.post(
            "/api/v1/video/input-images/from-album",
            json={"id": "2026-08-09/nope.png", "dir": "output"},
        )
        self.assertEqual(response.status_code, 404)

    def test_import_rejects_non_image(self):
        path = self.core.config.output_dir / "2026-08-09" / "wan2.2-ti2v-5b-00001.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not an image")
        response = self.client.post(
            "/api/v1/video/input-images/from-album",
            json={"id": "2026-08-09/wan2.2-ti2v-5b-00001.mp4", "dir": "output"},
        )
        self.assertEqual(response.status_code, 422)

    def test_import_rejects_path_traversal(self):
        response = self.client.post(
            "/api/v1/video/input-images/from-album",
            json={"id": "../jobs.sqlite3", "dir": "output"},
        )
        # 相册 resolve 对越界路径返回 400
        self.assertEqual(response.status_code, 400)


class VideoAlbumMetadataTests(ApiTestCase):
    """相册 mp4 的 metadata:参数来自 jobs 表的视频任务记录。"""

    def _submit_and_write_mp4(self):
        response = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN,
                "model_index": 1,
                "vae_index": 1,
                "prompt": "一只猫在雪地里奔跑",
                "shift": 10,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["jobs"][0]["id"]
        job = self.core.video_jobs[job_id]
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        job.output_path.write_bytes(b"fake-mp4")
        return job

    def test_mp4_metadata_returns_video_params(self):
        job = self._submit_and_write_mp4()
        relpath = f"{job.output_path.parent.name}/{job.output_path.name}"
        response = self.client.get(f"/api/v1/album/image/{relpath}/metadata")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["video_model"], WAN)
        self.assertEqual(body["generation_type"], "t2v")
        self.assertEqual(body["prompt"], "一只猫在雪地里奔跑")
        self.assertEqual(body["model_name"], "wan2.2-ti2v-5b.safetensors")
        self.assertEqual(body["vae_name"], "wan-vae.safetensors")
        self.assertEqual(body["shift"], 10.0)
        self.assertEqual(body["duration_seconds"], 5)
        self.assertEqual(body["fps"], 24)
        # 不泄露绝对路径
        self.assertNotIn(str(self.core.config.output_dir), str(body))

    def test_mp4_without_record_returns_404(self):
        path = self.core.config.output_dir / "2026-08-09" / "wan2.2-ti2v-5b-00099.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-mp4")
        response = self.client.get(
            "/api/v1/album/image/2026-08-09/wan2.2-ti2v-5b-00099.mp4/metadata"
        )
        self.assertEqual(response.status_code, 404)


class VideoJobQueryTests(ApiTestCase):
    def _submit(self):
        response = self.client.post(
            "/api/v1/video/jobs",
            json={
                "video_model": WAN,
                "model_index": 1,
                "vae_index": 1,
                "prompt": "一只猫",
            },
        )
        self.assertEqual(response.status_code, 202)
        return response.json()["jobs"][0]

    def test_get_video_job(self):
        job = self._submit()
        response = self.client.get(f"/api/v1/video/jobs/{job['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], job["id"])
        self.assertEqual(response.json()["video_model"], WAN)

    def test_get_unknown_video_job_returns_404(self):
        response = self.client.get("/api/v1/video/jobs/not-exist")
        self.assertEqual(response.status_code, 404)

    def test_video_file_serving(self):
        job = self._submit()
        record = self.core.video_jobs[job["id"]]
        # queued 且文件不存在 → 404 任务尚未完成
        queued = self.client.get(f"/api/v1/videos/{job['id']}")
        self.assertEqual(queued.status_code, 404)
        self.assertIn("尚未完成", queued.json()["detail"])

        # 文件生成后可以下载,media_type 为 video/mp4,job 响应出现 video_url
        record.output_path.parent.mkdir(parents=True, exist_ok=True)
        record.output_path.write_bytes(b"fake-mp4")
        served = self.client.get(f"/api/v1/videos/{job['id']}")
        self.assertEqual(served.status_code, 200)
        self.assertEqual(served.headers["content-type"], "video/mp4")
        self.assertEqual(served.content, b"fake-mp4")
        detail = self.client.get(f"/api/v1/video/jobs/{job['id']}").json()
        self.assertEqual(detail["video_url"], f"/api/v1/videos/{job['id']}")

        # 非排队状态文件丢失 → 410
        record.output_path.unlink()
        object.__setattr__(record, "status", "completed")
        gone = self.client.get(f"/api/v1/videos/{job['id']}")
        self.assertEqual(gone.status_code, 410)

    def test_unknown_video_file_returns_404(self):
        response = self.client.get("/api/v1/videos/not-exist")
        self.assertEqual(response.status_code, 404)


class VideoEventTests(ApiTestCase):
    def test_public_event_marks_video_url_for_video_artifact(self):
        event = {
            "type": "job_finished",
            "job_id": "job-1",
            "status": "completed",
            "artifact_type": "video",
            "output_path": "/secret/output/wan.mp4",
        }
        result = public_event(event, 1)
        self.assertEqual(result["video_url"], "/api/v1/videos/job-1")
        self.assertEqual(result["artifact_type"], "video")
        self.assertNotIn("image_url", result)
        self.assertNotIn("output_path", result)

    def test_public_event_image_artifact_keeps_image_url(self):
        event = {
            "type": "job_finished",
            "job_id": "job-2",
            "status": "completed",
            "artifact_type": "image",
            "output_path": "/secret/output/zit.png",
        }
        result = public_event(event, 1)
        self.assertEqual(result["image_url"], "/api/v1/images/job-2")
        self.assertNotIn("video_url", result)


class VideoModelCardTests(ApiTestCase):
    def test_list_video_model_cards(self):
        response = self.client.get(f"/api/v1/video-models/{WAN}")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "wan2.2-ti2v-5b.safetensors")
        self.assertEqual(models[0]["mode"], WAN)
        self.assertIsNone(models[0]["alias"])
        self.assertEqual(models[0]["note"], "")
        self.assertFalse(models[0]["has_cover"])

    def test_unconfigured_video_model_returns_404(self):
        self.core.config.video_resources.clear()
        response = self.client.get(f"/api/v1/video-models/{WAN}")
        self.assertEqual(response.status_code, 404)

    def test_info_supports_note_but_rejects_alias(self):
        name = "wan2.2-ti2v-5b.safetensors"
        ok = self.client.put(
            f"/api/v1/video-models/{WAN}/{name}/info", json={"note": "主力视频模型"}
        )
        self.assertEqual(ok.status_code, 200)
        models = self.client.get(f"/api/v1/video-models/{WAN}").json()["models"]
        self.assertEqual(models[0]["note"], "主力视频模型")

        rejected = self.client.put(
            f"/api/v1/video-models/{WAN}/{name}/info", json={"alias": "x"}
        )
        self.assertEqual(rejected.status_code, 422)

    def test_unknown_video_model_name_returns_404(self):
        response = self.client.put(
            f"/api/v1/video-models/{WAN}/nope.safetensors/info", json={"note": "x"}
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
