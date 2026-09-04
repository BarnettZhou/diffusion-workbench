import os
import unittest

from PIL import Image

from diffusion_workbench_core.png_metadata import (
    build_generation_metadata,
    create_png_info,
)
from test_api_jobs import ApiTestCase


class AlbumTests(ApiTestCase):
    def _metadata_command(self):
        return {
            "job_id": "job-x",
            "batch_id": None,
            "mode": "zit",
            "prompt": "一只猫",
            "negative_prompt": "模糊",
            "width": 576,
            "height": 576,
            "steps": 8,
            "seed": 42,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "model_path": str(self.core.root / "m.safetensors"),
            "vae_path": str(self.core.root / "v.safetensors"),
            "text_encoder_path": str(self.core.root / "te.safetensors"),
            "clip_type": "stable_diffusion",
        }

    def _png(self, relpath: str, mtime_ns: int, with_metadata=False, metadata=None):
        path = self.core.config.output_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if with_metadata:
            if metadata is None:
                metadata = build_generation_metadata(
                    self._metadata_command(),
                    {"comfyui": "0.1"},
                )
            Image.new("RGB", (16, 8)).save(path, pnginfo=create_png_info(metadata))
        else:
            Image.new("RGB", (16, 8)).save(path)
        os.utime(path, ns=(mtime_ns, mtime_ns))
        return path

    def test_list_order_and_pagination(self):
        self._png("2026-07-26/zit-00001.png", 1000)
        self._png("2026-07-27/zit-00002.png", 3000)
        self._png("2026-07-27/krea2-00003.png", 2000)

        first = self.client.get("/api/v1/album", params={"limit": 2})
        self.assertEqual(first.status_code, 200)
        body = first.json()
        self.assertEqual(
            [img["name"] for img in body["images"]],
            ["zit-00002.png", "krea2-00003.png"],
        )
        self.assertEqual(body["images"][0]["width"], 16)
        self.assertEqual(body["images"][0]["height"], 8)
        self.assertGreater(body["images"][0]["size_bytes"], 0)
        self.assertIsNotNone(body["next_cursor"])

        second = self.client.get(
            "/api/v1/album", params={"limit": 2, "cursor": body["next_cursor"]}
        )
        self.assertEqual(second.status_code, 200)
        body2 = second.json()
        self.assertEqual([img["name"] for img in body2["images"]], ["zit-00001.png"])
        self.assertIsNone(body2["next_cursor"])

    def test_deleted_file_disappears_from_index(self):
        path = self._png("2026-07-27/zit-00001.png", 1000)
        first = self.client.get("/api/v1/album")
        self.assertEqual(len(first.json()["images"]), 1)

        path.unlink()
        second = self.client.get("/api/v1/album")
        self.assertEqual(len(second.json()["images"]), 0)

    def test_image_download_and_traversal_rejected(self):
        self._png("2026-07-27/zit-00001.png", 1000)
        response = self.client.get("/api/v1/album/image/2026-07-27/zit-00001.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

        outside = self.client.get("/api/v1/album/image/..%2F..%2Fsecret.png")
        self.assertIn(outside.status_code, (400, 404, 422))

        missing = self.client.get("/api/v1/album/image/2026-07-27/gone.png")
        self.assertEqual(missing.status_code, 404)

    def test_delete_image(self):
        path = self._png("2026-07-27/zit-00001.png", 1000)
        response = self.client.delete("/api/v1/album/image/2026-07-27/zit-00001.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], "2026-07-27/zit-00001.png")
        self.assertFalse(path.exists())
        self.assertEqual(len(self.client.get("/api/v1/album").json()["images"]), 0)

        again = self.client.delete("/api/v1/album/image/2026-07-27/zit-00001.png")
        self.assertEqual(again.status_code, 404)

        outside = self.client.delete("/api/v1/album/image/..%2F..%2Fsecret.png")
        self.assertIn(outside.status_code, (400, 404, 422))

    def test_batch_delete_images(self):
        path_a = self._png("2026-07-27/zit-00001.png", 1000)
        path_b = self._png("2026-07-27/zit-00002.png", 2000)
        self._png("2026-07-27/zit-00003.png", 3000)
        # 混入一个已不存在的文件:幂等视为删除成功
        response = self.client.post(
            "/api/v1/album/batch-delete",
            json={
                "relpaths": [
                    "2026-07-27/zit-00001.png",
                    "2026-07-27/zit-00002.png",
                    "2026-07-27/zit-00003.png",
                    "2026-07-27/gone.png",
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            sorted(response.json()["deleted"]),
            [
                "2026-07-27/gone.png",
                "2026-07-27/zit-00001.png",
                "2026-07-27/zit-00002.png",
                "2026-07-27/zit-00003.png",
            ],
        )
        self.assertEqual(response.json()["failed"], [])
        self.assertFalse(path_a.exists())
        self.assertFalse(path_b.exists())
        self.assertEqual(len(self.client.get("/api/v1/album").json()["images"]), 0)

    def test_batch_delete_validation(self):
        for payload in (
            {},
            {"relpaths": []},
            {"relpaths": "not-a-list"},
            {"relpaths": [1, 2]},
            {"relpaths": [""]},
        ):
            response = self.client.post("/api/v1/album/batch-delete", json=payload)
            self.assertEqual(response.status_code, 422, f"{payload} 应被拒绝")

    def test_metadata_endpoint(self):
        self._png("2026-07-27/zit-meta.png", 1000, with_metadata=True)
        self._png("2026-07-27/zit-plain.png", 1000)

        rich = self.client.get("/api/v1/album/image/2026-07-27/zit-meta.png/metadata")
        self.assertEqual(rich.status_code, 200)
        body = rich.json()
        self.assertEqual(body["prompt"], "一只猫")
        self.assertEqual(body["negative_prompt"], "模糊")
        self.assertEqual(body["negative_conditioning"], "positive_reused")
        self.assertEqual(body["model_name"], "m.safetensors")
        self.assertNotIn(str(self.core.root), str(body))

        plain = self.client.get("/api/v1/album/image/2026-07-27/zit-plain.png/metadata")
        self.assertEqual(plain.status_code, 404)

    def test_v1_metadata_returns_compat_defaults(self):
        metadata = build_generation_metadata(self._metadata_command(), {"comfyui": "0.1"})
        metadata["schema_version"] = 1
        for key in ("negative_prompt", "negative_conditioning"):
            metadata["parameters"].pop(key, None)
        self._png("2026-07-27/zit-v1.png", 1000, with_metadata=True, metadata=metadata)

        response = self.client.get("/api/v1/album/image/2026-07-27/zit-v1.png/metadata")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["negative_prompt"], "")
        self.assertIsNone(body["negative_conditioning"])

    def test_invalid_cursor_returns_422(self):
        response = self.client.get("/api/v1/album", params={"cursor": "bad"})
        self.assertEqual(response.status_code, 422)

    def test_album_dirs_crud(self):
        body = self.client.get("/api/v1/album/dirs").json()
        self.assertEqual([d["id"] for d in body["dirs"]], ["output"])
        self.assertTrue(body["dirs"][0]["builtin"])

        extra = self.core.root / "extra"
        (extra / "sub").mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(extra / "sub" / "x.png")

        created = self.client.post(
            "/api/v1/album/dirs", json={"name": "额外目录", "path": str(extra)}
        )
        self.assertEqual(created.status_code, 201)
        dir_id = created.json()["id"]

        page = self.client.get("/api/v1/album", params={"dir": dir_id}).json()
        self.assertEqual([img["name"] for img in page["images"]], ["x.png"])
        image = self.client.get("/api/v1/album/image/sub/x.png", params={"dir": dir_id})
        self.assertEqual(image.status_code, 200)

        renamed = self.client.put(f"/api/v1/album/dirs/{dir_id}", json={"name": "新名字"})
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["name"], "新名字")

        # 内置目录不能改名/删除
        self.assertEqual(
            self.client.put("/api/v1/album/dirs/output", json={"name": "x"}).status_code,
            400,
        )
        self.assertEqual(self.client.delete("/api/v1/album/dirs/output").status_code, 400)

        # 删除只移出列表,磁盘文件保留;之后的访问 404
        self.assertEqual(self.client.delete(f"/api/v1/album/dirs/{dir_id}").status_code, 200)
        self.assertTrue((extra / "sub" / "x.png").is_file())
        self.assertEqual(
            self.client.get("/api/v1/album", params={"dir": dir_id}).status_code, 404
        )

    def test_album_dirs_validation(self):
        empty_name = self.client.post(
            "/api/v1/album/dirs", json={"name": " ", "path": "x"}
        )
        self.assertEqual(empty_name.status_code, 422)
        missing = self.client.post(
            "/api/v1/album/dirs",
            json={"name": "x", "path": str(self.core.root / "nope")},
        )
        self.assertEqual(missing.status_code, 422)
        unknown = self.client.put("/api/v1/album/dirs/ffffffff", json={"name": "x"})
        self.assertEqual(unknown.status_code, 404)


if __name__ == "__main__":
    unittest.main()


class AlbumVideoTests(AlbumTests):
    def _mp4(self, relpath: str, mtime_ns: int):
        path = self.core.config.output_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-mp4")
        os.utime(path, ns=(mtime_ns, mtime_ns))
        return path

    def test_mp4_listed_with_video_kind(self):
        self._png("2026-07-27/zit-00001.png", 1000)
        self._mp4("2026-07-27/wan2.2-ti2v-5b-00001.mp4", 2000)

        response = self.client.get("/api/v1/album")
        self.assertEqual(response.status_code, 200)
        images = response.json()["images"]
        self.assertEqual([img["kind"] for img in images], ["video", "image"])
        video = images[0]
        self.assertEqual(video["name"], "wan2.2-ti2v-5b-00001.mp4")
        # mp4 不做 PIL 尺寸解析,宽高记 0
        self.assertEqual(video["width"], 0)
        self.assertEqual(video["height"], 0)
        self.assertGreater(video["size_bytes"], 0)

    def test_mp4_download_uses_video_media_type(self):
        self._mp4("2026-07-27/wan2.2-ti2v-5b-00001.mp4", 1000)
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/wan2.2-ti2v-5b-00001.mp4"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.content, b"fake-mp4")

    def test_mp4_metadata_returns_404(self):
        self._mp4("2026-07-27/wan2.2-ti2v-5b-00001.mp4", 1000)
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/wan2.2-ti2v-5b-00001.mp4/metadata"
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("视频", response.json()["detail"])

    def test_mp4_delete(self):
        path = self._mp4("2026-07-27/wan2.2-ti2v-5b-00001.mp4", 1000)
        response = self.client.delete(
            "/api/v1/album/image/2026-07-27/wan2.2-ti2v-5b-00001.mp4"
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(path.exists())
        self.assertEqual(len(self.client.get("/api/v1/album").json()["images"]), 0)


class AlbumComfyuiMetadataTests(AlbumTests):
    def _comfyui_png(self, relpath: str, mtime_ns: int, graph: dict):
        import json

        from PIL import PngImagePlugin

        path = self.core.config.output_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("prompt", json.dumps(graph, ensure_ascii=False))
        Image.new("RGB", (16, 8)).save(path, pnginfo=png_info)
        os.utime(path, ns=(mtime_ns, mtime_ns))
        return path

    def _graph(self):
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "krea2\\Krea2-8Steps-fp8.safetensors"},
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "qwen_2.5_vl_7b.safetensors"},
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "flux-vae.safetensors"},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "一只猫", "clip": ["2", 0]},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "模糊", "clip": ["2", 0]},
            },
            "6": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["5", 0]},
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["4", 0],
                    "negative": ["6", 0],
                    "denoise": 1.0,
                },
            },
        }

    def test_comfyui_png_metadata(self):
        self._comfyui_png("2026-07-27/comfy_00001.png", 1000, self._graph())

        response = self.client.get(
            "/api/v1/album/image/2026-07-27/comfy_00001.png/metadata"
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "comfyui")
        self.assertEqual(body["mode"], "krea2")
        self.assertEqual(body["model_name"], "Krea2-8Steps-fp8.safetensors")
        self.assertEqual(body["vae_name"], "flux-vae.safetensors")
        self.assertEqual(body["text_encoder_name"], "qwen_2.5_vl_7b.safetensors")
        self.assertEqual(body["steps"], 8)
        self.assertEqual(body["cfg"], 1.0)
        self.assertEqual(body["sampler"], "euler")
        self.assertEqual(body["scheduler"], "simple")
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["prompt"], "一只猫")
        # negative 经 ConditioningZeroOut 透传到 CLIPTextEncode
        self.assertEqual(body["negative_prompt"], "模糊")
        self.assertEqual(body["width"], 16)
        self.assertEqual(body["height"], 8)

    def test_comfyui_first_sampler_wins(self):
        graph = self._graph()
        graph["8"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "noise_seed": 7,
                "steps": 30,
                "cfg": 4.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "positive": ["4", 0],
                "negative": ["5", 0],
            },
        }
        self._comfyui_png("2026-07-27/comfy_00002.png", 1000, graph)

        body = self.client.get(
            "/api/v1/album/image/2026-07-27/comfy_00002.png/metadata"
        ).json()
        # 多轮采样只取文档序第一个 KSampler 的参数
        self.assertEqual(body["steps"], 8)
        self.assertEqual(body["sampler"], "euler")

    def test_comfyui_advanced_sampler_seed(self):
        graph = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "ZImageTurbo.safetensors"},
            },
            "2": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "noise_seed": 99,
                    "steps": 20,
                    "cfg": 5.0,
                    "sampler_name": "uni_pc",
                    "scheduler": "simple",
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                },
            },
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "正"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ["9", 0]}},
        }
        self._comfyui_png("2026-07-27/comfy_00003.png", 1000, graph)

        body = self.client.get(
            "/api/v1/album/image/2026-07-27/comfy_00003.png/metadata"
        ).json()
        self.assertEqual(body["source"], "comfyui")
        self.assertEqual(body["mode"], "zit")
        self.assertEqual(body["seed"], 99)
        self.assertEqual(body["prompt"], "正")
        # text 是链接而非字面量时取不到
        self.assertIsNone(body["negative_prompt"])
        self.assertIsNone(body["vae_name"])

    def test_plain_png_still_404(self):
        self._png("2026-07-27/zit-00001.png", 1000)
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/zit-00001.png/metadata"
        )
        self.assertEqual(response.status_code, 404)


class AlbumSubdirTests(AlbumTests):
    def test_subdirs_lists_first_level_only(self):
        self._png("2026-07-27/a.png", 1000)
        self._png("2026-07-28/deep/b.png", 2000)
        (self.core.config.output_dir / "empty-dir").mkdir(parents=True)

        response = self.client.get("/api/v1/album/subdirs")
        self.assertEqual(response.status_code, 200)
        # 只列第一级子目录(不含 deep),按名称倒序;空目录也列出
        self.assertEqual(
            response.json()["subdirs"], ["empty-dir", "2026-07-28", "2026-07-27"]
        )

    def test_album_subdir_filters_recursively(self):
        self._png("2026-07-27/x.png", 1000)
        self._png("2026-07-27/deep/y.png", 2000)
        self._png("2026-07-28/z.png", 3000)

        response = self.client.get("/api/v1/album", params={"subdir": "2026-07-27"})
        self.assertEqual(response.status_code, 200)
        # 子目录内为深度查找,包含更深层级的图片
        self.assertEqual(
            [img["name"] for img in response.json()["images"]], ["y.png", "x.png"]
        )

    def test_album_subdir_validation(self):
        self._png("2026-07-27/x.png", 1000)
        for bad in ("..", "a/b", "../etc"):
            response = self.client.get("/api/v1/album", params={"subdir": bad})
            self.assertEqual(response.status_code, 422, bad)


class AlbumPosterTests(AlbumTests):
    def _real_mp4(self, relpath: str):
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("ffmpeg 不可用")
        path = self.core.config.output_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:size=16x16:duration=1",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        return path

    def test_video_poster_jpeg_and_cache(self):
        self._real_mp4("2026-07-27/wan-00001.mp4")

        response = self.client.get(
            "/api/v1/album/image/2026-07-27/wan-00001.mp4/poster"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        # JPEG 以 FF D8 开头
        self.assertTrue(response.content[:2] == b"\xff\xd8")

        cached = self.client.get(
            "/api/v1/album/image/2026-07-27/wan-00001.mp4/poster"
        )
        self.assertEqual(cached.status_code, 200)
        self.assertEqual(cached.content, response.content)

    def test_poster_rejects_non_video(self):
        self._png("2026-07-27/zit-00001.png", 1000)
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/zit-00001.png/poster"
        )
        self.assertEqual(response.status_code, 404)

    def test_poster_missing_video_404(self):
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/nope.mp4/poster"
        )
        self.assertEqual(response.status_code, 404)


class AlbumThumbnailTests(AlbumTests):
    """相册图片缩略图:首访懒生成,JPEG 200×200,按 (mtime, size) 缓存。"""

    def _png_of_size(self, relpath: str, mtime_ns: int, size, color=(255, 0, 0)):
        """生成已知尺寸的纯色图,方便校验缩放结果。不覆盖父类 _png 以免影响继承测试。"""
        from PIL import Image

        path = self.core.config.output_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        os.utime(path, ns=(mtime_ns, mtime_ns))
        return path

    def test_thumbnail_returns_jpeg_within_size_budget(self):
        # 800x600 源图,按面积等比缩放:scale = √(160000/480000) = 0.57735
        # → 461×346 = 159_506 像素(int 截断后略低于 160_000)
        self._png_of_size("2026-07-27/big.png", 1000, size=(800, 600))
        response = self.client.get("/api/v1/album/image/2026-07-27/big.png/thumbnail")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertTrue(response.content[:2] == b"\xff\xd8")

        import io

        from PIL import Image

        with Image.open(io.BytesIO(response.content)) as thumb:
            width, height = thumb.size
        self.assertLessEqual(width * height, 160_000)
        self.assertGreater(width * height, 159_000)
        self.assertEqual((width, height), (461, 346))
        self.assertAlmostEqual(width / height, 800 / 600, delta=0.01)

    def test_thumbnail_square_source_hits_pixel_budget(self):
        # 正方形源图 1024x1024:scale = √(160000/1048576) = 0.39062 → 400×400 = 160_000
        self._png_of_size("2026-07-27/square.png", 1000, size=(1024, 1024))
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/square.png/thumbnail"
        )
        self.assertEqual(response.status_code, 200)
        import io

        from PIL import Image

        with Image.open(io.BytesIO(response.content)) as thumb:
            width, height = thumb.size
        self.assertEqual((width, height), (400, 400))
        self.assertEqual(width * height, 160_000)
        self.assertLessEqual(width * height, 160_000)

    def test_thumbnail_landscape_16_9_fills_pixel_budget(self):
        # 16:9 横图按面积缩放:scale = √(160000/2073600) = 0.27787
        # → 533×300 = 159_900 像素
        self._png_of_size("2026-07-27/wide.png", 1000, size=(1920, 1080))
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/wide.png/thumbnail"
        )
        self.assertEqual(response.status_code, 200)
        import io

        from PIL import Image

        with Image.open(io.BytesIO(response.content)) as thumb:
            width, height = thumb.size
        self.assertEqual((width, height), (533, 300))
        self.assertEqual(width * height, 159_900)
        self.assertLessEqual(width * height, 160_000)
        self.assertGreater(width * height, 159_000)
        self.assertAlmostEqual(width / height, 1920 / 1080, delta=0.01)

    def test_thumbnail_caches_by_mtime_size(self):
        self._png_of_size("2026-07-27/cached.png", 1000, size=(400, 300))
        first = self.client.get("/api/v1/album/image/2026-07-27/cached.png/thumbnail")
        self.assertEqual(first.status_code, 200)

        # 同 mtime/size 二次请求:内容一致(命中缓存,服务端不再走 PIL)
        second = self.client.get("/api/v1/album/image/2026-07-27/cached.png/thumbnail")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.content, first.content)

        # 修改文件(mtime_ns 变化)→ 缓存键变化 → 重新生成,内容可能不同
        self._png_of_size(
            "2026-07-27/cached.png", 2000, size=(400, 300), color=(0, 0, 255)
        )
        third = self.client.get("/api/v1/album/image/2026-07-27/cached.png/thumbnail")
        self.assertEqual(third.status_code, 200)
        self.assertNotEqual(third.content, first.content)

    def test_thumbnail_rejects_video(self):
        # mp4 是当前唯一的视频格式,缩略图端点不处理视频(走 poster 端点)
        # 内联写最小 mp4 字节(不依赖 AlbumVideoTests._mp4,类层级不在继承链上)
        path = self.core.config.output_dir / "2026-07-27" / "wan-00001.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-mp4")
        os.utime(path, ns=(1000, 1000))
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/wan-00001.mp4/thumbnail"
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("缩略图", response.json()["detail"])

    def test_thumbnail_missing_file_404(self):
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/nope.png/thumbnail"
        )
        self.assertEqual(response.status_code, 404)

    def test_thumbnail_rejects_path_traversal(self):
        # 与原图读取一致,越界路径 400/404/422 都接受
        response = self.client.get(
            "/api/v1/album/image/..%2F..%2Fsecret.png/thumbnail"
        )
        self.assertIn(response.status_code, (400, 404, 422))

    def test_thumbnail_keeps_aspect_for_portrait(self):
        # 竖图 600x800:scale = √(160000/480000) = 0.57735 → 346×461 = 159_506
        self._png_of_size("2026-07-27/portrait.png", 1000, size=(600, 800))
        response = self.client.get(
            "/api/v1/album/image/2026-07-27/portrait.png/thumbnail"
        )
        self.assertEqual(response.status_code, 200)
        import io

        from PIL import Image

        with Image.open(io.BytesIO(response.content)) as thumb:
            width, height = thumb.size
        self.assertEqual((width, height), (346, 461))
        self.assertEqual(width * height, 159_506)
        self.assertLessEqual(width * height, 160_000)
        self.assertGreater(width * height, 159_000)
