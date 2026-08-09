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
