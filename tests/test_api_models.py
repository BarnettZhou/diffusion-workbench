import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from diffusion_workbench_api.models import scan_safetensors_quant
from test_api_jobs import ApiTestCase


class ScanSafetensorsQuantTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_checkpoint(self, name: str, tensors: dict) -> Path:
        path = self.root / name
        save_file(tensors, str(path))
        return path

    def test_detects_scaled_fp8(self):
        path = self.write_checkpoint(
            "m.safetensors",
            {
                "model.diffusion_model.blocks.0.attn.wq.weight": torch.zeros(
                    (2, 2), dtype=torch.float8_e4m3fn
                ),
                "model.diffusion_model.blocks.0.attn.wq.weight_scale": torch.tensor(
                    1.0, dtype=torch.float32
                ),
                "model.diffusion_model.blocks.0.attn.wq.comfy_quant": torch.tensor(
                    list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8
                ),
                "model.diffusion_model.norm": torch.ones(1, dtype=torch.bfloat16),
            },
        )
        self.assertEqual(scan_safetensors_quant(path), "fp8")
        # 文件没有变化时走缓存,结果一致
        self.assertEqual(scan_safetensors_quant(path), "fp8")

    def test_detects_int8(self):
        path = self.write_checkpoint(
            "m.safetensors",
            {"w": torch.zeros((2, 2), dtype=torch.int8)},
        )
        self.assertEqual(scan_safetensors_quant(path), "int8")

    def test_detects_mixed_fp8_int8(self):
        path = self.write_checkpoint(
            "m.safetensors",
            {
                "a": torch.zeros((1, 1), dtype=torch.float8_e4m3fn),
                "b": torch.zeros((1, 1), dtype=torch.int8),
            },
        )
        self.assertEqual(scan_safetensors_quant(path), "fp8+int8")

    def test_detects_bf16_without_quant(self):
        path = self.write_checkpoint(
            "m.safetensors", {"w": torch.ones(1, dtype=torch.bfloat16)}
        )
        self.assertEqual(scan_safetensors_quant(path), "bf16")

    def test_non_safetensors_and_missing_file_return_none(self):
        self.assertIsNone(scan_safetensors_quant(self.root / "m.ckpt"))
        self.assertIsNone(scan_safetensors_quant(self.root / "missing.safetensors"))


class FetchModelQuantTests(ApiTestCase):
    def write_checkpoint(self, name: str, tensors: dict) -> Path:
        path = self.core.root / name
        save_file(tensors, str(path))
        return path

    def test_fetch_scans_and_persists_quant(self):
        self.write_checkpoint(
            "zit.safetensors",
            {"w": torch.zeros((1, 1), dtype=torch.float8_e4m3fn)},
        )

        before = self.client.get("/api/v1/models/zit").json()["models"][0]
        self.assertIsNone(before["quant"])

        response = self.client.post("/api/v1/models/zit/zit.safetensors/quant")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["quant"], "fp8")

        after = self.client.get("/api/v1/models/zit").json()["models"][0]
        self.assertEqual(after["quant"], "fp8")

    def test_fetch_unreadable_file_returns_422(self):
        # zit2.safetensors 只存在于资源目录,磁盘上没有文件
        response = self.client.post("/api/v1/models/zit/zit2.safetensors/quant")
        self.assertEqual(response.status_code, 422)

    def test_fetch_unknown_model_returns_404(self):
        response = self.client.post("/api/v1/models/zit/nope.safetensors/quant")
        self.assertEqual(response.status_code, 404)


class ModelListTests(ApiTestCase):
    def test_lists_models_with_info(self):
        response = self.client.get("/api/v1/models/zit")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertEqual(len(models), 2)
        first = models[0]
        self.assertEqual(first["name"], "zit.safetensors")
        self.assertEqual(first["mode"], "zit")
        self.assertIsNone(first["alias"])
        self.assertIsNone(first["quant"])
        self.assertFalse(first["has_cover"])
        self.assertIsNone(first["cover_url"])

    def test_unknown_mode_returns_422(self):
        self.assertEqual(self.client.get("/api/v1/models/sd15").status_code, 422)

    def test_zib_models_listed(self):
        response = self.client.get("/api/v1/models/zib")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "zib.safetensors")
        self.assertEqual(models[0]["mode"], "zib")


class ZibModelRoutesTests(ApiTestCase):
    def test_zib_info_and_cover_routes(self):
        info = self.client.put(
            "/api/v1/models/zib/zib.safetensors/info",
            json={"alias": "base", "note": "ZIB 模型"},
        )
        self.assertEqual(info.status_code, 200)

        data = b"\x89PNG\r\n\x1a\nfakecover"
        upload = self.client.put(
            "/api/v1/models/zib/zib.safetensors/cover",
            content=data,
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(upload.status_code, 200)
        download = self.client.get("/api/v1/models/zib/zib.safetensors/cover")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.content, data)

        models = self.client.get("/api/v1/models/zib").json()["models"]
        self.assertEqual(models[0]["alias"], "base")
        self.assertEqual(models[0]["note"], "ZIB 模型")
        self.assertTrue(models[0]["has_cover"])

    def test_unknown_mode_info_and_cover_return_422(self):
        self.assertEqual(
            self.client.put(
                "/api/v1/models/sd15/x.safetensors/info", json={"note": "x"}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.put(
                "/api/v1/models/sd15/x.safetensors/cover",
                content=b"png",
                headers={"Content-Type": "image/png"},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.get("/api/v1/models/sd15/x.safetensors/cover").status_code, 422
        )


class ModelInfoTests(ApiTestCase):
    def test_update_alias_and_note(self):
        response = self.client.put(
            "/api/v1/models/zit/zit.safetensors/info",
            json={"alias": "人像", "note": "常用模型"},
        )
        self.assertEqual(response.status_code, 200)

        models = self.client.get("/api/v1/models/zit").json()["models"]
        first = models[0]
        self.assertEqual(first["alias"], "人像")
        self.assertEqual(first["note"], "常用模型")

    def test_update_unknown_model_returns_404(self):
        response = self.client.put(
            "/api/v1/models/zit/nope.safetensors/info", json={"note": "x"}
        )
        self.assertEqual(response.status_code, 404)


class ModelCoverTests(ApiTestCase):
    def test_upload_and_download_cover(self):
        data = b"\x89PNG\r\n\x1a\nfakecover"
        upload = self.client.put(
            "/api/v1/models/zit/zit.safetensors/cover",
            content=data,
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(upload.status_code, 200)

        download = self.client.get("/api/v1/models/zit/zit.safetensors/cover")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.content, data)
        self.assertEqual(download.headers["content-type"], "image/png")

        models = self.client.get("/api/v1/models/zit").json()["models"]
        self.assertTrue(models[0]["has_cover"])
        self.assertIsNotNone(models[0]["cover_url"])

    def test_cover_replaced_on_new_upload(self):
        self.client.put(
            "/api/v1/models/zit/zit.safetensors/cover",
            content=b"png-data",
            headers={"Content-Type": "image/png"},
        )
        self.client.put(
            "/api/v1/models/zit/zit.safetensors/cover",
            content=b"jpeg-data",
            headers={"Content-Type": "image/jpeg"},
        )
        download = self.client.get("/api/v1/models/zit/zit.safetensors/cover")
        self.assertEqual(download.content, b"jpeg-data")
        self.assertEqual(download.headers["content-type"], "image/jpeg")

    def test_unsupported_type_returns_415(self):
        response = self.client.put(
            "/api/v1/models/zit/zit.safetensors/cover",
            content=b"gif",
            headers={"Content-Type": "image/gif"},
        )
        self.assertEqual(response.status_code, 415)

    def test_missing_cover_returns_404(self):
        response = self.client.get("/api/v1/models/zit/zit.safetensors/cover")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
