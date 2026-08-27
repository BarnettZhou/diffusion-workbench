import io
import unittest

from PIL import Image

from test_api_jobs import ApiTestCase


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 64, 32)).save(buffer, format="PNG")
    return buffer.getvalue()


class CaptionTests(ApiTestCase):
    def _upload_image(self) -> str:
        # 反推复用编辑输入图上传通道,返回受控 id
        response = self.client.post(
            "/api/v1/edit/input-images",
            content=_png_bytes(),
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def test_caption_returns_text_and_records_call(self):
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption",
            json={"image_id": image_id, "hint": "focus on lighting", "max_length": 256},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["caption"], "fake caption")
        self.assertEqual(body["load_seconds"], 0.0)
        self.assertEqual(body["infer_seconds"], 0.0)
        # Core 收到的参数与请求一致
        call = self.core.caption_calls[-1]
        self.assertEqual(call["hint"], "focus on lighting")
        self.assertEqual(call["max_length"], 256)
        self.assertTrue(call["image_path"].is_file())

    def test_caption_uses_defaults(self):
        image_id = self._upload_image()
        response = self.client.post("/api/v1/caption", json={"image_id": image_id})
        self.assertEqual(response.status_code, 200)
        call = self.core.caption_calls[-1]
        self.assertEqual(call["hint"], "")
        self.assertEqual(call["max_length"], 2048)
        self.assertEqual(call["seed"], -1)

    def test_unknown_image_id_returns_404(self):
        response = self.client.post(
            "/api/v1/caption", json={"image_id": "0" * 32 + ".png"}
        )
        self.assertEqual(response.status_code, 404)

    def test_malformed_image_id_returns_404(self):
        response = self.client.post(
            "/api/v1/caption", json={"image_id": "../../jobs.sqlite3"}
        )
        self.assertEqual(response.status_code, 404)

    def test_max_length_out_of_range_returns_422(self):
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption", json={"image_id": image_id, "max_length": 8}
        )
        self.assertEqual(response.status_code, 422)

    def test_hint_too_long_returns_422(self):
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption", json={"image_id": image_id, "hint": "x" * 2001}
        )
        self.assertEqual(response.status_code, 422)


class RemoteCaptionTests(ApiTestCase):
    """API 反推路由:monkeypatch call_remote_caption / fetch_remote_models,不触网。"""

    def setUp(self):
        super().setUp()
        import diffusion_workbench_api.caption as caption_module

        self._module = caption_module
        self.captured = None

        async def fake_call(**kwargs):
            self.captured = kwargs
            return "remote caption", {
                "input_tokens": 11,
                "output_tokens": 7,
                "cached_tokens": None,
            }

        async def fake_fetch(**kwargs):
            self.captured = kwargs
            return ["qwen3-vl:8b", "llava:latest"]

        self._orig_call = caption_module.call_remote_caption
        self._orig_fetch = caption_module.fetch_remote_models
        caption_module.call_remote_caption = fake_call
        caption_module.fetch_remote_models = fake_fetch
        self.addCleanup(setattr, caption_module, "call_remote_caption", self._orig_call)
        self.addCleanup(setattr, caption_module, "fetch_remote_models", self._orig_fetch)
        # 配置好 caption_api 设置项
        response = self.client.put(
            "/api/v1/settings",
            json={"caption_api": {"base_url": "http://127.0.0.1:11434", "model": "qwen3-vl:8b"}},
        )
        self.assertEqual(response.status_code, 200)

    def _upload_image(self) -> str:
        response = self.client.post(
            "/api/v1/edit/input-images",
            content=_png_bytes(),
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def test_remote_caption_returns_text(self):
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption/remote",
            json={"image_id": image_id, "hint": "focus on lighting", "max_length": 256},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["caption"], "remote caption")
        self.assertEqual(body["load_seconds"], 0.0)
        self.assertGreaterEqual(body["infer_seconds"], 0.0)
        # 远端调用收到的参数与请求/设置一致
        self.assertEqual(self.captured["model"], "qwen3-vl:8b")
        self.assertEqual(self.captured["max_length"], 256)
        self.assertEqual(self.captured["think"], False)
        self.assertTrue(self.captured["image_b64"])
        # 提示词 = 设置项 prompt(默认与本地反推一致) + hint 追加
        self.assertIn("image prompt engineer", self.captured["prompt"])
        self.assertIn("bilingual", self.captured["prompt"])
        self.assertIn("Additional user instructions: focus on lighting", self.captured["prompt"])

    def test_remote_caption_uses_custom_prompt_from_settings(self):
        # 设置项是整项替换存储,局部提交需带上 model
        self.client.put(
            "/api/v1/settings",
            json={"caption_api": {"model": "qwen3-vl:8b", "prompt": "CUSTOM_PROMPT"}},
        )
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption/remote", json={"image_id": image_id}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.captured["prompt"].startswith("CUSTOM_PROMPT"))

    def test_remote_caption_not_configured_returns_400(self):
        self.client.put("/api/v1/settings", json={"caption_api": {"model": ""}})
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption/remote", json={"image_id": image_id}
        )
        self.assertEqual(response.status_code, 400)

    def test_remote_caption_unknown_image_id_returns_404(self):
        response = self.client.post(
            "/api/v1/caption/remote", json={"image_id": "0" * 32 + ".png"}
        )
        self.assertEqual(response.status_code, 404)

    def test_remote_caption_upstream_error_returns_502(self):
        import httpx

        async def failing_call(**kwargs):
            raise httpx.ConnectError("connection refused")

        self._module.call_remote_caption = failing_call
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption/remote", json={"image_id": image_id}
        )
        self.assertEqual(response.status_code, 502)
        self.assertIn("无法连接到反推 API 服务", response.json()["detail"])

    def test_remote_test_endpoint_success(self):
        response = self.client.post(
            "/api/v1/caption/remote/test",
            json={"base_url": "http://127.0.0.1:11434", "model": "qwen3-vl:8b"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIs(response.json()["ok"], True)

    def test_remote_test_endpoint_ollama_latest_suffix(self):
        # Ollama 允许省略 :latest 后缀
        response = self.client.post(
            "/api/v1/caption/remote/test",
            json={"base_url": "http://127.0.0.1:11434", "model": "llava"},
        )
        self.assertEqual(response.status_code, 200)

    def test_remote_test_endpoint_model_not_found(self):
        response = self.client.post(
            "/api/v1/caption/remote/test",
            json={"base_url": "http://127.0.0.1:11434", "model": "nope"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("不存在", response.json()["detail"])

    def test_remote_test_endpoint_falls_back_to_saved_settings(self):
        response = self.client.post("/api/v1/caption/remote/test", json={})
        self.assertEqual(response.status_code, 200)
        # 留空时回退到已保存设置里的 base_url(model 同理,模型命中说明已回退)
        self.assertEqual(self.captured["base_url"], "http://127.0.0.1:11434")

    def test_remote_test_endpoint_not_configured_returns_400(self):
        self.client.put("/api/v1/settings", json={"caption_api": {"model": ""}})
        response = self.client.post("/api/v1/caption/remote/test", json={})
        self.assertEqual(response.status_code, 400)


    def test_remote_caption_writes_request_record(self):
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption/remote",
            json={"image_id": image_id, "hint": "focus on lighting"},
        )
        self.assertEqual(response.status_code, 200)
        records = self.client.get("/api/v1/caption/remote/requests").json()
        self.assertEqual(records["total"], 1)
        item = records["items"][0]
        self.assertEqual(item["model"], "qwen3-vl:8b")
        self.assertEqual(item["response"], "remote caption")
        self.assertEqual(item["error"], "")
        self.assertEqual(item["input_tokens"], 11)
        self.assertEqual(item["output_tokens"], 7)
        # 请求内容含提示词但不落 base64 图片本体
        self.assertIn("focus on lighting", item["request"])
        self.assertNotIn("iVBOR", item["request"])

    def test_remote_caption_error_is_recorded(self):
        import httpx

        async def failing_call(**kwargs):
            raise httpx.ConnectError("connection refused")

        self._module.call_remote_caption = failing_call
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption/remote", json={"image_id": image_id}
        )
        self.assertEqual(response.status_code, 502)
        records = self.client.get("/api/v1/caption/remote/requests").json()
        self.assertEqual(records["total"], 1)
        item = records["items"][0]
        self.assertEqual(item["response"], "")
        self.assertIn("无法连接到反推 API 服务", item["error"])

    def test_remote_records_pagination(self):
        image_id = self._upload_image()
        for _ in range(3):
            self.client.post("/api/v1/caption/remote", json={"image_id": image_id})
        page1 = self.client.get(
            "/api/v1/caption/remote/requests?page=1&page_size=2"
        ).json()
        self.assertEqual(page1["total"], 3)
        self.assertEqual(len(page1["items"]), 2)
        page2 = self.client.get(
            "/api/v1/caption/remote/requests?page=2&page_size=2"
        ).json()
        self.assertEqual(len(page2["items"]), 1)


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class CallRemoteCaptionPayloadTests(unittest.IsolatedAsyncioTestCase):
    """直接验证 call_remote_caption 构造的请求体(图片与 think 的下发逻辑)。"""

    async def _capture(self, *, interface, response_data):
        import diffusion_workbench_api.caption as caption_module

        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return _FakeResponse(response_data)

        orig = caption_module.httpx.AsyncClient
        caption_module.httpx.AsyncClient = FakeClient
        self.addCleanup(setattr, caption_module.httpx, "AsyncClient", orig)
        result = await caption_module.call_remote_caption(
            interface=interface,
            base_url="http://127.0.0.1:1",
            api_key="sk-test",
            model="m",
            prompt="describe",
            image_b64="aGVsbG8=",
            mime="image/png",
            think=False,
            max_length=256,
            seed=42,
        )
        return result, captured

    async def test_ollama_payload(self):
        result, captured = await self._capture(
            interface="ollama", response_data={"message": {"content": "x"}}
        )
        self.assertEqual(result, ("x", {"input_tokens": None, "output_tokens": None, "cached_tokens": None}))
        self.assertTrue(captured["url"].endswith("/api/chat"))
        payload = captured["json"]
        message = payload["messages"][0]
        # Ollama 图片走 message.images 的 base64 列表
        self.assertEqual(message["images"], ["aGVsbG8="])
        self.assertEqual(message["content"], "describe")
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"], {"num_predict": 256, "seed": 42})

    async def test_openai_payload(self):
        result, captured = await self._capture(
            interface="openai",
            response_data={"choices": [{"message": {"content": "x"}}]},
        )
        self.assertEqual(result[0], "x")
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["headers"], {"Authorization": "Bearer sk-test"})
        payload = captured["json"]
        content = payload["messages"][0]["content"]
        # OpenAI 兼容:图片走 data URL 的 image_url 块
        self.assertEqual(
            content[0]["image_url"]["url"], "data:image/png;base64,aGVsbG8="
        )
        self.assertEqual(content[1], {"type": "text", "text": "describe"})
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["seed"], 42)
        # think=False 时下发 enable_thinking=false
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )


if __name__ == "__main__":
    unittest.main()
