import io
import unittest

import httpx
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
    """API 反推路由:monkeypatch call_remote_caption,不触网。"""

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

        self._orig_call = caption_module.call_remote_caption
        caption_module.call_remote_caption = fake_call
        self.addCleanup(setattr, caption_module, "call_remote_caption", self._orig_call)
        # 配置好 caption_api 设置项
        response = self.client.put(
            "/api/v1/settings",
            json={
                "caption_api": {
                    "endpoints": [
                        {
                            "id": "cc33dd44",
                            "name": "视觉",
                            "interface": "ollama",
                            "base_url": "http://127.0.0.1:11434",
                            "api_key": "",
                            "models": ["qwen3-vl:8b"],
                        }
                    ],
                    "selected": {
                        "endpoint_id": "cc33dd44",
                        "model": "qwen3-vl:8b",
                    },
                }
            },
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
        # 设置项是整项替换存储,局部提交需带上端点和选择
        self.client.put(
            "/api/v1/settings",
            json={
                "caption_api": {
                    "endpoints": [
                        {
                            "id": "cc33dd44",
                            "name": "视觉",
                            "interface": "ollama",
                            "base_url": "http://127.0.0.1:11434",
                            "api_key": "",
                            "models": ["qwen3-vl:8b"],
                        }
                    ],
                    "selected": {
                        "endpoint_id": "cc33dd44",
                        "model": "qwen3-vl:8b",
                    },
                    "prompt": "CUSTOM_PROMPT",
                }
            },
        )
        image_id = self._upload_image()
        response = self.client.post(
            "/api/v1/caption/remote", json={"image_id": image_id}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.captured["prompt"].startswith("CUSTOM_PROMPT"))

    def test_remote_caption_not_configured_returns_400(self):
        self.client.put(
            "/api/v1/settings",
            json={"caption_api": {"endpoints": [], "selected": {"endpoint_id": "", "model": ""}}},
        )
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

    def test_remote_models_endpoint_success_and_failure(self):
        from diffusion_workbench_api import llm as llm_module

        async def fake_fetch(**kwargs):
            self.captured = kwargs
            return ["qwen3:8b", "llava:latest"]

        original = llm_module.fetch_remote_models
        llm_module.fetch_remote_models = fake_fetch
        self.addCleanup(setattr, llm_module, "fetch_remote_models", original)
        response = self.client.post(
            "/api/v1/remote/models",
            json={"interface": "ollama", "base_url": " http://127.0.0.1:11434 "},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"], ["qwen3:8b", "llava:latest"])
        self.assertEqual(self.captured["base_url"], "http://127.0.0.1:11434")

        async def failing_fetch(**kwargs):
            raise httpx.ConnectError("connection refused")

        llm_module.fetch_remote_models = failing_fetch
        response = self.client.post(
            "/api/v1/remote/models",
            json={"base_url": "http://127.0.0.1:1"},
        )
        self.assertEqual(response.status_code, 502)
        self.assertIn("无法连接到大模型服务", response.json()["detail"])

    def test_fetch_remote_models_rejects_unexpected_shape(self):
        # LM Studio 对未知路径也回 200 + {"error": ...}(如 OpenAI 接口漏填 /v1),
        # 必须报错提示,而不是静默返回空列表
        import asyncio

        from diffusion_workbench_api import llm as llm_module

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"error": "Unexpected endpoint or method. (GET /models)"}

        class _FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, headers=None):
                return _FakeResponse()

        original_client = llm_module.httpx.AsyncClient
        llm_module.httpx.AsyncClient = _FakeClient
        self.addCleanup(setattr, llm_module.httpx, "AsyncClient", original_client)
        with self.assertRaisesRegex(ValueError, "响应格式不符合预期"):
            asyncio.run(
                llm_module.fetch_remote_models(
                    interface="openai", base_url="http://192.168.31.157:1234"
                )
            )

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
