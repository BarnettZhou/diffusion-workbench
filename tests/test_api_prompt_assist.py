import unittest

import httpx

from diffusion_workbench_api import llm as llm_module
from test_api_jobs import ApiTestCase


class PromptAssistTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.client.put(
            "/api/v1/settings",
            json={
                "llm": {
                    "interface": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "model": "qwen3",
                }
            },
        )
        self.captured = None
        self._orig_call = llm_module.call_llm_chat

        async def fake_call(**kwargs):
            self.captured = kwargs
            return self.fake_response, {
                "input_tokens": 11,
                "output_tokens": 7,
                "cached_tokens": 3,
            }

        self.fake_response = "Positive: a cat\nNegative: blurry"
        llm_module.call_llm_chat = fake_call
        self.addCleanup(setattr, llm_module, "call_llm_chat", self._orig_call)

    def test_parses_positive_and_negative(self):
        self.fake_response = (
            "Positive: a fluffy cat, studio light\nNegative: blurry, lowres"
        )
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "一只猫", "language": "zh"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["parsed"])
        self.assertEqual(body["positive"], "a fluffy cat, studio light")
        self.assertEqual(body["negative"], "blurry, lowres")
        self.assertEqual(body["raw"], self.fake_response)

    def test_parses_markdown_noise_and_case_insensitive_markers(self):
        self.fake_response = (
            "## POSITIVE:  a dog in the park\n"
            "- negative: text, watermark"
        )
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "a dog", "language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["parsed"])
        self.assertEqual(body["positive"], "a dog in the park")
        self.assertEqual(body["negative"], "text, watermark")

    def test_unparseable_returns_parsed_false_with_raw(self):
        self.fake_response = "sorry, I cannot help with that"
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "test", "language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["parsed"])
        self.assertEqual(body["raw"], self.fake_response)
        self.assertEqual(body["positive"], "")
        self.assertEqual(body["negative"], "")

    def test_missing_model_returns_400(self):
        self.client.put("/api/v1/settings", json={"llm": {"model": ""}})
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "test", "language": "en"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("请先在设置中配置大模型", response.json()["detail"])

    def test_language_placeholder_replaced_in_system_prompt(self):
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "test", "language": "zh"},
        )
        self.assertEqual(response.status_code, 200)
        system_prompt = self.captured["messages"][0]["content"]
        self.assertIn("中文", system_prompt)
        self.assertNotIn("{language}", system_prompt)

        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "test", "language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        system_prompt = self.captured["messages"][0]["content"]
        self.assertIn("ENG", system_prompt)
        self.assertNotIn("{language}", system_prompt)

    def test_prompt_style_selects_matching_custom_prompt(self):
        self.client.put(
            "/api/v1/settings",
            json={
                "llm": {
                    "interface": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "model": "qwen3",
                    "system_prompt": "FLUX_CUSTOM",
                    "sd_system_prompt": "SD_CUSTOM",
                }
            },
        )

        # 旧客户端省略 prompt_style 时仍使用现有 FLUX 配置。
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "test", "language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        system_prompt = self.captured["messages"][0]["content"]
        self.assertIn("FLUX_CUSTOM", system_prompt)
        self.assertNotIn("SD_CUSTOM", system_prompt)

        response = self.client.post(
            "/api/v1/prompt-assist",
            json={
                "instruction": "test",
                "language": "en",
                "prompt_style": "sd",
            },
        )
        self.assertEqual(response.status_code, 200)
        system_prompt = self.captured["messages"][0]["content"]
        self.assertIn("SD_CUSTOM", system_prompt)
        self.assertNotIn("FLUX_CUSTOM", system_prompt)

    def test_invalid_prompt_style_returns_422(self):
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={
                "instruction": "test",
                "language": "en",
                "prompt_style": "unknown",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_llm_error_returns_502(self):
        async def failing_call(**kwargs):
            raise httpx.ConnectError("connection refused")

        llm_module.call_llm_chat = failing_call
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "test", "language": "en"},
        )
        self.assertEqual(response.status_code, 502)

    def test_blank_instruction_returns_422(self):
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "   ", "language": "en"},
        )
        self.assertEqual(response.status_code, 422)

    def test_think_setting_is_passed_through(self):
        self.client.put(
            "/api/v1/settings",
            json={"llm": {"model": "qwen3", "think": True, "think_effort": "high"}},
        )
        response = self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": "test", "language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIs(self.captured["think"], True)
        self.assertEqual(self.captured["think_effort"], "high")


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class CallLlmChatPayloadTests(unittest.IsolatedAsyncioTestCase):
    """直接验证 call_llm_chat 构造的请求体(think / think_effort 的下发逻辑)。"""

    async def _capture_payload(self, *, interface, think, think_effort, response_data):
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
                return _FakeResponse(response_data)

        orig = llm_module.httpx.AsyncClient
        llm_module.httpx.AsyncClient = FakeClient
        self.addCleanup(setattr, llm_module.httpx, "AsyncClient", orig)
        await llm_module.call_llm_chat(
            interface=interface,
            base_url="http://127.0.0.1:1",
            api_key="",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            think=think,
            think_effort=think_effort,
        )
        return captured["json"]

    async def test_ollama_think_disabled(self):
        payload = await self._capture_payload(
            interface="ollama",
            think=False,
            think_effort="",
            response_data={"message": {"content": "x"}},
        )
        self.assertIs(payload["think"], False)

    async def test_ollama_think_with_effort_string(self):
        payload = await self._capture_payload(
            interface="ollama",
            think=True,
            think_effort="high",
            response_data={"message": {"content": "x"}},
        )
        self.assertEqual(payload["think"], "high")

    async def test_openai_disable_thinking_uses_chat_template_kwargs(self):
        payload = await self._capture_payload(
            interface="openai",
            think=False,
            think_effort="",
            response_data={"choices": [{"message": {"content": "x"}}]},
        )
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertNotIn("reasoning_effort", payload)

    async def test_openai_effort_uses_reasoning_effort(self):
        payload = await self._capture_payload(
            interface="openai",
            think=True,
            think_effort="max",
            response_data={"choices": [{"message": {"content": "x"}}]},
        )
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("chat_template_kwargs", payload)

    async def test_openai_think_without_effort_sends_nothing(self):
        payload = await self._capture_payload(
            interface="openai",
            think=True,
            think_effort="",
            response_data={"choices": [{"message": {"content": "x"}}]},
        )
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("reasoning_effort", payload)


if __name__ == "__main__":
    unittest.main()
