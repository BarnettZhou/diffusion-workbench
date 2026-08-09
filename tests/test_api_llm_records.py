import unittest

import httpx

from diffusion_workbench_api import llm as llm_module
from test_api_jobs import ApiTestCase


class LLMRecordTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.client.put(
            "/api/v1/settings",
            json={
                "llm": {
                    "interface": "openai",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "qwen3",
                }
            },
        )
        self._orig_call = llm_module.call_llm_chat
        self.usage = {"input_tokens": 11, "output_tokens": 7, "cached_tokens": 3}

        async def fake_call(**kwargs):
            return "Positive: a cat\nNegative: blurry", self.usage

        llm_module.call_llm_chat = fake_call
        self.addCleanup(setattr, llm_module, "call_llm_chat", self._orig_call)

    def _assist(self, instruction="一只猫"):
        return self.client.post(
            "/api/v1/prompt-assist",
            json={"instruction": instruction, "language": "zh"},
        )

    def test_success_records_request_with_tokens_and_session(self):
        response = self._assist()
        self.assertEqual(response.status_code, 200)
        session_id = response.json()["session_id"]
        self.assertTrue(session_id)

        records = self.client.get("/api/v1/llm/requests").json()
        self.assertEqual(records["total"], 1)
        item = records["items"][0]
        self.assertEqual(item["session_id"], session_id)
        self.assertEqual(item["base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(item["model"], "qwen3")
        self.assertIn("一只猫", item["request"])
        self.assertEqual(item["response"], "Positive: a cat\nNegative: blurry")
        self.assertEqual(item["error"], "")
        self.assertEqual(item["input_tokens"], 11)
        self.assertEqual(item["output_tokens"], 7)
        self.assertEqual(item["cached_tokens"], 3)
        self.assertTrue(item["timestamp"])

    def test_each_request_gets_distinct_session(self):
        self._assist("a")
        self._assist("b")
        records = self.client.get("/api/v1/llm/requests").json()
        self.assertEqual(records["total"], 2)
        sessions = {item["session_id"] for item in records["items"]}
        self.assertEqual(len(sessions), 2)

    def test_error_is_recorded(self):
        async def failing_call(**kwargs):
            raise httpx.ConnectError("connection refused")

        llm_module.call_llm_chat = failing_call
        response = self._assist()
        self.assertEqual(response.status_code, 502)

        records = self.client.get("/api/v1/llm/requests").json()
        self.assertEqual(records["total"], 1)
        item = records["items"][0]
        self.assertIn("无法连接到大模型服务", item["error"])
        self.assertEqual(item["response"], "")
        self.assertIsNone(item["input_tokens"])

    def test_pagination(self):
        for i in range(12):
            self._assist(f"prompt {i}")
        page1 = self.client.get("/api/v1/llm/requests?page=1&page_size=10").json()
        page2 = self.client.get("/api/v1/llm/requests?page=2&page_size=10").json()
        self.assertEqual(page1["total"], 12)
        self.assertEqual(len(page1["items"]), 10)
        self.assertEqual(len(page2["items"]), 2)
        # 倒序:最新的排在最前
        self.assertIn("prompt 11", page1["items"][0]["request"])
        ids = [item["id"] for item in page1["items"] + page2["items"]]
        self.assertEqual(ids, sorted(ids, reverse=True))

    def test_page_out_of_range_returns_empty_items(self):
        self._assist()
        data = self.client.get("/api/v1/llm/requests?page=5").json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"], [])


class CallLlmChatUsageTests(unittest.IsolatedAsyncioTestCase):
    """直接验证 call_llm_chat 对 token 用量的提取。"""

    async def _call(self, *, interface, response_data):
        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                return _FakeResponse()

        orig = llm_module.httpx.AsyncClient
        llm_module.httpx.AsyncClient = FakeClient
        self.addCleanup(setattr, llm_module.httpx, "AsyncClient", orig)
        _, usage = await llm_module.call_llm_chat(
            interface=interface,
            base_url="http://127.0.0.1:1",
            api_key="",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            think=False,
        )
        return usage

    async def test_openai_usage_with_cached_tokens(self):
        usage = await self._call(
            interface="openai",
            response_data={
                "choices": [{"message": {"content": "x"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
            },
        )
        self.assertEqual(
            usage, {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 64}
        )

    async def test_openai_usage_deepseek_cache_field(self):
        usage = await self._call(
            interface="openai",
            response_data={
                "choices": [{"message": {"content": "x"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 50,
                },
            },
        )
        self.assertEqual(usage["cached_tokens"], 50)

    async def test_openai_usage_missing(self):
        usage = await self._call(
            interface="openai",
            response_data={"choices": [{"message": {"content": "x"}}]},
        )
        self.assertEqual(
            usage,
            {"input_tokens": None, "output_tokens": None, "cached_tokens": None},
        )

    async def test_ollama_usage(self):
        usage = await self._call(
            interface="ollama",
            response_data={
                "message": {"content": "x"},
                "prompt_eval_count": 30,
                "eval_count": 12,
            },
        )
        self.assertEqual(
            usage, {"input_tokens": 30, "output_tokens": 12, "cached_tokens": None}
        )


if __name__ == "__main__":
    unittest.main()
