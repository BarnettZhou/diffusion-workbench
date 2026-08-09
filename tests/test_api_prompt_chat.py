import json
import unittest

import httpx

from diffusion_workbench_api import llm as llm_module
from test_api_jobs import ApiTestCase


def _sse_events(text):
    """把 SSE 响应体解析为事件字典列表。"""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


class PromptAssistChatTests(ApiTestCase):
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
        self.captured = []
        self.events = [
            {"type": "thinking", "delta": "想想…"},
            {"type": "content", "delta": "Positive: a cat\n"},
            {"type": "content", "delta": "Negative: blurry"},
            {
                "type": "usage",
                "input_tokens": 10,
                "output_tokens": 5,
                "cached_tokens": None,
            },
        ]
        self._orig_stream = llm_module.stream_llm_chat

        def fake_stream(**kwargs):
            self.captured.append(kwargs)
            events = self.events

            async def gen():
                for event in events:
                    yield event

            return gen()

        llm_module.stream_llm_chat = fake_stream
        self.addCleanup(setattr, llm_module, "stream_llm_chat", self._orig_stream)

    def _chat(
        self,
        instruction="一只猫",
        session_id=None,
        language="zh",
        prompt_style="flux",
    ):
        return self.client.post(
            "/api/v1/prompt-assist/chat",
            json={
                "instruction": instruction,
                "language": language,
                "prompt_style": prompt_style,
                "session_id": session_id,
            },
        )

    def test_first_request_creates_session_and_streams(self):
        response = self._chat()
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])

        events = _sse_events(response.text)
        types = [e["type"] for e in events]
        self.assertEqual(types, ["session", "thinking", "content", "content", "done"])
        session_id = events[0]["session_id"]
        self.assertTrue(session_id)
        self.assertEqual(events[-1]["usage"]["input_tokens"], 10)
        # usage 事件不单独下发,只在 done 里携带
        self.assertNotIn("usage", types)

        # 落请求记录:session 一致,response 为 content 拼接,token 已记录
        records = self.client.get("/api/v1/llm/requests").json()
        self.assertEqual(records["total"], 1)
        item = records["items"][0]
        self.assertEqual(item["session_id"], session_id)
        self.assertEqual(item["response"], "Positive: a cat\nNegative: blurry")
        self.assertEqual(item["input_tokens"], 10)
        self.assertEqual(item["output_tokens"], 5)

        # 首次请求:system + 一条 user
        messages = self.captured[0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "一只猫")

    def test_follow_up_reuses_session_with_history(self):
        first = _sse_events(self._chat("一只猫").text)
        session_id = first[0]["session_id"]

        second = _sse_events(self._chat("换成狗", session_id=session_id).text)
        self.assertEqual(second[0]["session_id"], session_id)

        # 第二次请求携带完整历史:system + 首轮 user/assistant + 新 user
        messages = self.captured[1]["messages"]
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[1]["content"], "一只猫")
        self.assertEqual(
            messages[2]["content"], "Positive: a cat\nNegative: blurry"
        )
        self.assertEqual(messages[3]["content"], "换成狗")

    def test_unknown_session_creates_new(self):
        events = _sse_events(self._chat(session_id="nonexistent").text)
        self.assertEqual(events[0]["type"], "session")
        self.assertNotEqual(events[0]["session_id"], "nonexistent")

    def test_switching_prompt_style_creates_clean_session(self):
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
        first = _sse_events(self._chat(prompt_style="flux").text)
        old_session_id = first[0]["session_id"]

        second = _sse_events(
            self._chat(
                "改成油画",
                session_id=old_session_id,
                prompt_style="sd",
            ).text
        )
        self.assertNotEqual(second[0]["session_id"], old_session_id)
        messages = self.captured[1]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("SD_CUSTOM", messages[0]["content"])
        self.assertNotIn("FLUX_CUSTOM", messages[0]["content"])

    def test_error_event_and_record_keeps_user_message(self):
        def failing_stream(**kwargs):
            self.captured.append(kwargs)

            async def gen():
                yield {"type": "content", "delta": "半截"}
                raise httpx.ConnectError("connection refused")

            return gen()

        llm_module.stream_llm_chat = failing_stream
        events = _sse_events(self._chat("一只猫").text)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("无法连接到大模型服务", events[-1]["detail"])
        session_id = events[0]["session_id"]

        records = self.client.get("/api/v1/llm/requests").json()
        item = records["items"][0]
        self.assertIn("无法连接到大模型服务", item["error"])
        self.assertEqual(item["session_id"], session_id)

        # 失败的助手回复不进上下文,但用户消息保留
        llm_module.stream_llm_chat = lambda **kwargs: self._default_stream(kwargs)
        self._chat("再试一次", session_id=session_id)
        messages = self.captured[-1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "user"])

    def _default_stream(self, kwargs):
        self.captured.append(kwargs)

        async def gen():
            yield {"type": "content", "delta": "Positive: x\nNegative: y"}

        return gen()

    def test_unconfigured_llm_returns_400_json(self):
        self.client.put("/api/v1/settings", json={"llm": {"model": ""}})
        response = self._chat()
        self.assertEqual(response.status_code, 400)
        self.assertIn("请先在设置中配置大模型", response.json()["detail"])


class StreamLlmChatParseTests(unittest.IsolatedAsyncioTestCase):
    """直接验证 stream_llm_chat 对两种接口流式响应的解析。"""

    async def _collect(self, *, interface, lines):
        captured = {}

        class FakeStreamResponse:
            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def aiter_lines(self):
                for line in lines:
                    yield line

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url, json=None, headers=None):
                captured["json"] = json
                return FakeStreamResponse()

        orig = llm_module.httpx.AsyncClient
        llm_module.httpx.AsyncClient = FakeClient
        self.addCleanup(setattr, llm_module.httpx, "AsyncClient", orig)
        events = []
        async for event in llm_module.stream_llm_chat(
            interface=interface,
            base_url="http://127.0.0.1:1",
            api_key="",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            think=True,
        ):
            events.append(event)
        return events, captured["json"]

    async def test_ollama_stream_parses_thinking_content_usage(self):
        lines = [
            json.dumps({"message": {"thinking": "考虑中"}, "done": False}),
            json.dumps({"message": {"content": "Pos"}, "done": False}),
            "",
            json.dumps(
                {
                    "message": {"content": "itive"},
                    "done": True,
                    "prompt_eval_count": 8,
                    "eval_count": 3,
                }
            ),
        ]
        events, payload = await self._collect(interface="ollama", lines=lines)
        self.assertEqual(
            events,
            [
                {"type": "thinking", "delta": "考虑中"},
                {"type": "content", "delta": "Pos"},
                {"type": "content", "delta": "itive"},
                {
                    "type": "usage",
                    "input_tokens": 8,
                    "output_tokens": 3,
                    "cached_tokens": None,
                },
            ],
        )
        self.assertIs(payload["stream"], True)

    async def test_openai_stream_parses_reasoning_content_and_usage(self):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "想"}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        ]
        lines = [*(f"data: {json.dumps(c)}" for c in chunks), "data: [DONE]"]
        events, payload = await self._collect(interface="openai", lines=lines)
        self.assertEqual(
            events,
            [
                {"type": "thinking", "delta": "想"},
                {"type": "content", "delta": "Hello"},
                {
                    "type": "usage",
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cached_tokens": None,
                },
            ],
        )
        self.assertEqual(payload["stream_options"], {"include_usage": True})


if __name__ == "__main__":
    unittest.main()
