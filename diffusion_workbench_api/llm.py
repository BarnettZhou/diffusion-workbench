from __future__ import annotations

import asyncio
import json
import re
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Literal

from .dependencies import (
    get_llm_record_store,
    get_prompt_session_store,
    get_settings_store,
)
from .settings import SettingsStore

router = APIRouter(prefix="/api/v1")

_LLM_TIMEOUT = 120.0
_MODELS_FETCH_TIMEOUT = 10.0

_LANGUAGES = {"en": "ENG", "zh": "中文"}
PromptStyle = Literal["flux", "sd"]
_STYLE_PROMPT_KEYS: dict[PromptStyle, str] = {
    "flux": "system_prompt",
    "sd": "sd_system_prompt",
}

class PromptAssistRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=4000)
    language: Literal["en", "zh"]
    prompt_style: PromptStyle = "flux"

    @field_validator("instruction")
    @classmethod
    def _instruction_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction 去空白后不能为空")
        return value

class PromptChatRequest(PromptAssistRequest):
    # 为空表示新对话:服务端在首次请求时创建 session
    session_id: str | None = None

class RemoteModelsRequest(BaseModel):
    interface: Literal["ollama", "openai"] = "ollama"
    base_url: str
    api_key: str = ""

class PromptAssistResponse(BaseModel):
    positive: str
    negative: str
    raw: str
    parsed: bool
    session_id: str

def _build_system_prompt(
    llm_config: dict, language: str, prompt_style: PromptStyle = "flux"
) -> str:
    style_prompt_key = _STYLE_PROMPT_KEYS[prompt_style]
    parts = [
        llm_config.get(style_prompt_key, "").strip(),
        llm_config.get("format_prompt", "").strip(),
        llm_config.get("language_prompt", "")
        .strip()
        .replace("{language}", _LANGUAGES[language]),
    ]
    return "\n".join(part for part in parts if part)

def _build_chat_request(
    *,
    interface: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    think: bool,
    think_effort: str,
    stream: bool,
) -> tuple[str, dict, dict]:
    """组装 chat 请求的 (url, payload, headers),流式与非流式共用。"""
    base = base_url.rstrip("/")
    effort = think_effort.strip()
    if interface == "ollama":
        url = f"{base}/api/chat"
        # Ollama 的 think 支持布尔或强度字符串(如 "low"/"high",gpt-oss 等)
        think_param = effort if (think and effort) else think
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think_param,
        }
        headers = {}
    else:
        url = f"{base}/chat/completions"
        payload = {"model": model, "messages": messages}
        # 关闭思考:下发 enable_thinking=false(vLLM/sglang 等本地服务识别;
        # OpenAI 官方接口不认识该字段,开启时保持默认行为不下发)
        if not think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif effort:
            payload["reasoning_effort"] = effort
        if stream:
            payload["stream"] = True
            # 让服务在末尾 chunk 上报 token 用量(不支持的服务会忽略)
            payload["stream_options"] = {"include_usage": True}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return url, payload, headers

def _openai_usage(usage_raw: dict | None) -> dict:
    """从 OpenAI 兼容响应的 usage 字段提取 token 用量,缺项为 None。"""
    usage_raw = usage_raw or {}
    details = usage_raw.get("prompt_tokens_details") or {}
    # DeepSeek 用 prompt_cache_hit_tokens,OpenAI 用 prompt_tokens_details.cached_tokens
    cached = usage_raw.get("prompt_cache_hit_tokens")
    if cached is None:
        cached = details.get("cached_tokens")
    return {
        "input_tokens": usage_raw.get("prompt_tokens"),
        "output_tokens": usage_raw.get("completion_tokens"),
        "cached_tokens": cached,
    }

async def call_llm_chat(
    *,
    interface: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    think: bool,
    think_effort: str = "",
) -> tuple[str, dict]:
    """调用 LLM chat 接口,返回 (文本内容, token 用量)。独立成函数便于测试 monkeypatch。

    token 用量字典键为 input_tokens / output_tokens / cached_tokens,
    服务未上报的项为 None(Ollama 不暴露缓存命中统计)。
    """
    url, payload, headers = _build_chat_request(
        interface=interface,
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        think=think,
        think_effort=think_effort,
        stream=False,
    )
    async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
    data = response.json()
    if interface == "ollama":
        usage = {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "cached_tokens": None,
        }
        return data["message"]["content"], usage
    return data["choices"][0]["message"]["content"], _openai_usage(data.get("usage"))

async def stream_llm_chat(
    *,
    interface: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    think: bool,
    think_effort: str = "",
) -> AsyncIterator[dict]:
    """流式调用 LLM chat,依次产出事件字典。独立成函数便于测试 monkeypatch。

    事件类型:{"type": "thinking"|"content", "delta": str},
    结束时产出 {"type": "usage", input/output/cached_tokens}(服务未上报则各项为 None,
    Ollama 在 done chunk 携带用量,OpenAI 兼容需支持 stream_options.include_usage)。
    """
    url, payload, headers = _build_chat_request(
        interface=interface,
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        think=think,
        think_effort=think_effort,
        stream=True,
    )
    async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            if interface == "ollama":
                # Ollama 流式为逐行 JSON;think 开启时 thinking 与 content 分开推送
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message") or {}
                    if message.get("thinking"):
                        yield {"type": "thinking", "delta": message["thinking"]}
                    if message.get("content"):
                        yield {"type": "content", "delta": message["content"]}
                    if chunk.get("done"):
                        yield {
                            "type": "usage",
                            "input_tokens": chunk.get("prompt_eval_count"),
                            "output_tokens": chunk.get("eval_count"),
                            "cached_tokens": None,
                        }
            else:
                # OpenAI 兼容 SSE:data: {...} 行,以 data: [DONE] 结束;
                # 思考内容走 delta.reasoning_content(DeepSeek/vLLM 约定)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    if chunk.get("usage"):
                        yield {"type": "usage", **_openai_usage(chunk["usage"])}
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content"):
                            yield {"type": "thinking", "delta": delta["reasoning_content"]}
                        if delta.get("content"):
                            yield {"type": "content", "delta": delta["content"]}

_MARKER = re.compile(r"^[#\s\-*]*(positive|negative)\s*[:：]\s*", re.IGNORECASE)

def _parse_content(content: str) -> tuple[str, str] | None:
    """从模型输出中拆出 Positive/Negative 两段,容忍大小写和 markdown 噪声。"""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in content.splitlines():
        match = _MARKER.match(line)
        if match:
            current = match.group(1).lower()
            sections.setdefault(current, [])
            rest = line[match.end() :].strip()
            if rest:
                sections[current].append(rest)
        elif current is not None:
            sections[current].append(line.rstrip())
    positive = "\n".join(sections.get("positive", [])).strip()
    negative = "\n".join(sections.get("negative", [])).strip()
    if "negative" not in sections:
        return None
    return positive, negative

class LLMRecordStore:
    """cache 目录下的 JSON 大模型请求记录,按时间倒序,最多保留 _MAX_RECORDS 条。"""

    _MAX_RECORDS = 200

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._records: list[dict] = []
        self._next_id = 1
        if self._path.is_file():
            try:
                stored = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = []
            if isinstance(stored, list):
                self._records = [r for r in stored if isinstance(r, dict)]
                ids = [
                    r["id"] for r in self._records if isinstance(r.get("id"), int)
                ]
                if ids:
                    self._next_id = max(ids) + 1

    def add(self, record: dict) -> dict:
        with self._lock:
            entry = {"id": self._next_id, **record}
            self._next_id += 1
            self._records.insert(0, entry)
            del self._records[self._MAX_RECORDS :]
            self._save()
            return entry

    def list(self, page: int, page_size: int) -> dict:
        with self._lock:
            start = (page - 1) * page_size
            return {
                "total": len(self._records),
                "page": page,
                "page_size": page_size,
                "items": json.loads(
                    json.dumps(self._records[start : start + page_size])
                ),
            }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)

@dataclass(frozen=True)
class PromptSessionContext:
    language: str
    prompt_style: PromptStyle

@dataclass
class _PromptSession:
    context: PromptSessionContext | None
    history: list[dict] = field(default_factory=list)

class PromptSessionStore:
    """内存中的提示词对话会话,只存 user/assistant 轮次(system prompt 每次请求现拼)。

    不持久化:服务重启即清空,前端也不提供历史 session 浏览。
    """

    _MAX_SESSIONS = 50

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, _PromptSession] = OrderedDict()

    def get_or_create(
        self,
        session_id: str | None,
        context: PromptSessionContext | None = None,
    ) -> tuple[str, list[dict]]:
        """返回会话和历史;会话不存在或语言/风格变化时创建新会话。"""
        with self._lock:
            if (
                session_id
                and session_id in self._sessions
                and (
                    context is None or self._sessions[session_id].context == context
                )
            ):
                self._sessions.move_to_end(session_id)
                return session_id, list(self._sessions[session_id].history)
            new_id = uuid4().hex
            self._sessions[new_id] = _PromptSession(context=context)
            while len(self._sessions) > self._MAX_SESSIONS:
                self._sessions.popitem(last=False)
            return new_id, []

    def append(self, session_id: str, *messages: dict) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.history.extend(messages)
                self._sessions.move_to_end(session_id)

def _llm_error_detail(exc: Exception) -> str:
    """把 httpx/解析异常映射为面向用户的错误描述(与 HTTP 502 detail 一致)。"""
    if isinstance(exc, httpx.ConnectError):
        return f"无法连接到大模型服务: {exc}"
    if isinstance(exc, httpx.TimeoutException):
        return "大模型服务响应超时"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"大模型服务返回错误: HTTP {exc.response.status_code}"
    return f"大模型服务返回异常: {exc}"


async def fetch_remote_models(
    *, interface: str, base_url: str, api_key: str = ""
) -> list[str]:
    """拉取远端可用模型名列表(Ollama /api/tags,OpenAI 兼容 /models)。"""
    base = base_url.rstrip("/")
    if interface == "ollama":
        url = f"{base}/api/tags"
        headers = {}
    else:
        url = f"{base}/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=_MODELS_FETCH_TIMEOUT) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
    data = response.json()
    # 响应结构校验:LM Studio 等服务对未知路径也回 200 + {"error": ...},
    # 缺字段直接报错并提示检查 Base URL,而不是静默返回空列表
    key = "models" if interface == "ollama" else "data"
    items = data.get(key) if isinstance(data, dict) else None
    if not isinstance(items, list):
        detail = data.get("error") if isinstance(data, dict) else None
        raise ValueError(
            f"模型列表响应格式不符合预期({detail or f'缺少 {key} 字段'});"
            "请确认接口类型与 Base URL 是否正确(OpenAI 兼容接口通常以 /v1 结尾)"
        )
    if interface == "ollama":
        return [item.get("name", "") for item in items if isinstance(item, dict)]
    return [item.get("id", "") for item in items if isinstance(item, dict)]


_LLM_CALL_ERRORS = (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError)


def _resolve_llm_target(store: SettingsStore) -> dict:
    config = store.get().get("llm", {})
    selected = config.get("selected", {})
    endpoint_id = selected.get("endpoint_id", "") if isinstance(selected, dict) else ""
    model = selected.get("model", "") if isinstance(selected, dict) else ""
    endpoint = next(
        (item for item in config.get("endpoints", []) if item.get("id") == endpoint_id),
        None,
    )
    if endpoint is None or not model:
        raise HTTPException(
            status_code=400, detail="请先在设置中配置大模型并选择模型"
        )
    resolved = dict(endpoint)
    resolved["model"] = model
    for key in (
        "system_prompt",
        "sd_system_prompt",
        "format_prompt",
        "language_prompt",
        "think",
        "think_effort",
    ):
        resolved[key] = config.get(key, False if key == "think" else "")
    return resolved


_MODELS_FETCH_ERRORS = (
    httpx.HTTPError,
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    AttributeError,
)


@router.post("/remote/models")
async def fetch_remote_models_route(payload: RemoteModelsRequest):
    base_url = payload.base_url.strip()
    if not base_url:
        raise HTTPException(status_code=400, detail="Base URL 去空白后不能为空")
    try:
        models = await fetch_remote_models(
            interface=payload.interface, base_url=base_url, api_key=payload.api_key
        )
    except _MODELS_FETCH_ERRORS as exc:
        raise HTTPException(status_code=502, detail=_llm_error_detail(exc)) from exc
    return {"models": models}


@router.post("/prompt-assist", response_model=PromptAssistResponse)
async def prompt_assist(
    payload: PromptAssistRequest,
    store: SettingsStore = Depends(get_settings_store),
    records: LLMRecordStore = Depends(get_llm_record_store),
):
    llm_config = _resolve_llm_target(store)
    system_prompt = _build_system_prompt(
        llm_config, payload.language, payload.prompt_style
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload.instruction},
    ]
    # 一次性非流式接口:不挂会话,每次请求使用独立的 session id 落记录
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": uuid4().hex,
        "base_url": llm_config["base_url"],
        "model": llm_config["model"],
        "endpoint_name": llm_config.get("name", ""),
        "request": json.dumps(messages, ensure_ascii=False, indent=2),
        "response": "",
        "error": "",
        "input_tokens": None,
        "output_tokens": None,
        "cached_tokens": None,
    }
    try:
        content, usage = await call_llm_chat(
            interface=llm_config["interface"],
            base_url=llm_config["base_url"],
            api_key=llm_config.get("api_key", ""),
            model=llm_config["model"],
            messages=messages,
            think=llm_config.get("think", False),
            think_effort=llm_config.get("think_effort", ""),
        )
    except _LLM_CALL_ERRORS as exc:
        detail = _llm_error_detail(exc)
        record["error"] = detail
        await asyncio.to_thread(records.add, record)
        raise HTTPException(status_code=502, detail=detail) from exc
    record["response"] = content
    record.update(usage)
    await asyncio.to_thread(records.add, record)
    parsed = _parse_content(content)
    if parsed is None:
        return PromptAssistResponse(
            positive="",
            negative="",
            raw=content,
            parsed=False,
            session_id=record["session_id"],
        )
    positive, negative = parsed
    return PromptAssistResponse(
        positive=positive,
        negative=negative,
        raw=content,
        parsed=True,
        session_id=record["session_id"],
    )

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

@router.post("/prompt-assist/chat")
async def prompt_assist_chat(
    payload: PromptChatRequest,
    store: SettingsStore = Depends(get_settings_store),
    records: LLMRecordStore = Depends(get_llm_record_store),
    sessions: PromptSessionStore = Depends(get_prompt_session_store),
):
    """对话式提示词生成,SSE 流式返回。事件序列:

    session(会话 id) → thinking/content 增量 → done(全文+用量) 或 error。
    session_id 为空或已失效时创建新会话;会话内连续对话可调整已生成的提示词。
    """
    llm_config = _resolve_llm_target(store)
    system_prompt = _build_system_prompt(
        llm_config, payload.language, payload.prompt_style
    )
    session_id, history = sessions.get_or_create(
        payload.session_id,
        PromptSessionContext(
            language=payload.language,
            prompt_style=payload.prompt_style,
        ),
    )
    user_message = {"role": "user", "content": payload.instruction}
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        user_message,
    ]
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "base_url": llm_config["base_url"],
        "model": llm_config["model"],
        "endpoint_name": llm_config.get("name", ""),
        "request": json.dumps(messages, ensure_ascii=False, indent=2),
        "response": "",
        "error": "",
        "input_tokens": None,
        "output_tokens": None,
        "cached_tokens": None,
    }

    async def event_stream():
        content_parts: list[str] = []
        usage = {"input_tokens": None, "output_tokens": None, "cached_tokens": None}
        yield _sse({"type": "session", "session_id": session_id})
        try:
            async for event in stream_llm_chat(
                interface=llm_config["interface"],
                base_url=llm_config["base_url"],
                api_key=llm_config.get("api_key", ""),
                model=llm_config["model"],
                messages=messages,
                think=llm_config.get("think", False),
                think_effort=llm_config.get("think_effort", ""),
            ):
                if event["type"] == "content":
                    content_parts.append(event["delta"])
                elif event["type"] == "usage":
                    usage.update(event)
                    usage.pop("type", None)
                    continue  # 用量只在 done 事件里下发
                yield _sse(event)
        except _LLM_CALL_ERRORS as exc:
            detail = _llm_error_detail(exc)
            record["error"] = detail
            await asyncio.to_thread(records.add, record)
            # 用户消息保留在会话里,便于下一条消息继续;失败的助手回复不进上下文
            sessions.append(session_id, user_message)
            yield _sse({"type": "error", "detail": detail})
            return
        content = "".join(content_parts)
        record["response"] = content
        record.update(usage)
        await asyncio.to_thread(records.add, record)
        sessions.append(
            session_id, user_message, {"role": "assistant", "content": content}
        )
        yield _sse({"type": "done", "usage": usage})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/llm/requests")
async def list_llm_requests(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    records: LLMRecordStore = Depends(get_llm_record_store),
):
    return await asyncio.to_thread(records.list, page, page_size)
