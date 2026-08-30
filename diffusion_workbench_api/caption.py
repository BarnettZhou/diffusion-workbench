from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from diffusion_workbench_core.caption import DEFAULT_CAPTION_SYSTEM_PROMPT

from .dependencies import get_caption_record_store, get_core, get_settings_store
from .llm import LLMRecordStore, fetch_remote_models
from .service import resolve_video_input_image
from .settings import SettingsStore

router = APIRouter(prefix="/api/v1")

# API 反推走外部视觉模型,生成可能较慢
_REMOTE_TIMEOUT = 300.0


class CaptionRequest(BaseModel):
    """图片反推请求;image_id 复用 /edit/input-images 上传通道返回的受控 id。"""

    image_id: str
    hint: str = Field(default="", max_length=2000)
    max_length: int = Field(default=2048, ge=64, le=2048)
    seed: int = -1


def _build_remote_request(
    *,
    interface: str,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_b64: str,
    mime: str,
    think: bool,
    max_length: int,
    seed: int,
) -> tuple[str, dict, dict]:
    """组装视觉反推请求的 (url, payload, headers);think 语义与 llm.py 保持一致。"""
    base = base_url.rstrip("/")
    if interface == "ollama":
        url = f"{base}/api/chat"
        options: dict = {"num_predict": max_length}
        if seed >= 0:
            options["seed"] = seed
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "images": [image_b64]}
            ],
            "stream": False,
            "think": think,
            "options": options,
        }
        headers = {}
    else:
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_length,
        }
        if seed >= 0:
            payload["seed"] = seed
        # 关闭思考:下发 enable_thinking=false(vLLM/sglang 等本地服务识别)
        if not think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return url, payload, headers


async def call_remote_caption(
    *,
    interface: str,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_b64: str,
    mime: str,
    think: bool,
    max_length: int,
    seed: int,
) -> tuple[str, dict]:
    """调用外部视觉模型接口反推图片,返回 (文本内容, token 用量)。

    独立成函数便于测试 monkeypatch。token 用量字典键为
    input_tokens / output_tokens / cached_tokens,服务未上报的项为 None。
    """
    url, payload, headers = _build_remote_request(
        interface=interface,
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        image_b64=image_b64,
        mime=mime,
        think=think,
        max_length=max_length,
        seed=seed,
    )
    async with httpx.AsyncClient(timeout=_REMOTE_TIMEOUT) as client:
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
    usage_raw = data.get("usage") or {}
    usage = {
        "input_tokens": usage_raw.get("prompt_tokens"),
        "output_tokens": usage_raw.get("completion_tokens"),
        "cached_tokens": None,
    }
    return data["choices"][0]["message"]["content"], usage


def _remote_error_detail(exc: Exception) -> str:
    """把 httpx/解析异常映射为面向用户的错误描述(与 HTTP 502 detail 一致)。"""
    if isinstance(exc, httpx.ConnectError):
        return f"无法连接到反推 API 服务: {exc}"
    if isinstance(exc, httpx.TimeoutException):
        return "反推 API 服务响应超时"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"反推 API 服务返回错误: HTTP {exc.response.status_code}"
    return f"反推 API 服务返回异常: {exc}"


_REMOTE_CALL_ERRORS = (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError)

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _resolve_remote_config(store: SettingsStore) -> dict:
    config = store.get().get("caption_api", {})
    selected = config.get("selected", {})
    endpoint_id = selected.get("endpoint_id", "") if isinstance(selected, dict) else ""
    model = selected.get("model", "") if isinstance(selected, dict) else ""
    endpoint = next(
        (item for item in config.get("endpoints", []) if item.get("id") == endpoint_id),
        None,
    )
    if endpoint is None or not model:
        raise HTTPException(
            status_code=400,
            detail="请先在设置中配置图片反推 API 并选择模型",
        )
    resolved = dict(endpoint)
    resolved["model"] = model
    resolved["prompt"] = config.get("prompt", DEFAULT_CAPTION_SYSTEM_PROMPT)
    resolved["think"] = config.get("think", False)
    return resolved


@router.post("/caption")
async def create_caption(payload: CaptionRequest, core=Depends(get_core)):
    """同步图片反推;LookupError→404,ValueError/FileNotFoundError→422。

    反推与生成任务共用同一个 GPU Worker 串行执行(经 runtime 生成锁排队),
    正在跑生成任务时本请求会阻塞等待;结果不落库、不发 WebSocket 事件,
    stop/skip 不作用于反推。
    """
    try:
        image_path = await asyncio.to_thread(
            resolve_video_input_image, core, payload.image_id
        )
        result = await asyncio.to_thread(
            core.describe_image,
            image_path,
            hint=payload.hint,
            max_length=payload.max_length,
            seed=payload.seed,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Worker 关闭/反推执行失败等运行时错误,按 500 返回并保留中文错误信息。
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/caption/remote")
async def create_caption_remote(
    payload: CaptionRequest,
    core=Depends(get_core),
    store: SettingsStore = Depends(get_settings_store),
    records: LLMRecordStore = Depends(get_caption_record_store),
):
    """API 反推:把图片发给设置中配置的外部视觉模型接口(Ollama / OpenAI 兼容)。

    与本地反推一样是同步请求,但不经过 GPU Worker、不排队;结果不落库、
    不发 WebSocket 事件。提示词取 settings.caption_api.prompt,hint 追加在末尾。
    每次调用(含失败)落一条请求记录,见 GET /caption/remote/requests。
    """
    config = _resolve_remote_config(store)
    try:
        image_path = await asyncio.to_thread(
            resolve_video_input_image, core, payload.image_id
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        image_bytes = await asyncio.to_thread(image_path.read_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    prompt = config.get("prompt", "").strip() or DEFAULT_CAPTION_SYSTEM_PROMPT
    if payload.hint:
        prompt += f"\n\nAdditional user instructions: {payload.hint}"
    # 请求记录不落 base64 图片本体,只记大小
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_url": config["base_url"],
        "model": config["model"],
        "endpoint_name": config.get("name", ""),
        "request": json.dumps(
            {
                "prompt": prompt,
                "image": f"<base64, {len(image_bytes)} bytes>",
                "think": config.get("think", False),
                "max_length": payload.max_length,
                "seed": payload.seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "response": "",
        "error": "",
        "input_tokens": None,
        "output_tokens": None,
        "cached_tokens": None,
    }
    started = time.monotonic()
    try:
        caption, usage = await call_remote_caption(
            interface=config["interface"],
            base_url=config["base_url"],
            api_key=config.get("api_key", ""),
            model=config["model"],
            prompt=prompt,
            image_b64=base64.b64encode(image_bytes).decode("ascii"),
            mime=_MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png"),
            think=config.get("think", False),
            max_length=payload.max_length,
            seed=payload.seed,
        )
    except _REMOTE_CALL_ERRORS as exc:
        detail = _remote_error_detail(exc)
        record["error"] = detail
        await asyncio.to_thread(records.add, record)
        raise HTTPException(status_code=502, detail=detail) from exc
    record["response"] = caption
    record.update(usage)
    await asyncio.to_thread(records.add, record)
    return {
        "caption": caption,
        "load_seconds": 0.0,
        "infer_seconds": time.monotonic() - started,
    }


@router.get("/caption/remote/requests")
async def list_caption_remote_requests(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    records: LLMRecordStore = Depends(get_caption_record_store),
):
    """API 反推请求记录(按时间倒序分页,最多保留 200 条)。"""
    return await asyncio.to_thread(records.list, page, page_size)
