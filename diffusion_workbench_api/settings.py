from __future__ import annotations

import asyncio
import json
import math
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from diffusion_workbench_core.caption import DEFAULT_CAPTION_SYSTEM_PROMPT
from diffusion_workbench_core.domain import (
    H3_REF2VA_MAX_AUDIOS,
    H3_REF2VA_MAX_IMAGES,
    H3_REF2VA_MAX_TOTAL,
    H3_REF2VA_MAX_VIDEOS,
    MAX_STEPS,
    MIN_STEPS,
    SAMPLERS,
    SCHEDULERS,
    Mode,
)

from .dependencies import get_settings_store

router = APIRouter(prefix="/api/v1")

_DEFAULT_VIDEO_SIZE_PRESETS = [
    [704, 960],
    [960, 704],
    [1280, 720],
    [720, 1280],
]

_DEFAULT_SETTINGS = {
    "size_presets": [[576, 576], [768, 768], [1024, 1024], [960, 1280]],
    # 尺寸卡片「自动计算」模式可选的图片比例([宽, 高] 的最简整数比)
    "aspect_ratio_presets": [[1, 1], [3, 4], [4, 3], [5, 4], [4, 5], [16, 9], [9, 16]],
    # wan 系列视频模型(16 的倍数)与 MiniMax 系列(32 的倍数)各自的尺寸标签
    "wan_video_size_presets": _DEFAULT_VIDEO_SIZE_PRESETS,
    "minimax_video_size_presets": [],
    # Ref2VA 客户端表单输入上限(本地算力有限,由用户自主控制;硬上限见 core domain)
    "ref2va_limits": {"max_images": 3, "max_videos": 1, "max_audios": 1},
    "prompt_presets": [],
    "sampling_defaults": {
        mode.value: {"steps": 8, "sampler": "euler", "scheduler": "simple", "cfg": 1}
        for mode in Mode
    },
    "llm": {
        "endpoints": [],
        "selected": {"endpoint_id": "", "model": ""},
        "system_prompt": "",
        "sd_system_prompt": (
            "你是 Stable Diffusion / SDXL 提示词专家。将用户描述改写为适合 SDXL 的"
            "逗号分隔短语,把主体和关键特征放在前面,再补充构图、镜头、光线、材质与"
            "画面风格。避免 FLUX 式长句和无意义质量词堆叠;仅在确有必要时使用"
            "(关键词:权重)语法。负面提示词应列出与画面目标直接相关的缺陷和排除项。"
        ),
        "format_prompt": (
            "你是文生图提示词专家,根据用户描述生成一段正面提示词和一段负面提示词。"
            "严格按以下两行格式输出,不要输出任何其他内容:\n"
            "Positive: ...\n"
            "Negative: ..."
        ),
        "language_prompt": "正面提示词和负面提示词均使用 {language} 输出。",
        "think": False,
        "think_effort": "",
    },
    # API 反推(图片反推 tab 的「API 反推」):走外部视觉模型接口,不占用本地 GPU。
    # prompt 默认值与本地反推的系统提示词一致
    "caption_api": {
        "endpoints": [],
        "selected": {"endpoint_id": "", "model": ""},
        "prompt": DEFAULT_CAPTION_SYSTEM_PROMPT,
        "think": False,
    },
}


def _make_size_presets_validator(multiple: int):
    def validate(value) -> list[list[int]]:
        if not isinstance(value, list):
            raise ValueError("size_presets 必须是 [宽, 高] 列表")
        presets = []
        seen = set()
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("每个尺寸标签必须是 [宽, 高] 两项")
            width, height = item
            if (
                not isinstance(width, int)
                or not isinstance(height, int)
                or isinstance(width, bool)
                or isinstance(height, bool)
                or width <= 0
                or height <= 0
                or width % multiple
                or height % multiple
            ):
                raise ValueError(f"宽高必须是正整数且为 {multiple} 的倍数")
            key = (width, height)
            if key not in seen:
                seen.add(key)
                presets.append([width, height])
        return presets

    return validate


_validate_size_presets = _make_size_presets_validator(16)
_validate_minimax_size_presets = _make_size_presets_validator(32)


def _validate_aspect_ratio_presets(value) -> list[list[int]]:
    """图片比例预设:每项是 [宽, 高] 的正整数比(无倍数要求),自动去重。"""
    if not isinstance(value, list):
        raise ValueError("aspect_ratio_presets 必须是 [宽, 高] 比例列表")
    presets = []
    seen = set()
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("每个比例必须是 [宽, 高] 两项")
        width, height = item
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or isinstance(width, bool)
            or isinstance(height, bool)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("比例的宽和高必须是正整数")
        key = (width, height)
        if key not in seen:
            seen.add(key)
            presets.append([width, height])
    return presets


# 新设置项在这里注册默认值和校验器。
_LLM_STRING_KEYS = (
    "system_prompt",
    "sd_system_prompt",
    "format_prompt",
    "language_prompt",
    "think_effort",
)
_LLM_LEGACY_KEYS = ("interface", "base_url", "api_key", "model")
_LLM_ALLOWED_KEYS = (
    "endpoints",
    "selected",
    "system_prompt",
    "sd_system_prompt",
    "format_prompt",
    "language_prompt",
    "think",
    "think_effort",
    *_LLM_LEGACY_KEYS,
)


def _merge_settings(default: dict) -> dict:
    merged = dict(default)
    merged["endpoints"] = list(default["endpoints"])
    merged["selected"] = dict(default["selected"])
    return merged


def _validate_endpoints(value) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("endpoints 必须是列表")
    endpoints = []
    used_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("每个端点必须是对象")
        for key in item:
            if key not in ("id", "name", "interface", "base_url", "api_key", "models"):
                raise ValueError(f"端点包含未知键: {key}")
        endpoint_id = item.get("id")
        if (
            not isinstance(endpoint_id, str)
            or len(endpoint_id) != 8
            or any(char not in "0123456789abcdefABCDEF" for char in endpoint_id)
        ):
            # id 缺失或格式非法(如前端本地新建端点的 local-* 临时 id)时自动生成
            endpoint_id = uuid4().hex[:8]
        while endpoint_id in used_ids:
            endpoint_id = uuid4().hex[:8]
        used_ids.add(endpoint_id)
        if not isinstance(item.get("name"), str):
            raise ValueError("端点 name 必须是字符串")
        interface = item.get("interface")
        if interface not in ("ollama", "openai"):
            raise ValueError("端点 interface 只允许 ollama 或 openai")
        base_url = item.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("端点 base_url 必须是非空字符串")
        if not isinstance(item.get("api_key"), str):
            raise ValueError("端点 api_key 必须是字符串")
        models = item.get("models")
        if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
            raise ValueError("端点 models 必须是字符串列表")
        model_names = list(dict.fromkeys(model for model in models if model))
        endpoints.append(
            {
                "id": endpoint_id,
                "name": item["name"],
                "interface": interface,
                "base_url": base_url,
                "api_key": item["api_key"],
                "models": model_names,
            }
        )
    return endpoints


def _validate_selected(value, endpoints) -> dict:
    if not isinstance(value, dict):
        raise ValueError("selected 必须是对象")
    for key in value:
        if key not in ("endpoint_id", "model"):
            raise ValueError(f"selected 包含未知键: {key}")
    endpoint_id = value.get("endpoint_id")
    model = value.get("model")
    if not isinstance(endpoint_id, str) or not isinstance(model, str):
        raise ValueError("selected.endpoint_id 和 selected.model 必须是字符串")
    if endpoint_id == "" and model == "":
        return {"endpoint_id": "", "model": ""}
    if endpoint_id not in {endpoint["id"] for endpoint in endpoints}:
        raise ValueError("selected.endpoint_id 必须引用已配置的端点")
    if not model:
        raise ValueError("selected.model 必须是非空字符串")
    return {"endpoint_id": endpoint_id, "model": model}


def _validate_legacy_values(value, label: str) -> None:
    interface = value.get("interface")
    if interface is not None and interface not in ("ollama", "openai"):
        raise ValueError(f"{label}.interface 只允许 ollama 或 openai")
    for key in ("base_url", "api_key", "model"):
        if key not in value:
            continue
        item = value[key]
        if not isinstance(item, str):
            raise ValueError(f"{label}.{key} 必须是字符串")
        if key == "base_url" and not item.strip():
            raise ValueError(f"{label}.base_url 必须是非空字符串")


def _validate_llm(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("llm 必须是对象")
    for key in value:
        if key not in _LLM_ALLOWED_KEYS:
            raise ValueError(f"llm 包含未知键: {key}")
    _validate_legacy_values(value, "llm")
    merged = _merge_settings(_DEFAULT_SETTINGS["llm"])
    merged.update(value)
    legacy_keys = [key for key in _LLM_LEGACY_KEYS if key in value]
    if legacy_keys:
        # 旧平铺格式迁移:update 时传入值已与已存值合并(可能带 endpoints),
        # 因此只要出现旧键就按旧值重置为单个端点
        endpoint = {
            "id": "",
            "name": "",
            "interface": merged.get("interface", "ollama"),
            "base_url": merged.get("base_url", ""),
            "api_key": merged.get("api_key", ""),
            "models": [merged["model"]] if isinstance(merged.get("model"), str) and merged["model"] else [],
        }
        merged["endpoints"] = _validate_endpoints([endpoint])
        merged["selected"] = {
            "endpoint_id": merged["endpoints"][0]["id"] if merged["model"] else "",
            "model": merged["model"] if isinstance(merged.get("model"), str) else "",
        }
    for key in legacy_keys:
        merged.pop(key, None)
    merged["endpoints"] = _validate_endpoints(merged.get("endpoints"))
    merged["selected"] = _validate_selected(
        merged.get("selected", {"endpoint_id": "", "model": ""}), merged["endpoints"]
    )
    for key in _LLM_STRING_KEYS:
        if not isinstance(merged[key], str):
            raise ValueError(f"llm.{key} 必须是字符串")
    if not isinstance(merged["think"], bool):
        raise ValueError("llm.think 必须是布尔值")
    return merged


_CAPTION_API_STRING_KEYS = ("prompt",)
_CAPTION_API_LEGACY_KEYS = ("interface", "base_url", "api_key", "model")
_CAPTION_API_ALLOWED_KEYS = (
    "endpoints",
    "selected",
    "prompt",
    "think",
    *_CAPTION_API_LEGACY_KEYS,
)


def _validate_caption_api(value) -> dict:
    """API 反推设置;允许只提交部分键,缺失键用默认值补齐。"""
    if not isinstance(value, dict):
        raise ValueError("caption_api 必须是对象")
    for key in value:
        if key not in _CAPTION_API_ALLOWED_KEYS:
            raise ValueError(f"caption_api 包含未知键: {key}")
    _validate_legacy_values(value, "caption_api")
    merged = _merge_settings(_DEFAULT_SETTINGS["caption_api"])
    merged.update(value)
    legacy_keys = [key for key in _CAPTION_API_LEGACY_KEYS if key in value]
    if legacy_keys:
        # 旧平铺格式迁移:同 _validate_llm,出现旧键即重置为单个端点
        endpoint = {
            "id": "",
            "name": "",
            "interface": merged.get("interface", "ollama"),
            "base_url": merged.get("base_url", ""),
            "api_key": merged.get("api_key", ""),
            "models": [merged["model"]] if isinstance(merged.get("model"), str) and merged["model"] else [],
        }
        merged["endpoints"] = _validate_endpoints([endpoint])
        merged["selected"] = {
            "endpoint_id": merged["endpoints"][0]["id"] if merged["model"] else "",
            "model": merged["model"] if isinstance(merged.get("model"), str) else "",
        }
    for key in legacy_keys:
        merged.pop(key, None)
    merged["endpoints"] = _validate_endpoints(merged.get("endpoints"))
    merged["selected"] = _validate_selected(
        merged.get("selected", {"endpoint_id": "", "model": ""}), merged["endpoints"]
    )
    for key in _CAPTION_API_STRING_KEYS:
        if not isinstance(merged[key], str):
            raise ValueError(f"caption_api.{key} 必须是字符串")
    if not isinstance(merged["think"], bool):
        raise ValueError("caption_api.think 必须是布尔值")
    return merged


def _validate_sampling_defaults(value) -> dict:
    """按模式校验默认采样参数;允许只提交部分模式/部分字段,缺失项用默认值补齐。"""
    if not isinstance(value, dict):
        raise ValueError("sampling_defaults 必须是对象")
    defaults = _DEFAULT_SETTINGS["sampling_defaults"]
    for mode in value:
        if mode not in defaults:
            raise ValueError(f"sampling_defaults 包含未知模式: {mode}")
    merged = {mode: dict(params) for mode, params in defaults.items()}
    for mode, params in value.items():
        if not isinstance(params, dict):
            raise ValueError(f"sampling_defaults.{mode} 必须是对象")
        for key in params:
            if key not in ("steps", "sampler", "scheduler", "cfg"):
                raise ValueError(f"sampling_defaults.{mode} 包含未知键: {key}")
        merged[mode].update(params)
    for mode, params in merged.items():
        steps = params["steps"]
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or not MIN_STEPS <= steps <= MAX_STEPS
        ):
            raise ValueError(
                f"sampling_defaults.{mode}.steps 必须在 {MIN_STEPS} 到 {MAX_STEPS} 之间"
            )
        cfg = params["cfg"]
        if (
            isinstance(cfg, bool)
            or not isinstance(cfg, (int, float))
            or not math.isfinite(cfg)
            or cfg <= 0
        ):
            raise ValueError(f"sampling_defaults.{mode}.cfg 必须是大于 0 的数值")
        sampler = params["sampler"]
        if sampler not in SAMPLERS:
            raise ValueError(f"sampling_defaults.{mode}.sampler 不支持: {sampler}")
        scheduler = params["scheduler"]
        if scheduler not in SCHEDULERS:
            raise ValueError(f"sampling_defaults.{mode}.scheduler 不支持: {scheduler}")
    return merged


def _validate_prompt_presets(value) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("prompt_presets 必须是提示词预设列表")
    presets = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("每个提示词预设必须是对象")
        for key in item:
            if key not in ("id", "title", "kind", "text"):
                raise ValueError(f"提示词预设包含未知键: {key}")
        pid = item.get("id")
        if not isinstance(pid, str) or not pid:
            raise ValueError("提示词预设 id 必须是非空字符串")
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("提示词预设标题不能为空")
        kind = item.get("kind")
        if kind not in ("positive", "negative"):
            raise ValueError("提示词预设类型只允许 positive 或 negative")
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("提示词预设内容必须是字符串")
        presets.append({"id": pid, "title": title, "kind": kind, "text": text})
    return presets


def _validate_ref2va_limits(value) -> dict:
    """Ref2VA 表单输入上限;允许只提交部分键,缺失项用默认值补齐。"""
    if not isinstance(value, dict):
        raise ValueError("ref2va_limits 必须是对象")
    for key in value:
        if key not in ("max_images", "max_videos", "max_audios"):
            raise ValueError(f"ref2va_limits 包含未知键: {key}")
    merged = dict(_DEFAULT_SETTINGS["ref2va_limits"])
    merged.update(value)
    bounds = {
        "max_images": H3_REF2VA_MAX_IMAGES,
        "max_videos": H3_REF2VA_MAX_VIDEOS,
        "max_audios": H3_REF2VA_MAX_AUDIOS,
    }
    for key, upper in bounds.items():
        item = merged[key]
        if not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= upper:
            raise ValueError(f"ref2va_limits.{key} 必须是 0 到 {upper} 的整数")
    if sum(merged[key] for key in bounds) > H3_REF2VA_MAX_TOTAL:
        raise ValueError(f"ref2va_limits 三项之和不能超过 {H3_REF2VA_MAX_TOTAL}")
    return merged


_VALIDATORS = {
    "size_presets": _validate_size_presets,
    "aspect_ratio_presets": _validate_aspect_ratio_presets,
    "ref2va_limits": _validate_ref2va_limits,
    "wan_video_size_presets": _validate_size_presets,
    "minimax_video_size_presets": _validate_minimax_size_presets,
    "prompt_presets": _validate_prompt_presets,
    "llm": _validate_llm,
    "caption_api": _validate_caption_api,
    "sampling_defaults": _validate_sampling_defaults,
}


class SettingsStore:
    """cache 目录下的 JSON 设置存储,所有设置项集中在这一个文件里。"""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._data = {key: value for key, value in _DEFAULT_SETTINGS.items()}
        if self._path.is_file():
            try:
                stored = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = {}
            if isinstance(stored, dict):
                # 旧版统一的 video_size_presets 属于 wan 系列,迁移为新键
                if "wan_video_size_presets" not in stored and "video_size_presets" in stored:
                    stored["wan_video_size_presets"] = stored["video_size_presets"]
                for key, validator in _VALIDATORS.items():
                    if key in stored:
                        try:
                            self._data[key] = validator(stored[key])
                        except ValueError:
                            pass  # 损坏的项回退默认值

    def get(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def update(self, patch: dict) -> dict:
        with self._lock:
            for key, value in patch.items():
                validator = _VALIDATORS.get(key)
                if validator is None:
                    raise ValueError(f"未知设置项: {key}")
                current = self._data[key]
                if isinstance(current, dict) and isinstance(value, dict):
                    # 部分键提交(如只更新 llm.selected)基于当前已存值合并,
                    # 不能回退默认值,否则未提交的键(如 endpoints)会被清空
                    value = {**current, **value}
                self._data[key] = validator(value)
            self._save()
            return self.get()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)


@router.get("/settings")
async def get_settings(store: SettingsStore = Depends(get_settings_store)):
    return await asyncio.to_thread(store.get)


@router.put("/settings")
async def update_settings(
    payload: dict, store: SettingsStore = Depends(get_settings_store)
):
    try:
        return await asyncio.to_thread(store.update, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
