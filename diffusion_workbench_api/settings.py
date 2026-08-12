from __future__ import annotations

import asyncio
import json
import math
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from diffusion_workbench_core.domain import (
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
    # wan 系列视频模型(16 的倍数)与 MiniMax 系列(32 的倍数)各自的尺寸标签
    "wan_video_size_presets": _DEFAULT_VIDEO_SIZE_PRESETS,
    "minimax_video_size_presets": [],
    "prompt_presets": [],
    "sampling_defaults": {
        mode.value: {"steps": 8, "sampler": "euler", "scheduler": "simple", "cfg": 1}
        for mode in Mode
    },
    "llm": {
        "interface": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "api_key": "",
        "model": "",
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


# 新设置项在这里注册默认值和校验器。
_LLM_STRING_KEYS = (
    "base_url",
    "api_key",
    "model",
    "system_prompt",
    "sd_system_prompt",
    "format_prompt",
    "language_prompt",
    "think_effort",
)


def _validate_llm(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("llm 必须是对象")
    for key in value:
        if key != "interface" and key != "think" and key not in _LLM_STRING_KEYS:
            raise ValueError(f"llm 包含未知键: {key}")
    # 整项替换存储,缺失键用默认值补齐(方便前端局部提交)
    merged = dict(_DEFAULT_SETTINGS["llm"])
    merged.update(value)
    if merged["interface"] not in ("ollama", "openai"):
        raise ValueError("llm.interface 只允许 ollama 或 openai")
    for key in _LLM_STRING_KEYS:
        if not isinstance(merged[key], str):
            raise ValueError(f"llm.{key} 必须是字符串")
    if not isinstance(merged["think"], bool):
        raise ValueError("llm.think 必须是布尔值")
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


_VALIDATORS = {
    "size_presets": _validate_size_presets,
    "wan_video_size_presets": _validate_size_presets,
    "minimax_video_size_presets": _validate_minimax_size_presets,
    "prompt_presets": _validate_prompt_presets,
    "llm": _validate_llm,
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
