from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from diffusion_workbench_core import WorkbenchCore

from .events import EventHub

if TYPE_CHECKING:
    from .album import AlbumManager
    from .llm import LLMRecordStore, PromptSessionStore
    from .models import ModelInfoStore
    from .settings import SettingsStore


def get_core(request: Request) -> WorkbenchCore:
    return request.app.state.core


def get_event_hub(request: Request) -> EventHub:
    return request.app.state.event_hub


def get_album_manager(request: Request) -> AlbumManager:
    return request.app.state.album_manager


def get_settings_store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def get_model_info_store(request: Request) -> ModelInfoStore:
    return request.app.state.model_info_store


def get_llm_record_store(request: Request) -> LLMRecordStore:
    return request.app.state.llm_record_store


def get_caption_record_store(request: Request) -> LLMRecordStore:
    """API 反推请求记录;与 llm 共用 LLMRecordStore 实现,独立 JSON 文件。"""
    return request.app.state.caption_record_store


def get_prompt_session_store(request: Request) -> PromptSessionStore:
    return request.app.state.prompt_session_store
