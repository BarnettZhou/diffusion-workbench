from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from diffusion_workbench_core import WorkbenchCore

from . import album, caption, edit, events, files, jobs, llm, models, settings, video
from .album import AlbumDirStore, AlbumManager
from .console_status import ConsoleStatusBar
from .events import EventHub
from .llm import LLMRecordStore, PromptSessionStore
from .models import ModelInfoStore
from .settings import SettingsStore

DEFAULT_CONFIG_PATH = "configs/workbench.yaml"


def create_app(
    config_path: str | Path | None = None,
    core_factory: Callable[[], WorkbenchCore] | None = None,
) -> FastAPI:
    """创建 FastAPI 应用。Core 只在 lifespan 中创建/关闭，单进程单实例。

    ``core_factory`` 仅供测试注入 fake Core；生产使用默认的
    ``WorkbenchCore.from_config``。
    """
    config = Path(config_path or os.environ.get("DWB_CONFIG", DEFAULT_CONFIG_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 终端底部状态栏仅 TTY 启用（非 TTY 为 no-op），需在 Core 创建前启动，
        # 让启动期日志也从状态栏上方滚出
        status_bar = ConsoleStatusBar()
        status_bar.start()
        if core_factory is not None:
            core = await asyncio.to_thread(core_factory)
        else:
            core = await asyncio.to_thread(WorkbenchCore.from_config, config)
        hub = EventHub(asyncio.get_running_loop())

        def event_sink(event: dict) -> None:
            # Core 只允许单 event sink，在此扇出到 WebSocket Hub 与终端状态栏
            hub.emit_from_core_thread(event)
            status_bar.handle_event(event)

        core.set_event_sink(event_sink)
        core.set_preview_enabled(True)
        app.state.core = core
        app.state.event_hub = hub
        app.state.album_manager = AlbumManager(
            core.config.output_dir,
            AlbumDirStore(core.config.database.parent / "album_dirs.json"),
            poster_dir=core.config.database.parent / "album_posters",
            thumbnail_dir=core.config.database.parent / "gallery_thumbnails",
        )
        app.state.settings_store = SettingsStore(
            core.config.database.parent / "settings.json"
        )
        app.state.model_info_store = ModelInfoStore(core.config.database.parent)
        app.state.llm_record_store = LLMRecordStore(
            core.config.database.parent / "llm_requests.json"
        )
        app.state.caption_record_store = LLMRecordStore(
            core.config.database.parent / "caption_requests.json"
        )
        app.state.prompt_session_store = PromptSessionStore()
        try:
            yield
        finally:
            hub.close()
            await asyncio.to_thread(core.shutdown)
            status_bar.stop()

    app = FastAPI(title="diffusion-workbench API", lifespan=lifespan)
    app.include_router(jobs.router)
    app.include_router(files.router)
    app.include_router(album.router)
    app.include_router(settings.router)
    app.include_router(models.router)
    app.include_router(llm.router)
    app.include_router(events.router)
    app.include_router(video.router)
    app.include_router(edit.router)
    app.include_router(caption.router)

    # 静态托管前端构建产物(单端口同源,无需反向代理)。API 路由先注册,
    # 优先于根挂载;dist 不存在时(如未构建)跳过。
    from fastapi.staticfiles import StaticFiles

    # JS/CSS 产物文件名带内容 hash,可放心缓存;但 index.html 必须每次重新校验,
    # 否则 iOS 等浏览器的启发式缓存会一直引用旧 hash 的资源,前端更新不生效。
    @app.middleware("http")
    async def _html_no_cache(request, call_next):
        response = await call_next(request)
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
    return app


app = create_app()
