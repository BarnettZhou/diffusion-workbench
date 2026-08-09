"""FastAPI HTTP/WebSocket 调用端，包装唯一的 WorkbenchCore。"""

from .app import create_app

__all__ = ["create_app"]
