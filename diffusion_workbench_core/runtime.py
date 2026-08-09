from collections.abc import Callable
from typing import Protocol

from .domain import JobRecord, VideoJobRecord


class GenerationCancelled(RuntimeError):
    pass


class GenerationRuntime(Protocol):
    def generate(
        self,
        job: JobRecord | VideoJobRecord,
        progress: Callable[[int, int, dict], None],
        stage: Callable[[str, int | None], None],
        preview: Callable[[dict], None] | None = None,
    ) -> dict: ...

    def cancel(self) -> None: ...

    def release_resources(self) -> None: ...

    def close(self) -> None: ...

    def status(self) -> dict: ...

    def set_preview_enabled(self, enabled: bool) -> None: ...
