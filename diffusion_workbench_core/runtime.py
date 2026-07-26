from collections.abc import Callable
from typing import Protocol

from .domain import JobRecord


class GenerationCancelled(RuntimeError):
    pass


class GenerationRuntime(Protocol):
    def generate(
        self,
        job: JobRecord,
        progress: Callable[[int, int], None],
        stage: Callable[[str, int | None], None],
    ) -> dict: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...

    def status(self) -> dict: ...
