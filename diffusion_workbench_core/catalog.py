from __future__ import annotations

from pathlib import Path

from .config import WorkbenchConfig
from .domain import Mode, ResourceItem, ResourceKind
from .storage import JobStore


class ResourceCatalog:
    def __init__(self, config: WorkbenchConfig, store: JobStore):
        self.config = config
        self.store = store

    def list(self, mode: Mode, kind: ResourceKind) -> list[ResourceItem]:
        if mode not in self.config.resources:
            return []
        configured_paths = getattr(self.config.resources[mode], kind.value)
        paths: dict[Path, Path] = {}
        for configured_path in configured_paths:
            if configured_path.is_file() and configured_path.suffix.casefold() in {
                ".safetensors",
                ".sft",
            }:
                paths[configured_path.resolve()] = configured_path.resolve()
            elif configured_path.is_dir():
                for pattern in ("*.safetensors", "*.sft"):
                    for path in configured_path.glob(pattern):
                        paths[path.resolve()] = path.resolve()
        aliases = self.store.get_aliases(mode, kind)
        ordered = sorted(paths.values(), key=lambda path: path.name.casefold())
        return [
            ResourceItem(index=index, path=path, alias=aliases.get(path))
            for index, path in enumerate(ordered, start=1)
        ]

    def set_alias(
        self, mode: Mode, kind: ResourceKind, path: Path, alias: str
    ) -> None:
        self.store.set_alias(mode, kind, path, alias)

    def list_upscale_models(self) -> list[ResourceItem]:
        paths: dict[Path, Path] = {}
        extensions = ("*.pth", "*.pt", "*.safetensors", "*.sft")
        for directory in self.config.upscaling.models:
            if not directory.is_dir():
                continue
            for pattern in extensions:
                for path in directory.glob(pattern):
                    paths[path.resolve()] = path.resolve()
        ordered = sorted(paths.values(), key=lambda path: path.name.casefold())
        return [
            ResourceItem(index=index, path=path)
            for index, path in enumerate(ordered, start=1)
        ]
