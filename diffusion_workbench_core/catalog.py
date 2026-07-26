from pathlib import Path

from .config import WorkbenchConfig
from .domain import Mode, ResourceItem, ResourceKind
from .storage import JobStore


class ResourceCatalog:
    def __init__(self, config: WorkbenchConfig, store: JobStore):
        self.config = config
        self.store = store

    def list(self, mode: Mode, kind: ResourceKind) -> list[ResourceItem]:
        directories = getattr(self.config.resources[mode], kind.value)
        paths: dict[Path, Path] = {}
        for directory in directories:
            if directory.is_dir():
                for path in directory.glob("*.safetensors"):
                    paths[path.resolve()] = path.resolve()
                for path in directory.glob("*.sft"):
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
