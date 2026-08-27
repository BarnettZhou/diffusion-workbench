from __future__ import annotations

from pathlib import Path

from .config import WorkbenchConfig
from .domain import Mode, ResourceItem, ResourceKind, VideoModel, is_h3_ref2va_model_name
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
        # 模型文件已从磁盘删除时,自动删除索引中对应的别名记录
        self.store.prune_aliases(mode, kind, set(paths.values()))
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

    def list_video_models(self, video_model: VideoModel) -> list[ResourceItem]:
        resources = self.config.video_resources.get(video_model)
        if resources is None:
            return []
        items = self._list_model_files(resources.diffusion, include_gguf=True)
        # MiniMax H3 任务权重通常混放在同一目录,按文件名约定(ref2va 子串)拆分列表;
        # turbo 与 fl2va 共用同一批 fl2va 权重
        if video_model in (VideoModel.MINIMAX_H3_FL2VA, VideoModel.MINIMAX_H3_TURBO):
            return self._reindex(
                item for item in items if not is_h3_ref2va_model_name(item.path.name)
            )
        if video_model == VideoModel.MINIMAX_H3_REF2VA:
            return self._reindex(
                item for item in items if is_h3_ref2va_model_name(item.path.name)
            )
        return items

    @staticmethod
    def _reindex(items) -> list[ResourceItem]:
        return [
            ResourceItem(index=index, path=item.path, alias=item.alias)
            for index, item in enumerate(items, start=1)
        ]

    def list_video_vaes(self, video_model: VideoModel) -> list[ResourceItem]:
        resources = self.config.video_resources.get(video_model)
        if resources is None:
            return []
        return self._list_model_files(resources.vae)

    def list_video_text_encoders(self, video_model: VideoModel) -> list[ResourceItem]:
        # 视频 text encoder 与图片侧一样是"目录/文件列表 + index 选择";不含 gguf
        # 与 diffusion 列表的拆分逻辑无关,直接全量扫描。
        resources = self.config.video_resources.get(video_model)
        if resources is None:
            return []
        return self._list_model_files(resources.text_encoder, include_gguf=True)

    @staticmethod
    def _list_model_files(
        configured_paths: tuple[Path, ...], *, include_gguf: bool = False
    ) -> list[ResourceItem]:
        paths: dict[Path, Path] = {}
        suffixes = {".safetensors", ".sft"}
        patterns = ["*.safetensors", "*.sft"]
        if include_gguf:
            suffixes.add(".gguf")
            patterns.append("*.gguf")
        for configured_path in configured_paths:
            if configured_path.is_file() and configured_path.suffix.casefold() in suffixes:
                paths[configured_path.resolve()] = configured_path.resolve()
            elif configured_path.is_dir():
                for pattern in patterns:
                    for path in configured_path.glob(pattern):
                        paths[path.resolve()] = path.resolve()
        ordered = sorted(paths.values(), key=lambda path: path.name.casefold())
        return [
            ResourceItem(index=index, path=path)
            for index, path in enumerate(ordered, start=1)
        ]
