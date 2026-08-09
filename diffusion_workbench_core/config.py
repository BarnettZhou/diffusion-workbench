from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .domain import Mode, ModelLoader, VideoModel


@dataclass(frozen=True)
class ComfyConfig:
    root: Path
    python: Path


@dataclass(frozen=True)
class ModeResources:
    diffusion: tuple[Path, ...]
    vae: tuple[Path, ...]
    text_encoder: Path | None
    clip_type: str | None
    model_loader: ModelLoader = ModelLoader.COMPONENTS


@dataclass(frozen=True)
class UpscalingConfig:
    models: tuple[Path, ...] = ()


@dataclass(frozen=True)
class VideoResources:
    diffusion: tuple[Path, ...]
    vae: Path
    text_encoder: Path
    clip_type: str = "wan"


@dataclass(frozen=True)
class WorkbenchConfig:
    path: Path
    comfyui: ComfyConfig
    resources: dict[Mode, ModeResources]
    output_dir: Path
    database: Path
    worker_timeout_seconds: float
    upscaling: UpscalingConfig = field(default_factory=UpscalingConfig)
    video_resources: dict[VideoModel, VideoResources] = field(default_factory=dict)
    video_worker_timeout_seconds: float = 1800


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> WorkbenchConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workbench YAML 顶层必须是映射")
    base = config_path.parent
    if base.name == "configs":
        base = base.parent
    comfy_raw = raw["comfyui"]
    resources_raw = raw.get("resources") or {}
    resources = {}
    for mode in Mode:
        if mode.value not in resources_raw:
            continue
        item = resources_raw[mode.value]
        model_loader = ModelLoader(item.get("model_loader", ModelLoader.COMPONENTS))
        if model_loader == ModelLoader.COMPONENTS:
            vae = tuple(_resolve(value, base) for value in item["vae"])
            text_encoder = _resolve(item["text_encoder"], base)
            clip_type = str(item["clip_type"])
        else:
            vae = ()
            text_encoder = None
            clip_type = None
        resources[mode] = ModeResources(
            diffusion=tuple(_resolve(value, base) for value in item["diffusion"]),
            vae=vae,
            text_encoder=text_encoder,
            clip_type=clip_type,
            model_loader=model_loader,
        )
    timeout = float(raw.get("worker_timeout_seconds", 300))
    if timeout <= 0:
        raise ValueError("worker_timeout_seconds 必须大于 0")
    video_timeout = float(raw.get("video_worker_timeout_seconds", 1800))
    if video_timeout <= 0:
        raise ValueError("video_worker_timeout_seconds 必须大于 0")
    video_resources_raw = raw.get("video_resources") or {}
    if not isinstance(video_resources_raw, dict):
        raise ValueError("video_resources 必须是映射")
    video_resources = {}
    for video_model in VideoModel:
        if video_model.value not in video_resources_raw:
            continue
        item = video_resources_raw[video_model.value]
        clip_type = str(item.get("clip_type", "wan"))
        if clip_type != "wan":
            raise ValueError(f"{video_model.value} clip_type 固定为 wan")
        video_resources[video_model] = VideoResources(
            diffusion=tuple(_resolve(value, base) for value in item["diffusion"]),
            vae=_resolve(item["vae"], base),
            text_encoder=_resolve(item["text_encoder"], base),
            clip_type=clip_type,
        )
    if not resources and not video_resources:
        raise ValueError("resources 或 video_resources 至少必须配置一种生成模式")
    upscaling_raw = raw.get("upscaling") or {}
    if not isinstance(upscaling_raw, dict):
        raise ValueError("upscaling 必须是映射")
    models_raw = upscaling_raw.get("models", ())
    if not isinstance(models_raw, (list, tuple)):
        raise ValueError("upscaling.models 必须是目录列表")
    return WorkbenchConfig(
        path=config_path,
        comfyui=ComfyConfig(
            root=_resolve(comfy_raw["root"], base),
            python=_resolve(comfy_raw["python"], base),
        ),
        resources=resources,
        output_dir=_resolve(raw.get("output_dir", "output"), base),
        database=_resolve(raw.get("database", ".cache/diffusion_workbench.sqlite3"), base),
        worker_timeout_seconds=timeout,
        upscaling=UpscalingConfig(
            models=tuple(_resolve(value, base) for value in models_raw)
        ),
        video_resources=video_resources,
        video_worker_timeout_seconds=video_timeout,
    )
