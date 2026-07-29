from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .domain import Mode


@dataclass(frozen=True)
class ComfyConfig:
    root: Path
    python: Path


@dataclass(frozen=True)
class ModeResources:
    diffusion: tuple[Path, ...]
    vae: tuple[Path, ...]
    text_encoder: Path
    clip_type: str


@dataclass(frozen=True)
class UpscalingConfig:
    models: tuple[Path, ...] = ()


@dataclass(frozen=True)
class WorkbenchConfig:
    path: Path
    comfyui: ComfyConfig
    resources: dict[Mode, ModeResources]
    output_dir: Path
    database: Path
    worker_timeout_seconds: float
    upscaling: UpscalingConfig = field(default_factory=UpscalingConfig)


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
    resources = {}
    for mode in Mode:
        item = raw["resources"][mode.value]
        resources[mode] = ModeResources(
            diffusion=tuple(_resolve(value, base) for value in item["diffusion"]),
            vae=tuple(_resolve(value, base) for value in item["vae"]),
            text_encoder=_resolve(item["text_encoder"], base),
            clip_type=str(item["clip_type"]),
        )
    timeout = float(raw.get("worker_timeout_seconds", 300))
    if timeout <= 0:
        raise ValueError("worker_timeout_seconds 必须大于 0")
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
    )
