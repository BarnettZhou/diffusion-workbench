from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .domain import MINIMAX_H3_MODELS, Mode, ModelLoader, VideoModel


@dataclass(frozen=True)
class ComfyConfig:
    root: Path
    python: Path


@dataclass(frozen=True)
class RemoteEncoderConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 50051
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 60.0
    fallback_to_local: bool = False
    encoder_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModeResources:
    diffusion: tuple[Path, ...]
    vae: tuple[Path, ...]
    # text encoder 候选目录/文件列表（与 diffusion/vae 同规则扫描第一层
    # *.safetensors/*.sft），提交时按 index 选择；checkpoint 模式为空元组。
    text_encoder: tuple[Path, ...]
    clip_type: str | None
    model_loader: ModelLoader = ModelLoader.COMPONENTS
    # Krea2 图像编辑专用 LoRA；仅 Krea2/Krea2-Edit 实际使用，未配置时为 None。
    edit_lora: Path | None = None
    # krea2 / zit 模式可选 LoRA 候选目录/文件列表（与 diffusion/vae 同规则扫描），
    # 提交时按 index 选择，最多 3 个；未配置时为空元组。
    loras: tuple[Path, ...] = ()


@dataclass(frozen=True)
class UpscalingConfig:
    models: tuple[Path, ...] = ()


@dataclass(frozen=True)
class VideoResources:
    diffusion: tuple[Path, ...]
    vae: tuple[Path, ...]
    # text encoder 候选目录/文件列表（与图片侧同规则扫描），提交时按 index 选择；
    # 兼容旧的单文件标量写法（load_config 自动包成单元素元组）。
    text_encoder: tuple[Path, ...]
    clip_type: str = "wan"
    audio_vae: Path | None = None
    # MiniMax H3 Turbo 专用蒸馏 LoRA；仅 minimax-h3-turbo 必须配置，其余类型为 None。
    turbo_lora: Path | None = None


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
    video_worker_timeout_seconds: float = 3600
    remote_encoder: RemoteEncoderConfig = field(default_factory=RemoteEncoderConfig)


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
            # text_encoder 兼容旧的单文件写法（标量）与新的目录/文件列表写法
            text_encoder_raw = item["text_encoder"]
            if not isinstance(text_encoder_raw, (list, tuple)):
                text_encoder_raw = (text_encoder_raw,)
            text_encoder = tuple(_resolve(value, base) for value in text_encoder_raw)
            clip_type = str(item["clip_type"])
            edit_lora_raw = item.get("edit_lora")
            edit_lora = _resolve(edit_lora_raw, base) if edit_lora_raw else None
            loras = tuple(_resolve(value, base) for value in item.get("loras", ()))
        else:
            vae = ()
            text_encoder = ()
            clip_type = None
            edit_lora = None
            loras = ()
        resources[mode] = ModeResources(
            diffusion=tuple(_resolve(value, base) for value in item["diffusion"]),
            vae=vae,
            text_encoder=text_encoder,
            clip_type=clip_type,
            model_loader=model_loader,
            edit_lora=edit_lora,
            loras=loras,
        )
    timeout = float(raw.get("worker_timeout_seconds", 3600))
    if timeout <= 0:
        raise ValueError("worker_timeout_seconds 必须大于 0")
    video_timeout = float(raw.get("video_worker_timeout_seconds", 3600))
    if video_timeout <= 0:
        raise ValueError("video_worker_timeout_seconds 必须大于 0")
    video_resources_raw = raw.get("video_resources") or {}
    if not isinstance(video_resources_raw, dict):
        raise ValueError("video_resources 必须是映射")
    video_resources = {}
    for video_model in VideoModel:
        # 历史遗留 minimax-h3 仅用于读取旧任务，不再接受配置
        if video_model == VideoModel.MINIMAX_H3:
            continue
        if video_model.value not in video_resources_raw:
            continue
        item = video_resources_raw[video_model.value]
        expected_clip_type = (
            "minimax" if video_model in MINIMAX_H3_MODELS else "wan"
        )
        clip_type = str(item.get("clip_type", expected_clip_type))
        if clip_type != expected_clip_type:
            raise ValueError(
                f"{video_model.value} clip_type 固定为 {expected_clip_type}"
            )
        audio_vae = item.get("audio_vae")
        if video_model in MINIMAX_H3_MODELS and not audio_vae:
            raise ValueError(f"{video_model.value} 必须配置 audio_vae")
        turbo_lora = item.get("turbo_lora")
        if video_model == VideoModel.MINIMAX_H3_TURBO and not turbo_lora:
            raise ValueError(f"{video_model.value} 必须配置 turbo_lora")
        # text_encoder 兼容旧的单文件写法（标量）与新的目录/文件列表写法
        video_te_raw = item["text_encoder"]
        if not isinstance(video_te_raw, (list, tuple)):
            video_te_raw = (video_te_raw,)
        video_resources[video_model] = VideoResources(
            diffusion=tuple(_resolve(value, base) for value in item["diffusion"]),
            vae=tuple(_resolve(value, base) for value in item["vae"]),
            text_encoder=tuple(_resolve(value, base) for value in video_te_raw),
            clip_type=clip_type,
            audio_vae=_resolve(audio_vae, base) if audio_vae else None,
            turbo_lora=_resolve(turbo_lora, base) if turbo_lora else None,
        )
    if not resources and not video_resources:
        raise ValueError("resources 或 video_resources 至少必须配置一种生成模式")
    remote_raw = raw.get("remote_encoder") or {}
    if not isinstance(remote_raw, dict):
        raise ValueError("remote_encoder 必须是映射")
    remote_encoder = RemoteEncoderConfig(
        enabled=bool(remote_raw.get("enabled", False)),
        host=str(remote_raw.get("host", "127.0.0.1")),
        port=int(remote_raw.get("port", 50051)),
        connect_timeout_seconds=float(remote_raw.get("connect_timeout_seconds", 5)),
        request_timeout_seconds=float(remote_raw.get("request_timeout_seconds", 60)),
        fallback_to_local=bool(remote_raw.get("fallback_to_local", False)),
        encoder_ids=tuple(str(item) for item in (remote_raw.get("encoder_ids") or ())),
    )
    if not (1 <= remote_encoder.port <= 65535):
        raise ValueError("remote_encoder.port 必须在 1 到 65535 之间")
    if remote_encoder.connect_timeout_seconds <= 0 or remote_encoder.request_timeout_seconds <= 0:
        raise ValueError("remote_encoder 超时必须大于 0")
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
        remote_encoder=remote_encoder,
    )
