from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

from PIL import Image

from diffusion_workbench_core import (
    GenerationSettings,
    Mode,
    ModelLoader,
    ResourceKind,
    WorkbenchCore,
)
from diffusion_workbench_core.domain import (
    MINIMAX_H3_MODELS,
    UpscaleMethod,
    UpscaleSettings,
    VideoGenerationSettings,
    VideoModel,
    resolve_video_diffusion_pair,
)

from .schemas import (
    CreateEditJobsRequest,
    CreateJobsRequest,
    CreateRebalanceJobsRequest,
    CreateVideoJobsRequest,
    UpscaleRequest,
)


VIDEO_MODEL_CAPABILITIES = {
    VideoModel.WAN22_TI2V_5B: {
        "label": "Wan 2.2 TI2V-5B",
        "generation_types": ("t2v", "i2v"),
        "requires_input_image": False,
    },
    VideoModel.WAN22_I2V_14B: {
        "label": "Wan 2.2 I2V-14B",
        "generation_types": ("i2v",),
        "requires_input_image": True,
    },
    VideoModel.MINIMAX_H3_FL2VA: {
        "label": "MiniMax H3 FL2VA",
        "generation_types": ("t2v", "i2v"),
        "requires_input_image": False,
        "supports_last_frame": True,
    },
    VideoModel.MINIMAX_H3_REF2VA: {
        "label": "MiniMax H3 Ref2VA",
        "generation_types": ("r2v",),
        "requires_input_image": False,
        "reference_limits": {"images": 9, "videos": 3, "audios": 3, "total": 12},
    },
    VideoModel.MINIMAX_H3_TURBO: {
        "label": "MiniMax H3 FL2VA Turbo",
        "generation_types": ("t2v", "i2v"),
        "requires_input_image": False,
        "supports_last_frame": True,
    },
}


def video_model_capabilities(video_model: VideoModel) -> dict:
    """返回 API 可公开的视频模型能力,未知枚举采用保守的 I2V 约束。"""
    capabilities = VIDEO_MODEL_CAPABILITIES.get(
        video_model,
        {
            "label": video_model.value,
            "generation_types": ("i2v",),
            "requires_input_image": True,
        },
    )
    result = {
        "video_model": video_model.value,
        "label": capabilities["label"],
        "generation_types": list(capabilities["generation_types"]),
        "requires_input_image": capabilities["requires_input_image"],
    }
    for key in ("supports_last_frame", "reference_limits"):
        if key in capabilities:
            result[key] = capabilities[key]
    return result

# I2V 输入图片的受控上传:只允许 png/jpeg/webp,落盘到 cache 下固定目录,
# 客户端拿到的是服务端生成的 id(即文件名),提交任务时只能按 id 引用
VIDEO_INPUT_MEDIA_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
# Ref2VA 参考视频/音频的受控上传:同样只允许白名单类型,落盘到同一目录
VIDEO_REF_VIDEO_MEDIA_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}
VIDEO_REF_AUDIO_MEDIA_TYPES = {
    "audio/x-wav": ".wav",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
}
VIDEO_INPUT_EXTENSIONS = {
    ext: media
    for media_types in (
        VIDEO_INPUT_MEDIA_TYPES,
        VIDEO_REF_VIDEO_MEDIA_TYPES,
        VIDEO_REF_AUDIO_MEDIA_TYPES,
    )
    for media, ext in media_types.items()
}
MAX_VIDEO_INPUT_BYTES = 32 * 1024 * 1024
MAX_REF_VIDEO_BYTES = 100 * 1024 * 1024
MAX_REF_AUDIO_BYTES = 20 * 1024 * 1024
# id 即 "<uuid4 hex><扩展名>",先过白名单正则再做路径解析,双重防目录穿越
_VIDEO_INPUT_ID_RE = re.compile(
    r"^[0-9a-f]{32}\.(png|jpg|webp|mp4|webm|mov|wav|mp3|flac|ogg|m4a)$"
)


def video_input_dir(core: WorkbenchCore) -> Path:
    return core.config.database.parent / "video_inputs"


def save_video_input_image(core: WorkbenchCore, media_type: str, data: bytes) -> dict:
    """校验并保存一张上传的 I2V 输入图片,返回受控 id;不做任何客户端路径拼接。"""
    ext = VIDEO_INPUT_MEDIA_TYPES.get(media_type)
    if ext is None:
        raise ValueError("输入图片只支持 png/jpeg/webp")
    return _store_video_input(core, ext, data)


def import_video_input_image(core: WorkbenchCore, source: Path) -> dict:
    """把相册(或受控目录)里已存在的图片复制为 I2V 输入图片。

    与上传同一套校验(类型/大小/PIL 可解码);调用方负责 source 路径的受控解析。
    """
    ext = source.suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if ext not in VIDEO_INPUT_EXTENSIONS:
        raise ValueError("输入图片只支持 png/jpeg/webp")
    try:
        data = source.read_bytes()
    except OSError:
        raise ValueError("输入图片读取失败") from None
    return _store_video_input(core, ext, data)


def _store_video_input(
    core: WorkbenchCore,
    ext: str,
    data: bytes,
    *,
    max_bytes: int = MAX_VIDEO_INPUT_BYTES,
    label: str = "输入图片",
    verify_image: bool = True,
) -> dict:
    if not data:
        raise ValueError(f"{label}内容为空")
    if len(data) > max_bytes:
        raise ValueError(f"{label}不能超过 {max_bytes // 1024 // 1024}MB")
    if verify_image:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except Exception:
            raise ValueError(f"{label}内容无法识别为有效图片") from None
    directory = video_input_dir(core)
    directory.mkdir(parents=True, exist_ok=True)
    media_id = f"{uuid.uuid4().hex}{ext}"
    (directory / media_id).write_bytes(data)
    return {"id": media_id}


def save_video_ref_video(core: WorkbenchCore, media_type: str, data: bytes) -> dict:
    """校验并保存一段上传的 Ref2VA 参考视频;仅扩展名+大小校验,解码在 Worker 生成时进行。"""
    ext = VIDEO_REF_VIDEO_MEDIA_TYPES.get(media_type)
    if ext is None:
        raise ValueError("参考视频只支持 mp4/webm/mov")
    return _store_video_input(
        core, ext, data,
        max_bytes=MAX_REF_VIDEO_BYTES, label="参考视频", verify_image=False,
    )


def save_video_ref_audio(core: WorkbenchCore, media_type: str, data: bytes) -> dict:
    """校验并保存一段上传的 Ref2VA 参考音频;仅扩展名+大小校验,解码在 Worker 生成时进行。"""
    ext = VIDEO_REF_AUDIO_MEDIA_TYPES.get(media_type)
    if ext is None:
        raise ValueError("参考音频只支持 wav/mp3/flac/ogg/m4a")
    return _store_video_input(
        core, ext, data,
        max_bytes=MAX_REF_AUDIO_BYTES, label="参考音频", verify_image=False,
    )


def resolve_video_input_image(core: WorkbenchCore, image_id: str) -> Path:
    """把上传接口返回的 id 映射回受控目录内的文件;越界或不存在一律 LookupError。"""
    if not _VIDEO_INPUT_ID_RE.match(image_id):
        raise LookupError("输入图片不存在或已被删除")
    root = video_input_dir(core).resolve()
    path = (root / image_id).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise LookupError("输入图片不存在或已被删除")
    return path


def resource_at(core: WorkbenchCore, mode: Mode, kind: ResourceKind, index: int):
    """只允许从 Core 当前资源列表中按 index 选择，拒绝任何客户端路径。"""
    for item in core.list_resources(mode, kind):
        if item.index == index:
            return item
    raise LookupError(f"resource index {index} not found")


def upscale_model_at(core: WorkbenchCore, index: int):
    """放大模型同样只能按 index 从服务端目录选择，不接受任何客户端路径。"""
    for item in core.list_upscale_models():
        if item.index == index:
            return item
    raise LookupError(f"upscale model index {index} not found")


def build_upscale_settings(
    core: WorkbenchCore, payload: UpscaleRequest | None
) -> UpscaleSettings:
    """把请求里的放大参数映射为 Core 的 UpscaleSettings 并交给 domain 校验。"""
    if payload is None or not payload.enabled:
        return UpscaleSettings()
    model = None
    if payload.method == UpscaleMethod.UPSCALE_MODEL.value:
        if payload.model_index is None:
            raise ValueError("upscale_model 方法必须选择放大模型")
        model = upscale_model_at(core, payload.model_index)
    settings = UpscaleSettings(
        enabled=True,
        method=UpscaleMethod(payload.method),
        scale=payload.scale,
        interpolation=payload.interpolation,
        model=model,
        tile=payload.tile,
        overlap=payload.overlap,
        steps=payload.steps,
        start_step=payload.start_step,
        cfg=payload.cfg,
        sampler=payload.sampler,
        scheduler=payload.scheduler,
        seed=payload.seed,
    )
    settings.validate()
    return settings


def submit_jobs(core: WorkbenchCore, payload: CreateJobsRequest):
    mode = Mode(payload.mode)
    model = resource_at(core, mode, ResourceKind.DIFFUSION, payload.model_index)
    upscale = build_upscale_settings(core, payload.upscale)
    fixed = core.config.resources[mode]
    if fixed.model_loader == ModelLoader.COMPONENTS:
        if payload.vae_index is None:
            raise ValueError(f"{mode.value} 必须选择 VAE")
        if payload.text_encoder_index is None:
            raise ValueError(f"{mode.value} 必须选择文本编码器")
        vae = resource_at(core, mode, ResourceKind.VAE, payload.vae_index)
        text_encoder = resource_at(
            core, mode, ResourceKind.TEXT_ENCODER, payload.text_encoder_index
        ).path
    else:
        if payload.vae_index is not None:
            raise ValueError(f"{mode.value} checkpoint 已内嵌 VAE")
        if payload.text_encoder_index is not None:
            raise ValueError(f"{mode.value} checkpoint 已内嵌文本编码器")
        vae = None
        text_encoder = None
    settings = GenerationSettings(
        mode=mode,
        model=model,
        vae=vae,
        text_encoder=text_encoder,
        clip_type=fixed.clip_type,
        model_loader=fixed.model_loader,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        seed=payload.seed,
        cfg=payload.cfg,
        sampler=payload.sampler,
        scheduler=payload.scheduler,
        upscale=upscale,
    )
    return core.submit(settings, payload.count)


# Krea2 图像编辑的默认值与编辑模式启用判断:仅在 krea2 资源存在且配置了 edit_lora 时启用
EDIT_DEFAULTS = {
    "grounding_px": 768,
    "ref_boost": 1.0,
}


def edit_enabled(core: WorkbenchCore) -> bool:
    """编辑模式仅在 Krea2 资源存在且 edit_lora 已配置时启用,否则隐藏编辑入口。"""
    resources = core.config.resources.get(Mode.KREA2)
    return resources is not None and resources.edit_lora is not None


def submit_edit_jobs(core: WorkbenchCore, payload: CreateEditJobsRequest):
    """把请求映射为编辑 GenerationSettings;model/VAE 复用 Krea2 资源列表。

    编辑未启用(krea2 资源缺失或 edit_lora 为 None)时抛 LookupError,与未配置资源统一为 404。
    """
    if not edit_enabled(core):
        raise LookupError("未配置 krea2 图像编辑（resources.krea2.edit_lora）")
    model = resource_at(core, Mode.KREA2, ResourceKind.DIFFUSION, payload.model_index)
    vae = resource_at(core, Mode.KREA2, ResourceKind.VAE, payload.vae_index)
    text_encoder = resource_at(
        core, Mode.KREA2, ResourceKind.TEXT_ENCODER, payload.text_encoder_index
    ).path
    # 输入图只接受受控上传接口的 id,路径解析在服务端完成
    input_image = resolve_video_input_image(core, payload.input_image_id)
    secondary_input_image = (
        resolve_video_input_image(core, payload.secondary_input_image_id)
        if payload.secondary_input_image_id
        else None
    )
    fixed = core.config.resources[Mode.KREA2]
    settings = GenerationSettings(
        mode=Mode.KREA2_EDIT,
        model=model,
        vae=vae,
        text_encoder=text_encoder,
        clip_type=fixed.clip_type,
        model_loader=fixed.model_loader,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        input_image=input_image,
        secondary_input_image=secondary_input_image,
        upscale=build_upscale_settings(core, payload.upscale),
        grounding_px=payload.grounding_px,
        ref_boost=payload.ref_boost,
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        seed=payload.seed,
        cfg=payload.cfg,
        sampler=payload.sampler,
        scheduler=payload.scheduler,
    )
    return core.submit(settings, payload.count)


# Krea2 参考图重排的默认值与启用判断:仅需 krea2 资源配置(不依赖 edit_lora),
# 运行时要求 ComfyUI 安装 ComfyUI-Conditioning-Rebalance 自定义节点包。
REBALANCE_DEFAULTS = {
    "token_tier": "normal",
    "token_tiers": ["low", "normal", "high", "max"],
    "max_reference_images": 4,
}


def rebalance_enabled(core: WorkbenchCore) -> bool:
    """参考图重排模式在 Krea2 资源存在时启用(节点包缺失由 Worker 在运行时报错)。"""
    return core.config.resources.get(Mode.KREA2) is not None


def submit_rebalance_jobs(core: WorkbenchCore, payload: CreateRebalanceJobsRequest):
    """把请求映射为参考图重排 GenerationSettings;model/VAE 复用 Krea2 资源列表。

    未启用(krea2 资源缺失)时抛 LookupError,与未配置资源统一为 404。
    """
    if not rebalance_enabled(core):
        raise LookupError("未配置 krea2 资源（resources.krea2）")
    model = resource_at(core, Mode.KREA2, ResourceKind.DIFFUSION, payload.model_index)
    vae = resource_at(core, Mode.KREA2, ResourceKind.VAE, payload.vae_index)
    text_encoder = resource_at(
        core, Mode.KREA2, ResourceKind.TEXT_ENCODER, payload.text_encoder_index
    ).path
    # 参考图只接受受控上传接口的 id,路径解析在服务端完成
    reference_images = tuple(
        resolve_video_input_image(core, image_id)
        for image_id in payload.reference_image_ids
    )
    fixed = core.config.resources[Mode.KREA2]
    settings = GenerationSettings(
        mode=Mode.KREA2_REBALANCE,
        model=model,
        vae=vae,
        text_encoder=text_encoder,
        clip_type=fixed.clip_type,
        model_loader=fixed.model_loader,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        reference_images=reference_images,
        reference_image_tokens=tuple(payload.reference_image_tokens or ()),
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        seed=payload.seed,
        cfg=payload.cfg,
        sampler=payload.sampler,
        scheduler=payload.scheduler,
    )
    return core.submit(settings, payload.count)



def video_resource_at(
    core: WorkbenchCore, video_model: VideoModel, kind: str, index: int
):
    """视频资源同样只允许按 index 从 Core 当前列表选择,拒绝任何客户端路径。

    kind 为 "diffusion"、"vae" 或 "text_encoder";video_model 未配置时返回空列表,
    同样报 index 未找到。
    """
    if kind == "diffusion":
        items = core.list_video_models(video_model)
    elif kind == "vae":
        items = core.list_video_vaes(video_model)
    elif kind == "text_encoder":
        items = core.list_video_text_encoders(video_model)
    else:
        raise LookupError(f"unknown video resource kind: {kind}")
    for item in items:
        if item.index == index:
            return item
    raise LookupError(f"video resource index {index} not found")


def submit_video_jobs(
    core: WorkbenchCore,
    payload: CreateVideoJobsRequest,
    ref2va_limits: dict | None = None,
):
    """把请求映射为 VideoGenerationSettings;带 input_image_id 时为 I2V。

    ref2va_limits 为设置页配置的 Ref2VA 输入上限,服务端在此强制,防止客户端绕过。
    """
    video_model = VideoModel(payload.video_model)
    if video_model not in core.config.video_resources:
        raise LookupError(f"未配置视频模型: {video_model.value}")
    if video_model == VideoModel.WAN22_I2V_14B and (
        payload.high_model_index is not None or payload.low_model_index is not None
    ):
        if payload.high_model_index is None or payload.low_model_index is None:
            raise ValueError("Wan I2V-14B 必须同时选择 high-noise 和 low-noise 模型")
        high_model = video_resource_at(
            core, video_model, "diffusion", payload.high_model_index
        )
        low_model = video_resource_at(
            core, video_model, "diffusion", payload.low_model_index
        )
        expected_high, expected_low = resolve_video_diffusion_pair(
            video_model, high_model.path
        )
        if high_model.path.resolve() != expected_high:
            raise ValueError("high_model_index 必须引用 high_noise 模型")
        if expected_low is None or low_model.path.resolve() != expected_low:
            raise ValueError("high-noise 与 low-noise 模型不匹配")
        model = high_model
    else:
        if payload.model_index is None:
            field = (
                "high_model_index/low_model_index"
                if video_model == VideoModel.WAN22_I2V_14B
                else "model_index"
            )
            raise ValueError(f"必须提供 {field}")
        # 兼容旧客户端：14B 仍可由任一专家的 model_index 自动解析配对。
        model = video_resource_at(core, video_model, "diffusion", payload.model_index)
    vae = video_resource_at(core, video_model, "vae", payload.vae_index)
    # text encoder 与图片侧一致:目录+index 选择,缺省报 422(与 submit_jobs 语义一致)
    if payload.text_encoder_index is None:
        raise ValueError(f"{video_model.value} 必须选择文本编码器")
    text_encoder = video_resource_at(
        core, video_model, "text_encoder", payload.text_encoder_index
    ).path
    # 输入媒体只接受受控上传接口的 id,路径解析在服务端完成
    input_image = (
        resolve_video_input_image(core, payload.input_image_id)
        if payload.input_image_id
        else None
    )
    last_frame_image = (
        resolve_video_input_image(core, payload.last_frame_image_id)
        if payload.last_frame_image_id
        else None
    )
    reference_images = tuple(
        resolve_video_input_image(core, image_id)
        for image_id in payload.reference_image_ids
    )
    reference_videos = tuple(
        resolve_video_input_image(core, video_id)
        for video_id in payload.reference_video_ids
    )
    reference_audios = tuple(
        resolve_video_input_image(core, audio_id)
        for audio_id in payload.reference_audio_ids
    )
    if video_model == VideoModel.MINIMAX_H3_REF2VA and ref2va_limits:
        if len(reference_images) > int(ref2va_limits["max_images"]):
            raise ValueError("参考图数量超过设置页配置的 Ref2VA 上限")
        if len(reference_videos) > int(ref2va_limits["max_videos"]):
            raise ValueError("参考视频数量超过设置页配置的 Ref2VA 上限")
        if len(reference_audios) > int(ref2va_limits["max_audios"]):
            raise ValueError("参考音频数量超过设置页配置的 Ref2VA 上限")
    capabilities = video_model_capabilities(video_model)
    if capabilities["requires_input_image"] and input_image is None:
        raise ValueError(f"{capabilities['label']} 必须提供输入图片")
    fixed = core.config.video_resources[video_model]
    sampler = payload.sampler
    scheduler = payload.scheduler
    steps = payload.steps
    if (
        video_model == VideoModel.WAN22_I2V_14B
        and "sampler" not in payload.model_fields_set
    ):
        sampler = "euler"
    if video_model == VideoModel.MINIMAX_H3_TURBO:
        # turbo 配方默认 8 步 euler + beta 调度;sigma shift 不生效,记录为 0
        if "steps" not in payload.model_fields_set:
            steps = 8
        if "sampler" not in payload.model_fields_set:
            sampler = "euler"
        if "scheduler" not in payload.model_fields_set:
            scheduler = "beta"
    elif video_model in MINIMAX_H3_MODELS and "sampler" not in payload.model_fields_set:
        sampler = "res_multistep"
    fps = 24 if video_model in MINIMAX_H3_MODELS else payload.fps
    cfg = 1.0 if video_model in MINIMAX_H3_MODELS else payload.cfg
    if video_model == VideoModel.MINIMAX_H3_TURBO:
        shift = 0.0
    else:
        shift = 12.0 if video_model in MINIMAX_H3_MODELS else payload.shift
    settings = VideoGenerationSettings(
        video_model=video_model,
        model=model,
        vae=vae,
        # text encoder 由客户端按 index 从配置的目录列表中选择
        text_encoder=text_encoder,
        audio_vae=fixed.audio_vae,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        input_image=input_image,
        last_frame_image=last_frame_image,
        reference_images=reference_images,
        reference_videos=reference_videos,
        reference_audios=reference_audios,
        width=payload.width,
        height=payload.height,
        duration_seconds=payload.duration_seconds,
        fps=fps,
        steps=steps,
        seed=payload.seed,
        cfg=cfg,
        shift=shift,
        latent_multiplier=payload.latent_multiplier,
        sampler=sampler,
        scheduler=scheduler,
    )
    return core.submit_video(settings, payload.count)
