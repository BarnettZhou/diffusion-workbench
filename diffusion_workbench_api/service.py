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
    UpscaleMethod,
    UpscaleSettings,
    VideoGenerationSettings,
    VideoModel,
    resolve_video_diffusion_pair,
)

from .schemas import CreateJobsRequest, CreateVideoJobsRequest, UpscaleRequest


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
    VideoModel.MINIMAX_H3: {
        "label": "MiniMax H3",
        "generation_types": ("t2v", "i2v", "r2v"),
        "requires_input_image": False,
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
    return {
        "video_model": video_model.value,
        "label": capabilities["label"],
        "generation_types": list(capabilities["generation_types"]),
        "requires_input_image": capabilities["requires_input_image"],
    }

# I2V 输入图片的受控上传:只允许 png/jpeg/webp,落盘到 cache 下固定目录,
# 客户端拿到的是服务端生成的 id(即文件名),提交任务时只能按 id 引用
VIDEO_INPUT_MEDIA_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
VIDEO_INPUT_EXTENSIONS = {ext: media for media, ext in VIDEO_INPUT_MEDIA_TYPES.items()}
MAX_VIDEO_INPUT_BYTES = 10 * 1024 * 1024
# id 即 "<uuid4 hex><扩展名>",先过白名单正则再做路径解析,双重防目录穿越
_VIDEO_INPUT_ID_RE = re.compile(r"^[0-9a-f]{32}\.(png|jpg|webp)$")


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


def _store_video_input(core: WorkbenchCore, ext: str, data: bytes) -> dict:
    if not data:
        raise ValueError("输入图片内容为空")
    if len(data) > MAX_VIDEO_INPUT_BYTES:
        raise ValueError("输入图片不能超过 10MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception:
        raise ValueError("输入图片内容无法识别为有效图片") from None
    directory = video_input_dir(core)
    directory.mkdir(parents=True, exist_ok=True)
    image_id = f"{uuid.uuid4().hex}{ext}"
    (directory / image_id).write_bytes(data)
    return {"id": image_id}


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
        vae = resource_at(core, mode, ResourceKind.VAE, payload.vae_index)
    else:
        if payload.vae_index is not None:
            raise ValueError(f"{mode.value} checkpoint 已内嵌 VAE")
        vae = None
    settings = GenerationSettings(
        mode=mode,
        model=model,
        vae=vae,
        text_encoder=fixed.text_encoder,
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



def video_resource_at(
    core: WorkbenchCore, video_model: VideoModel, kind: str, index: int
):
    """视频资源同样只允许按 index 从 Core 当前列表选择,拒绝任何客户端路径。

    kind 为 "diffusion" 或 "vae";video_model 未配置时返回空列表,同样报 index 未找到。
    """
    if kind == "diffusion":
        items = core.list_video_models(video_model)
    elif kind == "vae":
        items = core.list_video_vaes(video_model)
    else:
        raise LookupError(f"unknown video resource kind: {kind}")
    for item in items:
        if item.index == index:
            return item
    raise LookupError(f"video resource index {index} not found")


def submit_video_jobs(core: WorkbenchCore, payload: CreateVideoJobsRequest):
    """把请求映射为 VideoGenerationSettings;带 input_image_id 时为 I2V。"""
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
    # 输入图片只接受受控上传接口的 id,路径解析在服务端完成
    input_image = (
        resolve_video_input_image(core, payload.input_image_id)
        if payload.input_image_id
        else None
    )
    reference_image = (
        resolve_video_input_image(core, payload.reference_image_id)
        if payload.reference_image_id else None
    )
    if video_model == VideoModel.MINIMAX_H3 and input_image is not None and reference_image is not None:
        raise ValueError("MiniMax H3 首帧输入与参考图不能同时提供")
    capabilities = video_model_capabilities(video_model)
    if capabilities["requires_input_image"] and input_image is None:
        raise ValueError(f"{capabilities['label']} 必须提供输入图片")
    fixed = core.config.video_resources[video_model]
    sampler = payload.sampler
    if (
        video_model == VideoModel.WAN22_I2V_14B
        and "sampler" not in payload.model_fields_set
    ):
        sampler = "euler"
    if video_model == VideoModel.MINIMAX_H3 and "sampler" not in payload.model_fields_set:
        sampler = "res_multistep"
    fps = 24 if video_model == VideoModel.MINIMAX_H3 else payload.fps
    cfg = 1.0 if video_model == VideoModel.MINIMAX_H3 else payload.cfg
    shift = 12.0 if video_model == VideoModel.MINIMAX_H3 else payload.shift
    settings = VideoGenerationSettings(
        video_model=video_model,
        model=model,
        vae=vae,
        # text encoder 固定为服务端配置,不接受客户端指定
        text_encoder=fixed.text_encoder,
        audio_vae=fixed.audio_vae,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        input_image=input_image,
        reference_image=reference_image,
        width=payload.width,
        height=payload.height,
        duration_seconds=payload.duration_seconds,
        fps=fps,
        steps=payload.steps,
        seed=payload.seed,
        cfg=cfg,
        shift=shift,
        latent_multiplier=payload.latent_multiplier,
        sampler=sampler,
        scheduler=payload.scheduler,
    )
    return core.submit_video(settings, payload.count)
