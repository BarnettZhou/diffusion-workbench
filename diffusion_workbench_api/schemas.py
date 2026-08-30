from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from diffusion_workbench_core.domain import (
    SAMPLERS,
    SCHEDULERS,
    VideoJobRecord,
    JobRecord,
    Mode,
    ResourceItem,
    UpscaleSettings,
    VideoModel,
)


class UpscaleRequest(BaseModel):
    """任务级图片放大参数;结构校验在这里,合法组合以 core domain 为最终事实来源。"""

    enabled: bool = False
    method: Literal["resize", "upscale_model", "latent_hires"] = "latent_hires"
    scale: float = Field(default=2.0, gt=1, le=4, allow_inf_nan=False)
    interpolation: str = "bislerp"
    model_index: int | None = Field(default=None, ge=1)
    tile: int = 512
    overlap: int = 32
    steps: int = Field(default=9, ge=1, le=100)
    start_step: int = Field(default=4, ge=0)
    cfg: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sampler: str | None = None
    scheduler: str | None = None
    seed: int | None = Field(default=None, ge=0)


class CreateJobsRequest(BaseModel):
    mode: Mode = Mode.ZIT
    model_index: int = Field(ge=1)
    vae_index: int | None = Field(default=None, ge=1)
    # components 模式必传,checkpoint 模式必须为空(文本编码器已内嵌)
    text_encoder_index: int | None = Field(default=None, ge=1)
    prompt: str = Field(min_length=1, max_length=16_000)
    negative_prompt: str = Field(default="", max_length=16_000)
    width: int = 576
    height: int = 576
    steps: int = Field(default=8, ge=1, le=100)
    seed: int = Field(default=-1, ge=-1)
    count: int = Field(default=1, ge=1, le=32)
    cfg: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    sampler: str = "euler"
    scheduler: str = "simple"
    upscale: UpscaleRequest | None = None

    # 合法值以 core domain 的 SAMPLERS/SCHEDULERS 为唯一事实来源
    @field_validator("sampler")
    @classmethod
    def _check_sampler(cls, value: str) -> str:
        if value not in SAMPLERS:
            raise ValueError(f"不支持 sampler: {value}")
        return value

    @field_validator("scheduler")
    @classmethod
    def _check_scheduler(cls, value: str) -> str:
        if value not in SCHEDULERS:
            raise ValueError(f"不支持 scheduler: {value}")
        return value


class CreateEditJobsRequest(BaseModel):
    """Krea2 图像编辑任务参数;结构校验在这里,合法组合以 core domain 为最终事实来源。

    编辑模式复用 Krea2 的 diffusion/VAE/text encoder/clip_type 资源,
    客户端不能提交任何服务器路径;`input_image_id` 引用受控上传接口落盘的图片。
    编辑模式仅支持 resize / upscale_model 放大(latent_hires 由 core domain 拒绝)。
    """

    model_index: int = Field(ge=1)
    vae_index: int = Field(ge=1)
    # 文本编码器同样按 index 从 Krea2 配置的 text_encoder 目录/文件列表选择
    text_encoder_index: int = Field(ge=1)
    prompt: str = Field(min_length=1, max_length=16_000)
    negative_prompt: str = Field(default="", max_length=16_000)
    input_image_id: str = Field(min_length=1, max_length=64)
    # 双图编辑的第二张输入图;缺省为 None 表示单图编辑
    secondary_input_image_id: str | None = Field(default=None, min_length=1, max_length=64)
    width: int = 576
    height: int = 576
    steps: int = Field(default=8, ge=1, le=100)
    seed: int = Field(default=-1, ge=-1)
    count: int = Field(default=1, ge=1, le=32)
    cfg: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    sampler: str = "euler"
    scheduler: str = "simple"
    # 编辑专属参数;0 表示按原图尺寸,ref_boost=1.0 表示不强化。
    grounding_px: int = Field(default=768, ge=0, le=4096)
    ref_boost: float = Field(default=1.0, ge=0, le=1000, allow_inf_nan=False)
    upscale: UpscaleRequest | None = None

    # 合法值以 core domain 的 SAMPLERS/SCHEDULERS 为唯一事实来源
    @field_validator("sampler")
    @classmethod
    def _check_sampler(cls, value: str) -> str:
        if value not in SAMPLERS:
            raise ValueError(f"不支持 sampler: {value}")
        return value

    @field_validator("scheduler")
    @classmethod
    def _check_scheduler(cls, value: str) -> str:
        if value not in SCHEDULERS:
            raise ValueError(f"不支持 scheduler: {value}")
        return value


class CreateRebalanceJobsRequest(BaseModel):
    """Krea2 参考图重排任务参数;结构校验在这里,合法组合以 core domain 为最终事实来源。

    复用 Krea2 的 diffusion/VAE/text encoder/clip_type 资源,客户端不能提交任何服务器路径;
    `reference_image_ids` 引用受控上传接口(/edit/input-images)落盘的图片,1-4 张。
    `reference_image_tokens` 与参考图一一对应,缺省时全部按 "normal" 处理。
    """

    model_index: int = Field(ge=1)
    vae_index: int = Field(ge=1)
    # 文本编码器同样按 index 从 Krea2 配置的 text_encoder 目录/文件列表选择
    text_encoder_index: int = Field(ge=1)
    prompt: str = Field(min_length=1, max_length=16_000)
    negative_prompt: str = Field(default="", max_length=16_000)
    reference_image_ids: list[str] = Field(min_length=1, max_length=4)
    reference_image_tokens: (
        list[Literal["low", "normal", "high", "max"]] | None
    ) = None
    width: int = 576
    height: int = 576
    steps: int = Field(default=8, ge=1, le=100)
    seed: int = Field(default=-1, ge=-1)
    count: int = Field(default=1, ge=1, le=32)
    cfg: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    sampler: str = "euler"
    scheduler: str = "simple"

    # 合法值以 core domain 的 SAMPLERS/SCHEDULERS 为唯一事实来源
    @field_validator("sampler")
    @classmethod
    def _check_sampler(cls, value: str) -> str:
        if value not in SAMPLERS:
            raise ValueError(f"不支持 sampler: {value}")
        return value

    @field_validator("scheduler")
    @classmethod
    def _check_scheduler(cls, value: str) -> str:
        if value not in SCHEDULERS:
            raise ValueError(f"不支持 scheduler: {value}")
        return value


class AliasRequest(BaseModel):
    alias: str = Field(min_length=1, max_length=128)


class UpscaleResponse(BaseModel):
    enabled: bool
    method: str
    scale: float
    interpolation: str
    model_name: str | None
    tile: int
    overlap: int
    steps: int
    start_step: int
    cfg: float | None
    sampler: str | None
    scheduler: str | None
    seed: int | None


class JobResponse(BaseModel):
    id: str
    batch_id: str | None
    status: str
    mode: str
    seed: int
    width: int
    height: int
    steps: int
    sampler: str
    scheduler: str
    cfg: float
    model_name: str
    vae_name: str | None
    text_encoder_name: str | None = None
    prompt: str
    negative_prompt: str
    submitted_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float | None
    output_name: str
    image_url: str | None
    error: str | None
    upscale: UpscaleResponse
    upscaled_output_name: str | None
    upscaled_image_url: str | None
    # Krea2 图像编辑字段;非编辑任务保持 None。input_image_url 在编辑任务有输入图时
    # 指向 /api/v1/edit/input-images/{name},便于前端复用受控 URL。
    input_image_url: str | None = None
    secondary_input_image_url: str | None = None
    grounding_px: int | None = None
    ref_boost: float | None = None
    # Krea2 参考图重排字段;非 krea2-rebalance 任务保持空列表。URL 同样指向受控上传目录。
    reference_image_urls: list[str] = Field(default_factory=list)
    reference_image_tokens: list[str] = Field(default_factory=list)


def _error_summary(error: str | None) -> str | None:
    if not error:
        return None
    return error.splitlines()[0][:500]


def upscale_to_response(upscale: UpscaleSettings) -> UpscaleResponse:
    return UpscaleResponse(
        enabled=upscale.enabled,
        method=upscale.method.value,
        scale=upscale.scale,
        interpolation=upscale.interpolation,
        model_name=upscale.model.path.name if upscale.model else None,
        tile=upscale.tile,
        overlap=upscale.overlap,
        steps=upscale.steps,
        start_step=upscale.start_step,
        cfg=upscale.cfg,
        sampler=upscale.sampler,
        scheduler=upscale.scheduler,
        seed=upscale.seed,
    )


def job_to_response(job: JobRecord) -> JobResponse:
    # 放大失败/取消时原图可能已保存:URL 以文件实际存在为准,不只看 completed
    image_url = f"/api/v1/images/{job.id}" if job.output_path.is_file() else None
    upscaled_path = job.upscaled_output_path
    upscaled_image_url = (
        f"/api/v1/images/{job.id}/upscaled"
        if upscaled_path is not None and upscaled_path.is_file()
        else None
    )
    # 编辑任务的输入图由受控上传目录解析,文件名即 upload 接口返回的 id
    input_image_url = (
        f"/api/v1/edit/input-images/{job.input_image_path.name}"
        if job.input_image_path is not None
        else None
    )
    secondary_input_image_url = (
        f"/api/v1/edit/input-images/{job.secondary_input_image_path.name}"
        if job.secondary_input_image_path is not None
        else None
    )
    return JobResponse(
        id=job.id,
        batch_id=job.batch_id,
        status=job.status,
        mode=job.mode.value,
        seed=job.seed,
        width=job.width,
        height=job.height,
        steps=job.steps,
        sampler=job.sampler,
        scheduler=job.scheduler,
        cfg=job.cfg,
        model_name=job.model_path.name,
        vae_name=job.vae_path.name if job.vae_path else None,
        text_encoder_name=(
            job.text_encoder_path.name if job.text_encoder_path else None
        ),
        prompt=job.prompt,
        negative_prompt=job.negative_prompt,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        duration_seconds=job.duration_seconds,
        output_name=job.output_path.name,
        image_url=image_url,
        error=_error_summary(job.error),
        upscale=upscale_to_response(job.upscale),
        upscaled_output_name=upscaled_path.name if upscaled_path is not None else None,
        upscaled_image_url=upscaled_image_url,
        input_image_url=input_image_url,
        secondary_input_image_url=secondary_input_image_url,
        grounding_px=job.grounding_px,
        ref_boost=job.ref_boost,
        reference_image_urls=[
            f"/api/v1/edit/input-images/{path.name}"
            for path in job.reference_image_paths
        ],
        reference_image_tokens=list(job.reference_image_tokens),
    )


class ResourceItemResponse(BaseModel):
    index: int
    name: str
    display_name: str
    alias: str | None


def resource_to_response(item: ResourceItem) -> ResourceItemResponse:
    return ResourceItemResponse(
        index=item.index,
        name=item.path.name,
        display_name=item.display_name,
        alias=item.alias,
    )


class StatusResponse(BaseModel):
    queue: int
    running: str | None
    worker: str
    gpu: str
    memory: dict
    gpu_memory_used_gib: float | None
    gpu_memory_total_gib: float | None
    gpu_memory_percent: float | None
    gpu_utilization_percent: float | None
    loaded_resources: dict | None


def status_to_response(status: dict) -> StatusResponse:
    return StatusResponse(
        queue=int(status.get("queue", 0)),
        running=status.get("running"),
        worker=str(status.get("worker", "unknown")),
        gpu=str(status.get("gpu", "不可用")),
        memory=dict(status.get("memory") or {}),
        gpu_memory_used_gib=status.get("gpu_memory_used_gib"),
        gpu_memory_total_gib=status.get("gpu_memory_total_gib"),
        gpu_memory_percent=status.get("gpu_memory_percent"),
        gpu_utilization_percent=status.get("gpu_utilization_percent"),
        loaded_resources=_public_loaded_resources(status.get("loaded_resources")),
    )


def _public_loaded_resources(loaded: dict | None) -> dict | None:
    """Worker 已加载资源;模型、VAE、音频 VAE 和文本编码器只公开文件名。

    空 dict 表示 Worker 未加载任何资源,归一为 None。
    """
    if not loaded:
        return None
    return {
        key: (
            Path(str(value)).name
            if key in ("model", "vae", "audio_vae", "text_encoder") and value
            else value
        )
        for key, value in loaded.items()
    }


STEP_METRIC_KEYS = (
    "elapsed_seconds",
    "step_seconds",
    "seconds_per_step",
    "steps_per_second",
    "eta_seconds",
)

_ALLOWED_EVENT_KEYS = frozenset(
    {
        "type",
        "job_id",
        "status",
        "stage",
        "step",
        "total",
        "seed",
        "queued",
        "running",
        "sequence",
        "mime_type",
        "encoding",
        "width",
        "height",
        "data",
        *STEP_METRIC_KEYS,
        "artifact_type",
    }
)


def public_event(event: dict, event_id: int) -> dict:
    """把 Core 内部事件转成可公开的事件，隐藏本机路径和完整错误。"""
    result = {key: event[key] for key in _ALLOWED_EVENT_KEYS if key in event}
    # 视频产物走独立的 /videos 端点,此时不再设置 image_url;图片逻辑保持原样
    if (
        event.get("artifact_type") == "video"
        and event.get("output_path")
        and event.get("job_id")
    ):
        result["video_url"] = f"/api/v1/videos/{event['job_id']}"
    elif event.get("output_path") and event.get("job_id"):
        result["image_url"] = f"/api/v1/images/{event['job_id']}"
    if (
        event.get("status") == "completed"
        and event.get("upscaled_output_path")
        and event.get("job_id")
    ):
        result["upscaled_image_url"] = f"/api/v1/images/{event['job_id']}/upscaled"
    if "error" in event:
        result["error"] = _error_summary(str(event["error"]))
    result["event_id"] = event_id
    return result



class CreateVideoJobsRequest(BaseModel):
    """视频生成任务参数;结构校验在这里,合法组合以 core domain 为最终事实来源。

    input_image_id 引用受控上传接口(`POST /video/input-images`)落盘的图片,
    设置后为 I2V,不设置为 T2V;text_encoder 按 text_encoder_index 从服务端配置的
    目录/文件列表中选择(与图片模式一致)。
    """

    video_model: VideoModel
    model_index: int | None = Field(default=None, ge=1)
    high_model_index: int | None = Field(default=None, ge=1)
    low_model_index: int | None = Field(default=None, ge=1)
    vae_index: int = Field(ge=1)
    text_encoder_index: int | None = Field(default=None, ge=1)
    prompt: str = Field(min_length=1, max_length=16_000)
    negative_prompt: str = Field(default="", max_length=16_000)
    input_image_id: str | None = Field(default=None, max_length=64)
    last_frame_image_id: str | None = Field(default=None, max_length=64)
    reference_image_ids: list[str] = Field(default_factory=list, max_length=9)
    reference_video_ids: list[str] = Field(default_factory=list, max_length=3)
    reference_audio_ids: list[str] = Field(default_factory=list, max_length=3)
    width: int = 704
    height: int = 960
    duration_seconds: int = Field(default=5, ge=1)
    fps: int = Field(default=24, ge=1, le=120)
    steps: int = Field(default=20, ge=1, le=100)
    seed: int = Field(default=-1, ge=-1)
    count: int = Field(default=1, ge=1, le=8)
    cfg: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    shift: float = Field(default=8.0, ge=0, le=100, allow_inf_nan=False)
    latent_multiplier: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    sampler: str = "uni_pc"
    scheduler: str = "simple"

    # 合法值以 core domain 的 SAMPLERS/SCHEDULERS 为唯一事实来源
    @field_validator("sampler")
    @classmethod
    def _check_sampler(cls, value: str) -> str:
        if value not in SAMPLERS:
            raise ValueError(f"不支持 sampler: {value}")
        return value

    @field_validator("scheduler")
    @classmethod
    def _check_scheduler(cls, value: str) -> str:
        if value not in SCHEDULERS:
            raise ValueError(f"不支持 scheduler: {value}")
        return value


class VideoJobResponse(BaseModel):
    id: str
    batch_id: str | None
    status: str
    video_model: str
    generation_type: str
    seed: int
    width: int
    height: int
    duration_seconds: int
    fps: int
    length: int
    steps: int
    sampler: str
    scheduler: str
    cfg: float
    shift: float
    denoise: float
    latent_multiplier: float
    model_name: str
    vae_name: str
    text_encoder_name: str | None = None
    prompt: str
    negative_prompt: str
    submitted_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    elapsed_seconds: float | None
    output_name: str
    video_url: str | None
    input_image_url: str | None
    last_frame_image_url: str | None
    reference_image_urls: list[str]
    reference_video_urls: list[str]
    reference_audio_urls: list[str]
    error: str | None


def video_job_to_response(job: VideoJobRecord) -> VideoJobResponse:
    # 与图片一致:URL 以文件实际存在为准,不只看 completed
    video_url = f"/api/v1/videos/{job.id}" if job.output_path.is_file() else None
    # I2V 输入图片来自受控上传目录,文件名即上传接口返回的 id
    input_image_url = (
        f"/api/v1/video/input-images/{job.input_image_path.name}"
        if job.input_image_path is not None
        else None
    )
    last_frame_image_url = (
        f"/api/v1/video/input-images/{job.last_frame_image_path.name}"
        if job.last_frame_image_path is not None
        else None
    )
    reference_image_urls = [
        f"/api/v1/video/input-images/{path.name}"
        for path in job.reference_image_paths
    ]
    reference_video_urls = [
        f"/api/v1/video/input-videos/{path.name}"
        for path in job.reference_video_paths
    ]
    reference_audio_urls = [
        f"/api/v1/video/input-audios/{path.name}"
        for path in job.reference_audio_paths
    ]
    return VideoJobResponse(
        id=job.id,
        batch_id=job.batch_id,
        status=job.status,
        video_model=job.video_model.value,
        generation_type=job.generation_type,
        seed=job.seed,
        width=job.width,
        height=job.height,
        duration_seconds=job.duration_seconds,
        fps=job.fps,
        length=job.length,
        steps=job.steps,
        sampler=job.sampler,
        scheduler=job.scheduler,
        cfg=job.cfg,
        shift=job.shift,
        denoise=job.denoise,
        latent_multiplier=job.latent_multiplier,
        model_name=job.model_path.name,
        vae_name=job.vae_path.name,
        text_encoder_name=job.text_encoder_path.name,
        prompt=job.prompt,
        negative_prompt=job.negative_prompt,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        elapsed_seconds=job.elapsed_seconds,
        output_name=job.output_path.name,
        video_url=video_url,
        input_image_url=input_image_url,
        last_frame_image_url=last_frame_image_url,
        reference_image_urls=reference_image_urls,
        reference_video_urls=reference_video_urls,
        reference_audio_urls=reference_audio_urls,
        error=_error_summary(job.error),
    )