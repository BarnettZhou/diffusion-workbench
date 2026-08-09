from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from diffusion_workbench_core.domain import (
    SAMPLERS,
    SCHEDULERS,
    JobRecord,
    Mode,
    ResourceItem,
    UpscaleSettings,
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


def status_to_response(status: dict) -> StatusResponse:
    return StatusResponse(
        queue=int(status.get("queue", 0)),
        running=status.get("running"),
        worker=str(status.get("worker", "unknown")),
        gpu=str(status.get("gpu", "不可用")),
    )


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
    }
)


def public_event(event: dict, event_id: int) -> dict:
    """把 Core 内部事件转成可公开的事件，隐藏本机路径和完整错误。"""
    result = {key: event[key] for key in _ALLOWED_EVENT_KEYS if key in event}
    if event.get("output_path") and event.get("job_id"):
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
