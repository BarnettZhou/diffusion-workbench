from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from diffusion_workbench_core import Mode, ResourceKind
from diffusion_workbench_core.domain import (
    IMAGE_UPSCALE_INTERPOLATIONS,
    LATENT_UPSCALE_INTERPOLATIONS,
    SAMPLERS,
    SCHEDULERS,
    UpscaleMethod,
)

from .dependencies import get_core
from .schemas import (
    AliasRequest,
    CreateJobsRequest,
    StatusResponse,
    job_to_response,
    resource_to_response,
    status_to_response,
)
from .service import resource_at, submit_jobs

router = APIRouter(prefix="/api/v1")

ModeParam = Mode
KindParam = Literal["diffusion", "vae"]
JobStatusParam = Literal["queued", "running", "completed", "failed", "cancelled"]


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/sampling-options")
async def sampling_options():
    """全部可用采样器/调度器,供客户端渲染选项;合法值与提交校验同源。"""
    return {
        "samplers": list(SAMPLERS),
        "schedulers": list(SCHEDULERS),
        "defaults": {"sampler": "euler", "scheduler": "simple"},
    }


@router.get("/modes")
async def list_modes(core=Depends(get_core)):
    """返回服务端实际启用的模式及其模型加载能力。"""

    return {
        "modes": [
            {
                "mode": mode.value,
                "model_loader": resources.model_loader.value,
                "requires_vae": resources.model_loader.value == "components",
            }
            for mode, resources in core.config.resources.items()
        ]
    }


@router.get("/upscale-options")
async def upscale_options():
    """图片放大的方法/插值/采样选项与默认值;静态取值直接来自 core domain。"""
    return {
        "methods": [method.value for method in UpscaleMethod],
        "image_interpolations": list(IMAGE_UPSCALE_INTERPOLATIONS),
        "latent_interpolations": list(LATENT_UPSCALE_INTERPOLATIONS),
        "samplers": list(SAMPLERS),
        "schedulers": list(SCHEDULERS),
        "defaults": {
            "method": UpscaleMethod.LATENT_HIRES.value,
            "scale": 2.0,
            "interpolation": "bislerp",
            "tile": 512,
            "overlap": 32,
            "steps": 9,
            "start_step": 4,
        },
    }


@router.get("/upscale-models")
async def upscale_models(core=Depends(get_core)):
    """服务端配置的放大模型列表;只公开 index/name/display_name,不含路径。"""
    items = await asyncio.to_thread(core.list_upscale_models)
    return {"models": [resource_to_response(item) for item in items]}


@router.get("/status", response_model=StatusResponse)
async def get_status(core=Depends(get_core)):
    raw = await asyncio.to_thread(core.runtime_status)
    return status_to_response(raw)


@router.get("/resources/{mode}/{kind}")
async def list_resources(mode: ModeParam, kind: KindParam, core=Depends(get_core)):
    items = await asyncio.to_thread(core.list_resources, Mode(mode), ResourceKind(kind))
    return {"resources": [resource_to_response(item) for item in items]}


@router.put("/resources/{mode}/{kind}/{index}/alias")
async def set_alias(
    mode: ModeParam,
    kind: KindParam,
    index: int,
    payload: AliasRequest,
    core=Depends(get_core),
):
    mode_enum = Mode(mode)
    kind_enum = ResourceKind(kind)
    try:
        item = await asyncio.to_thread(resource_at, core, mode_enum, kind_enum, index)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        await asyncio.to_thread(core.set_alias, mode_enum, kind_enum, item.path, payload.alias)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True}


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_jobs(payload: CreateJobsRequest, core=Depends(get_core)):
    try:
        jobs = await asyncio.to_thread(submit_jobs, core, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"jobs": [job_to_response(job) for job in jobs]}


@router.get("/jobs")
async def list_jobs(
    job_status: JobStatusParam | None = Query(default=None, alias="status"),
    mode: ModeParam | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
    core=Depends(get_core),
):
    try:
        jobs, next_cursor = await asyncio.to_thread(
            core.list_jobs,
            status=job_status,
            mode=Mode(mode) if mode else None,
            limit=limit,
            cursor=cursor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"jobs": [job_to_response(job) for job in jobs], "next_cursor": next_cursor}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, core=Depends(get_core)):
    try:
        job = await asyncio.to_thread(core.get_job, job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"job {job_id} 不存在") from None
    return job_to_response(job)


@router.post("/control/stop")
async def stop(core=Depends(get_core)):
    await asyncio.to_thread(core.stop)
    return {"accepted": True, "scope": "running-and-entire-queue"}


@router.post("/control/skip")
async def skip_current(core=Depends(get_core)):
    """跳过当前运行中的任务，保留队列其余任务。"""
    try:
        skipped_id = await asyncio.to_thread(core.skip_current)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if skipped_id is None:
        return {"accepted": False, "skipped_job_id": None, "reason": "no-cancellable-job"}
    return {"accepted": True, "skipped_job_id": skipped_id, "scope": "current-job-only"}
