from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from diffusion_workbench_core import read_generation_metadata

from .dependencies import get_core

router = APIRouter(prefix="/api/v1")


async def _resolve_output(core, job_id: str, *, upscaled: bool):
    """共享校验:任务存在、路径在 output_dir 内、文件确实存在。

    原图不再只以 status == completed 为条件:放大失败或取消时原图可能已保存,
    以受控路径和文件实际存在为准。放大图未启用返回 404,有记录但文件丢失返回 410。
    """
    try:
        job = await asyncio.to_thread(core.get_job, job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"job {job_id} 不存在") from None
    stored = job.upscaled_output_path if upscaled else job.output_path
    if stored is None:
        raise HTTPException(status_code=404, detail="任务未启用图片放大")
    output_root = core.config.output_dir.resolve()
    path = stored.resolve()
    if not path.is_relative_to(output_root):
        raise HTTPException(status_code=500, detail="invalid stored output path")
    if not path.is_file():
        if job.status in ("queued", "running"):
            raise HTTPException(
                status_code=404,
                detail="放大图尚未生成" if upscaled else "任务尚未完成",
            )
        raise HTTPException(
            status_code=410,
            detail="放大文件已丢失" if upscaled else "输出文件已丢失",
        )
    return job, path


@router.get("/images/{job_id}")
async def get_image(job_id: str, core=Depends(get_core)):
    """受控读取任务的原图 PNG,客户端不能传任意服务器路径。"""
    _, path = await _resolve_output(core, job_id, upscaled=False)
    return FileResponse(path, media_type="image/png")


@router.get("/images/{job_id}/upscaled")
async def get_upscaled_image(job_id: str, core=Depends(get_core)):
    """受控读取任务的放大图 PNG;未启用放大返回 404。"""
    _, path = await _resolve_output(core, job_id, upscaled=True)
    return FileResponse(path, media_type="image/png")


@router.get("/images/{job_id}/metadata")
async def get_image_metadata(job_id: str, core=Depends(get_core)):
    """读取 PNG 内嵌的生成参数(图片来源的唯一事实,SQLite 只做日志)。

    响应已脱敏:只含生成参数和资源文件名,不含绝对路径与 SHA-256。
    """
    _, path = await _resolve_output(core, job_id, upscaled=False)
    return await _read_public_metadata(path)


@router.get("/images/{job_id}/upscaled/metadata")
async def get_upscaled_metadata(job_id: str, core=Depends(get_core)):
    """读取放大图的内嵌元数据;artifact 反映放大后的真实尺寸与产物类型。"""
    _, path = await _resolve_output(core, job_id, upscaled=True)
    return await _read_public_metadata(path)


async def _read_public_metadata(path: Path):
    try:
        metadata = await asyncio.to_thread(read_generation_metadata, path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if metadata is None:
        raise HTTPException(status_code=404, detail="图片不包含生成元数据")
    return _public_metadata(metadata)


def _public_upscale(upscale: dict, upscale_model: dict) -> dict:
    """脱敏的放大参数:parameters.upscale 里的 model_path 是服务器绝对路径,只公开文件名。"""
    model_path = upscale.get("model_path")
    return {
        "enabled": upscale.get("enabled", False),
        "method": upscale.get("method"),
        "scale": upscale.get("scale"),
        "interpolation": upscale.get("interpolation"),
        "model_name": upscale_model.get("filename")
        or (Path(model_path).name if model_path else None),
        "tile": upscale.get("tile"),
        "overlap": upscale.get("overlap"),
        "steps": upscale.get("steps"),
        "start_step": upscale.get("start_step"),
        "cfg": upscale.get("cfg"),
        "sampler": upscale.get("sampler"),
        "scheduler": upscale.get("scheduler"),
        "seed": upscale.get("seed"),
    }


def _public_metadata(metadata: dict) -> dict:
    parameters = metadata.get("parameters", {})
    resources = metadata.get("resources", {})
    diffusion = resources.get("diffusion_model") or {}
    vae = resources.get("vae") or {}
    return {
        "mode": parameters.get("mode"),
        "prompt": parameters.get("prompt"),
        "negative_prompt": parameters.get("negative_prompt", ""),
        "negative_conditioning": parameters.get("negative_conditioning"),
        "width": parameters.get("width"),
        "height": parameters.get("height"),
        "steps": parameters.get("steps"),
        "seed": parameters.get("seed"),
        "cfg": parameters.get("cfg"),
        "sampler": parameters.get("sampler"),
        "scheduler": parameters.get("scheduler"),
        "model_name": diffusion.get("filename"),
        "vae_name": vae.get("filename"),
        # 当前文件的真实尺寸与产物类型;parameters.width/height 是首次生成尺寸
        "artifact": metadata.get("artifact"),
        "upscale": _public_upscale(
            parameters.get("upscale") or {}, resources.get("upscale_model") or {}
        ),
    }



@router.get("/videos/{job_id}")
async def get_video(job_id: str, core=Depends(get_core)):
    """受控读取视频任务的 mp4,路径校验规则与图片一致。"""
    try:
        job = await asyncio.to_thread(core.get_video_job, job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"video job {job_id} 不存在") from None
    output_root = core.config.output_dir.resolve()
    path = job.output_path.resolve()
    if not path.is_relative_to(output_root):
        raise HTTPException(status_code=500, detail="invalid stored output path")
    if not path.is_file():
        if job.status in ("queued", "running"):
            raise HTTPException(status_code=404, detail="任务尚未完成")
        raise HTTPException(status_code=410, detail="输出文件已丢失")
    return FileResponse(path, media_type="video/mp4")
