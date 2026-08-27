from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from diffusion_workbench_core import VideoModel

from .dependencies import get_album_manager, get_core, get_settings_store
from .schemas import CreateVideoJobsRequest, resource_to_response, video_job_to_response
from .service import (
    VIDEO_INPUT_EXTENSIONS,
    import_video_input_image,
    resolve_video_input_image,
    save_video_input_image,
    save_video_ref_audio,
    save_video_ref_video,
    submit_video_jobs,
    video_model_capabilities,
)

router = APIRouter(prefix="/api/v1")

VideoModelParam = VideoModel


class ImportAlbumImageRequest(BaseModel):
    """从相册目录导入 I2V 输入图片;id/dir 语义与相册端点一致,路径解析在服务端完成。"""

    id: str
    dir: str = "output"


class ImportJobImageRequest(BaseModel):
    """从生成/编辑任务导入 I2V 输入图片;job_id 引用服务端任务,客户端不提交任何路径。"""

    job_id: str


def _require_video_model(core, video_model: VideoModel) -> None:
    """video_model 未在服务端配置时统一 404,不暴露配置细节。"""
    if video_model not in core.config.video_resources:
        raise HTTPException(status_code=404, detail=f"未配置视频模型: {video_model.value}")


@router.get("/video/models")
async def list_video_models(core=Depends(get_core)):
    """服务端实际配置的视频模型列表。"""
    return {
        "video_models": [
            video_model_capabilities(video_model)
            for video_model in core.config.video_resources
        ]
    }


@router.get("/video/models/{video_model}/resources")
async def list_video_resources(video_model: VideoModelParam, core=Depends(get_core)):
    """指定视频模型可选的 diffusion/VAE/text encoder 资源,只公开 index/name,不含路径。"""
    video_model = VideoModel(video_model)
    _require_video_model(core, video_model)
    models = await asyncio.to_thread(core.list_video_models, video_model)
    vaes = await asyncio.to_thread(core.list_video_vaes, video_model)
    text_encoders = await asyncio.to_thread(core.list_video_text_encoders, video_model)
    return {
        "models": [resource_to_response(item) for item in models],
        "vaes": [resource_to_response(item) for item in vaes],
        "text_encoders": [resource_to_response(item) for item in text_encoders],
    }


@router.post("/video/input-images", status_code=status.HTTP_201_CREATED)
async def upload_video_input_image(request: Request, core=Depends(get_core)):
    """受控上传 I2V 输入图片(png/jpeg/webp ≤32MB),返回服务端生成的 id。

    客户端之后只能用这个 id 引用图片,不能提交任何服务器路径。
    """
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    data = await request.body()
    try:
        saved = await asyncio.to_thread(save_video_input_image, core, media_type, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    image_id = saved["id"]
    return {"id": image_id, "url": f"/api/v1/video/input-images/{image_id}"}


@router.post("/video/input-images/from-album", status_code=status.HTTP_201_CREATED)
async def import_video_input_from_album(
    payload: ImportAlbumImageRequest,
    core=Depends(get_core),
    manager=Depends(get_album_manager),
):
    """把相册里已存在的图片直接导入为 I2V 输入图片(服务端本地复制,不经客户端上传)。

    路径解析复用相册的受控逻辑:越界 400、目录/文件不存在 404。
    """
    index = await asyncio.to_thread(manager.index_for, payload.dir)
    path = await asyncio.to_thread(index.resolve, payload.id)
    try:
        saved = await asyncio.to_thread(import_video_input_image, core, path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    image_id = saved["id"]
    return {"id": image_id, "url": f"/api/v1/video/input-images/{image_id}"}


@router.post("/video/input-images/from-job", status_code=status.HTTP_201_CREATED)
async def import_video_input_from_job(
    payload: ImportJobImageRequest,
    core=Depends(get_core),
):
    """把生成/编辑任务的输出图直接导入为 I2V 输入图片(服务端本地复制,不经客户端上传)。

    job 不存在或输出文件缺失返回 404;与 from-album 同一套类型/大小校验。
    """
    try:
        job = await asyncio.to_thread(core.get_job, payload.job_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"job {payload.job_id} 不存在"
        ) from None
    if not job.output_path.is_file():
        raise HTTPException(status_code=404, detail="任务输出图片不存在")
    try:
        saved = await asyncio.to_thread(import_video_input_image, core, job.output_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    image_id = saved["id"]
    return {"id": image_id, "url": f"/api/v1/video/input-images/{image_id}"}


@router.get("/video/input-images/{image_id}")
async def get_video_input_image(image_id: str, core=Depends(get_core)):
    """按 id 读取受控上传的输入图片;id 校验与提交任务时一致。"""
    try:
        path = await asyncio.to_thread(resolve_video_input_image, core, image_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=VIDEO_INPUT_EXTENSIONS[path.suffix.lower()])


@router.post("/video/input-videos", status_code=status.HTTP_201_CREATED)
async def upload_video_ref_video(request: Request, core=Depends(get_core)):
    """受控上传 Ref2VA 参考视频(mp4/webm/mov ≤100MB),返回服务端生成的 id。

    只校验扩展名与大小;无法解码的文件在生成阶段以任务失败呈现。
    """
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    data = await request.body()
    try:
        saved = await asyncio.to_thread(save_video_ref_video, core, media_type, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    video_id = saved["id"]
    return {"id": video_id, "url": f"/api/v1/video/input-videos/{video_id}"}


@router.get("/video/input-videos/{video_id}")
async def get_video_ref_video(video_id: str, core=Depends(get_core)):
    """按 id 读取受控上传的参考视频;id 校验与提交任务时一致。"""
    try:
        path = await asyncio.to_thread(resolve_video_input_image, core, video_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=VIDEO_INPUT_EXTENSIONS[path.suffix.lower()])


@router.post("/video/input-audios", status_code=status.HTTP_201_CREATED)
async def upload_video_ref_audio(request: Request, core=Depends(get_core)):
    """受控上传 Ref2VA 参考音频(wav/mp3/flac/ogg/m4a ≤20MB),返回服务端生成的 id。

    只校验扩展名与大小;无法解码的文件在生成阶段以任务失败呈现。
    """
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    data = await request.body()
    try:
        saved = await asyncio.to_thread(save_video_ref_audio, core, media_type, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audio_id = saved["id"]
    return {"id": audio_id, "url": f"/api/v1/video/input-audios/{audio_id}"}


@router.get("/video/input-audios/{audio_id}")
async def get_video_ref_audio(audio_id: str, core=Depends(get_core)):
    """按 id 读取受控上传的参考音频;id 校验与提交任务时一致。"""
    try:
        path = await asyncio.to_thread(resolve_video_input_image, core, audio_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=VIDEO_INPUT_EXTENSIONS[path.suffix.lower()])


@router.post("/video/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_video_jobs(
    payload: CreateVideoJobsRequest,
    core=Depends(get_core),
    settings_store=Depends(get_settings_store),
):
    """提交视频生成任务;带 input_image_id 为 I2V,否则 T2V。"""
    ref2va_limits = (await asyncio.to_thread(settings_store.get)).get("ref2va_limits")
    try:
        jobs = await asyncio.to_thread(
            submit_video_jobs, core, payload, ref2va_limits
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"jobs": [video_job_to_response(job) for job in jobs]}


@router.get("/video/jobs/{job_id}")
async def get_video_job(job_id: str, core=Depends(get_core)):
    try:
        job = await asyncio.to_thread(core.get_video_job, job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"video job {job_id} 不存在") from None
    return video_job_to_response(job)
