from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .dependencies import get_album_manager, get_core
from .schemas import CreateEditJobsRequest, CreateRebalanceJobsRequest, job_to_response
from .service import (
    EDIT_DEFAULTS,
    REBALANCE_DEFAULTS,
    VIDEO_INPUT_EXTENSIONS,
    edit_enabled,
    import_video_input_image,
    rebalance_enabled,
    resolve_video_input_image,
    save_video_input_image,
    submit_edit_jobs,
    submit_rebalance_jobs,
)

router = APIRouter(prefix="/api/v1")


class ImportAlbumImageRequest(BaseModel):
    """从相册目录导入 Krea2 编辑输入图片;id/dir 语义与相册端点一致,路径解析在服务端完成。

    与视频版 ImportAlbumImageRequest 字段一致,前端可以共用表单组件;这里独立定义以便
    后端语义变化时各自演进。
    """

    id: str
    dir: str = "output"


class ImportJobImageRequest(BaseModel):
    """从生成/编辑任务导入输入图片;job_id 引用服务端任务,客户端不提交任何路径。"""

    job_id: str


@router.get("/edit/info")
async def edit_info(core=Depends(get_core)):
    """编辑/参考图重排可用性 + 默认参数;前端用 enabled 决定是否渲染对应入口。"""
    return {
        "enabled": edit_enabled(core),
        "defaults": dict(EDIT_DEFAULTS),
        "rebalance": {
            "enabled": rebalance_enabled(core),
            "defaults": dict(REBALANCE_DEFAULTS),
        },
    }


@router.post("/edit/input-images", status_code=status.HTTP_201_CREATED)
async def upload_edit_input_image(request: Request, core=Depends(get_core)):
    """受控上传编辑输入图片(png/jpeg/webp ≤32MB),返回服务端生成的 id。

    复用视频 I2V 输入图的同一受控目录与白名单校验;客户端之后只能按 id 引用,不能提交服务器路径。
    """
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    data = await request.body()
    try:
        saved = await asyncio.to_thread(save_video_input_image, core, media_type, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    image_id = saved["id"]
    return {"id": image_id, "url": f"/api/v1/edit/input-images/{image_id}"}


@router.post("/edit/input-images/from-album", status_code=status.HTTP_201_CREATED)
async def import_edit_input_from_album(
    payload: ImportAlbumImageRequest,
    core=Depends(get_core),
    manager=Depends(get_album_manager),
):
    """把相册里已存在的图片直接导入为编辑输入图片(服务端本地复制,不经客户端上传)。

    路径解析复用相册的受控逻辑:越界 400、目录/文件不存在 404。
    """
    index = await asyncio.to_thread(manager.index_for, payload.dir)
    path = await asyncio.to_thread(index.resolve, payload.id)
    try:
        saved = await asyncio.to_thread(import_video_input_image, core, path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    image_id = saved["id"]
    return {"id": image_id, "url": f"/api/v1/edit/input-images/{image_id}"}


@router.post("/edit/input-images/from-job", status_code=status.HTTP_201_CREATED)
async def import_edit_input_from_job(
    payload: ImportJobImageRequest,
    core=Depends(get_core),
):
    """把生成/编辑任务的输出图直接导入为编辑输入图片(服务端本地复制,不经客户端上传)。

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
    return {"id": image_id, "url": f"/api/v1/edit/input-images/{image_id}"}


@router.get("/edit/input-images/{image_id}")
async def get_edit_input_image(image_id: str, core=Depends(get_core)):
    """按 id 读取受控上传的编辑输入图片;id 校验与提交任务时一致。"""
    try:
        path = await asyncio.to_thread(resolve_video_input_image, core, image_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=VIDEO_INPUT_EXTENSIONS[path.suffix.lower()])


@router.post("/edit/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_edit_jobs(payload: CreateEditJobsRequest, core=Depends(get_core)):
    """提交 Krea2 图像编辑任务;LookupError→404,ValueError→422,FileNotFoundError→422。

    FileNotFoundError 来自 core.submit 在 edit_lora / input_image 缺失时抛出,按 422 处理以保持
    与 ValueError 一致的客户端语义(都是请求级参数问题);与 I2V 视频的 LookupError(图片不存在)
    区分开。
    """
    try:
        jobs = await asyncio.to_thread(submit_edit_jobs, core, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"jobs": [job_to_response(job) for job in jobs]}


@router.post("/edit/rebalance-jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_rebalance_jobs(payload: CreateRebalanceJobsRequest, core=Depends(get_core)):
    """提交 Krea2 参考图重排任务;LookupError→404,ValueError→422,FileNotFoundError→422。

    参考图按 /edit/input-images 返回的 id 引用,服务端解析路径;FileNotFoundError 来自
    core.submit 在参考图文件缺失时抛出,按 422 处理以保持与 ValueError 一致的客户端语义。
    """
    try:
        jobs = await asyncio.to_thread(submit_rebalance_jobs, core, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"jobs": [job_to_response(job) for job in jobs]}