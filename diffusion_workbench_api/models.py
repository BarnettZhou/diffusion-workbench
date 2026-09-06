from __future__ import annotations

import asyncio
import json
import struct
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from diffusion_workbench_core import Mode, ResourceKind, VideoModel

from .dependencies import get_core, get_model_info_store
from .jobs import ModeParam
from .video import _require_video_model

router = APIRouter(prefix="/api/v1")

# safetensors 头部 dtype → 量化标签;量化 dtype 优先于全精度 dtype
QUANT_DTYPES = {"F8_E4M3": "fp8", "F8_E5M2": "fp8", "I8": "int8"}
PRECISION_DTYPES = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}
# safetensors 头部大小上限,防止异常文件读爆内存
MAX_HEADER_BYTES = 256 * 1024 * 1024
# 头部扫描结果缓存:path → (size, mtime_ns, quant)
_SCAN_CACHE: dict[str, tuple[int, int, str | None]] = {}
_SCAN_LOCK = threading.Lock()
COVER_MEDIA_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
COVER_EXTENSIONS = {ext: media for media, ext in COVER_MEDIA_TYPES.items()}
MAX_COVER_BYTES = 10 * 1024 * 1024


def scan_safetensors_quant(path: Path) -> str | None:
    """读取 safetensors 头部 JSON,按基础权重张量的 dtype 推断量化方式。

    只读文件开头几 KB 的头部,不加载权重;结果按 (size, mtime) 缓存。
    非 safetensors 文件或读取失败时返回 None。
    """
    if path.suffix.lower() != ".safetensors":
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    cache_key = str(path)
    with _SCAN_LOCK:
        cached = _SCAN_CACHE.get(cache_key)
        if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            return cached[2]

    quant = _read_safetensors_quant(path)
    with _SCAN_LOCK:
        _SCAN_CACHE[cache_key] = (stat.st_size, stat.st_mtime_ns, quant)
    return quant


def _read_safetensors_quant(path: Path) -> str | None:
    try:
        with open(path, "rb") as fh:
            (header_size,) = struct.unpack("<Q", fh.read(8))
            if header_size <= 0 or header_size > MAX_HEADER_BYTES:
                return None
            header = json.loads(fh.read(header_size))
    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(header, dict):
        return None

    quants: set[str] = set()
    precisions: set[str] = set()
    for key, info in header.items():
        if key == "__metadata__" or key.endswith(("_scale", ".comfy_quant")):
            continue
        if not isinstance(info, dict):
            continue
        dtype = info.get("dtype")
        if dtype in QUANT_DTYPES:
            quants.add(QUANT_DTYPES[dtype])
        elif dtype in PRECISION_DTYPES:
            precisions.add(PRECISION_DTYPES[dtype])
    if quants:
        return "+".join(sorted(quants))
    if precisions:
        return "+".join(sorted(precisions))
    return None


class ModelInfoStore:
    """模型附加信息:备注与量化类型存 model_info.json,封面存 covers/ 目录,都在 cache 下。"""

    def __init__(self, cache_dir: Path):
        self._dir = Path(cache_dir)
        self._path = self._dir / "model_info.json"
        self._cover_dir = self._dir / "covers"
        self._lock = threading.RLock()
        self._data: dict = {}
        if self._path.is_file():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, dict):
                self._data = raw

    def note(self, mode: str, name: str) -> str:
        with self._lock:
            return self._data.get(mode, {}).get(name, {}).get("note", "")

    def set_note(self, mode: str, name: str, note: str) -> None:
        with self._lock:
            self._data.setdefault(mode, {}).setdefault(name, {})["note"] = note
            self._save()

    def quant(self, mode: str, name: str) -> str | None:
        with self._lock:
            return self._data.get(mode, {}).get(name, {}).get("quant")

    def set_quant(self, mode: str, name: str, quant: str) -> None:
        with self._lock:
            self._data.setdefault(mode, {}).setdefault(name, {})["quant"] = quant
            self._save()

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self._path)

    def cover_path(self, mode: str, name: str) -> Path | None:
        with self._lock:
            for ext in COVER_EXTENSIONS:
                candidate = self._cover_dir / mode / f"{name}{ext}"
                if candidate.is_file():
                    return candidate
            return None

    def save_cover(self, mode: str, name: str, ext: str, data: bytes) -> Path:
        with self._lock:
            directory = self._cover_dir / mode
            directory.mkdir(parents=True, exist_ok=True)
            for old in directory.glob(f"{name}.*"):
                if old.suffix.lower() in COVER_EXTENSIONS:
                    old.unlink()
            path = directory / f"{name}{ext}"
            path.write_bytes(data)
            return path


class ModelInfoUpdate(BaseModel):
    alias: str | None = None
    note: str | None = None


def _find_model(core, mode: Mode, name: str):
    for item in core.list_resources(mode, ResourceKind.DIFFUSION):
        if item.path.name == name:
            return item
    raise HTTPException(status_code=404, detail=f"模型 {name} 不存在")


@router.get("/models/{mode}")
async def list_models(
    mode: ModeParam, core=Depends(get_core), store=Depends(get_model_info_store)
):
    mode_enum = Mode(mode)

    def collect():
        models = []
        for item in core.list_resources(mode_enum, ResourceKind.DIFFUSION):
            name = item.path.name
            try:
                size = item.path.stat().st_size
            except OSError:
                size = None
            has_cover = store.cover_path(mode, name) is not None
            models.append(
                {
                    "index": item.index,
                    "name": name,
                    "alias": item.alias,
                    "mode": mode,
                    "size_bytes": size,
                    "quant": store.quant(mode, name),
                    "note": store.note(mode, name),
                    "has_cover": has_cover,
                    "cover_url": f"/api/v1/models/{mode}/{name}/cover" if has_cover else None,
                }
            )
        return models

    return {"models": await asyncio.to_thread(collect)}


@router.get("/models/{mode}/{name}/cover")
async def get_cover(mode: ModeParam, name: str, store=Depends(get_model_info_store)):
    path = await asyncio.to_thread(store.cover_path, mode, name)
    if path is None:
        raise HTTPException(status_code=404, detail="暂无封面")
    return FileResponse(path, media_type=COVER_EXTENSIONS[path.suffix.lower()])


@router.put("/models/{mode}/{name}/cover")
async def upload_cover(
    mode: ModeParam,
    name: str,
    request: Request,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    mode_enum = Mode(mode)
    await asyncio.to_thread(_find_model, core, mode_enum, name)
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    ext = COVER_MEDIA_TYPES.get(media_type)
    if ext is None:
        raise HTTPException(status_code=415, detail="封面只支持 png/jpeg/webp")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="封面内容为空")
    if len(data) > MAX_COVER_BYTES:
        raise HTTPException(status_code=422, detail="封面不能超过 10MB")
    await asyncio.to_thread(store.save_cover, mode, name, ext, data)
    return {"ok": True, "cover_url": f"/api/v1/models/{mode}/{name}/cover"}


@router.post("/models/{mode}/{name}/quant")
async def fetch_model_quant(
    mode: ModeParam,
    name: str,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    """扫描 safetensors 头部识别量化方式,写入 model_info.json 后返回。"""
    mode_enum = Mode(mode)
    item = await asyncio.to_thread(_find_model, core, mode_enum, name)
    quant = await asyncio.to_thread(scan_safetensors_quant, item.path)
    if quant is None:
        raise HTTPException(status_code=422, detail="无法从文件识别量化方式")
    await asyncio.to_thread(store.set_quant, mode, name, quant)
    return {"ok": True, "quant": quant}


@router.put("/models/{mode}/{name}/info")
async def update_model_info(
    mode: ModeParam,
    name: str,
    payload: ModelInfoUpdate,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    mode_enum = Mode(mode)
    item = await asyncio.to_thread(_find_model, core, mode_enum, name)
    if payload.alias is not None:
        try:
            await asyncio.to_thread(
                core.set_alias, mode_enum, ResourceKind.DIFFUSION, item.path, payload.alias
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload.note is not None:
        await asyncio.to_thread(store.set_note, mode, name, payload.note)
    return {"ok": True}


def _lora_store_key(mode: str) -> str:
    """LoRA 的封面/备注/量化存储键;加后缀与同 mode 的 diffusion 模型隔离,避免同名互相覆盖。"""
    return f"{mode}-loras"


def _find_lora(core, mode: Mode, name: str):
    for item in core.list_resources(mode, ResourceKind.LORA):
        if item.path.name == name:
            return item
    raise HTTPException(status_code=404, detail=f"LoRA {name} 不存在")


@router.get("/loras/{mode}")
async def list_loras(
    mode: ModeParam, core=Depends(get_core), store=Depends(get_model_info_store)
):
    """krea2 / zit 可选 LoRA 画册卡片,字段与 models 端点一致(不含 quant);
    其他 mode 没有 LoRA 资源,返回空列表。"""
    mode_enum = Mode(mode)
    key = _lora_store_key(mode)

    def collect():
        loras = []
        for item in core.list_resources(mode_enum, ResourceKind.LORA):
            name = item.path.name
            try:
                size = item.path.stat().st_size
            except OSError:
                size = None
            has_cover = store.cover_path(key, name) is not None
            loras.append(
                {
                    "index": item.index,
                    "name": name,
                    "alias": item.alias,
                    "mode": mode,
                    "size_bytes": size,
                    "note": store.note(key, name),
                    "has_cover": has_cover,
                    "cover_url": f"/api/v1/loras/{mode}/{name}/cover" if has_cover else None,
                }
            )
        return loras

    return {"loras": await asyncio.to_thread(collect)}


@router.get("/loras/{mode}/{name}/cover")
async def get_lora_cover(mode: ModeParam, name: str, store=Depends(get_model_info_store)):
    path = await asyncio.to_thread(store.cover_path, _lora_store_key(mode), name)
    if path is None:
        raise HTTPException(status_code=404, detail="暂无封面")
    return FileResponse(path, media_type=COVER_EXTENSIONS[path.suffix.lower()])


@router.put("/loras/{mode}/{name}/cover")
async def upload_lora_cover(
    mode: ModeParam,
    name: str,
    request: Request,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    mode_enum = Mode(mode)
    await asyncio.to_thread(_find_lora, core, mode_enum, name)
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    ext = COVER_MEDIA_TYPES.get(media_type)
    if ext is None:
        raise HTTPException(status_code=415, detail="封面只支持 png/jpeg/webp")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="封面内容为空")
    if len(data) > MAX_COVER_BYTES:
        raise HTTPException(status_code=422, detail="封面不能超过 10MB")
    await asyncio.to_thread(store.save_cover, _lora_store_key(mode), name, ext, data)
    return {"ok": True, "cover_url": f"/api/v1/loras/{mode}/{name}/cover"}


@router.put("/loras/{mode}/{name}/info")
async def update_lora_info(
    mode: ModeParam,
    name: str,
    payload: ModelInfoUpdate,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    """LoRA 标题复用资源别名机制(ResourceKind.LORA),备注存 model_info.json。"""
    mode_enum = Mode(mode)
    item = await asyncio.to_thread(_find_lora, core, mode_enum, name)
    if payload.alias is not None:
        try:
            await asyncio.to_thread(
                core.set_alias, mode_enum, ResourceKind.LORA, item.path, payload.alias
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload.note is not None:
        await asyncio.to_thread(store.set_note, _lora_store_key(mode), name, payload.note)
    return {"ok": True}



def _find_video_model(core, video_model: VideoModel, name: str):
    for item in core.list_video_models(video_model):
        if item.path.name == name:
            return item
    raise HTTPException(status_code=404, detail=f"视频模型 {name} 不存在")


@router.get("/video-models/{video_model}")
async def list_video_model_cards(
    video_model: VideoModel,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    """视频模型卡片,字段与图片 models 端点一致;视频目录没有 alias 机制,固定为 None。

    ModelInfoStore 以 video_model 字符串作 mode key,与图片 mode 不会冲突。
    """
    _require_video_model(core, video_model)
    key = video_model.value

    def collect():
        models = []
        for item in core.list_video_models(video_model):
            name = item.path.name
            try:
                size = item.path.stat().st_size
            except OSError:
                size = None
            has_cover = store.cover_path(key, name) is not None
            models.append(
                {
                    "index": item.index,
                    "name": name,
                    "alias": None,
                    "mode": key,
                    "size_bytes": size,
                    "quant": store.quant(key, name),
                    "note": store.note(key, name),
                    "has_cover": has_cover,
                    "cover_url": (
                        f"/api/v1/video-models/{key}/{name}/cover" if has_cover else None
                    ),
                }
            )
        return models

    return {"models": await asyncio.to_thread(collect)}


@router.get("/video-models/{video_model}/{name}/cover")
async def get_video_model_cover(
    video_model: VideoModel, name: str, store=Depends(get_model_info_store)
):
    path = await asyncio.to_thread(store.cover_path, video_model.value, name)
    if path is None:
        raise HTTPException(status_code=404, detail="暂无封面")
    return FileResponse(path, media_type=COVER_EXTENSIONS[path.suffix.lower()])


@router.put("/video-models/{video_model}/{name}/cover")
async def upload_video_model_cover(
    video_model: VideoModel,
    name: str,
    request: Request,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    _require_video_model(core, video_model)
    await asyncio.to_thread(_find_video_model, core, video_model, name)
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    ext = COVER_MEDIA_TYPES.get(media_type)
    if ext is None:
        raise HTTPException(status_code=415, detail="封面只支持 png/jpeg/webp")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="封面内容为空")
    if len(data) > MAX_COVER_BYTES:
        raise HTTPException(status_code=422, detail="封面不能超过 10MB")
    await asyncio.to_thread(store.save_cover, video_model.value, name, ext, data)
    return {"ok": True, "cover_url": f"/api/v1/video-models/{video_model.value}/{name}/cover"}


@router.post("/video-models/{video_model}/{name}/quant")
async def fetch_video_model_quant(
    video_model: VideoModel,
    name: str,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    """扫描 safetensors 头部识别量化方式,写入 model_info.json 后返回。"""
    _require_video_model(core, video_model)
    item = await asyncio.to_thread(_find_video_model, core, video_model, name)
    quant = await asyncio.to_thread(scan_safetensors_quant, item.path)
    if quant is None:
        raise HTTPException(status_code=422, detail="无法从文件识别量化方式")
    await asyncio.to_thread(store.set_quant, video_model.value, name, quant)
    return {"ok": True, "quant": quant}


@router.put("/video-models/{video_model}/{name}/info")
async def update_video_model_info(
    video_model: VideoModel,
    name: str,
    payload: ModelInfoUpdate,
    core=Depends(get_core),
    store=Depends(get_model_info_store),
):
    """只支持 note:视频目录没有 alias 机制,payload 带 alias 时按 422 拒绝。"""
    _require_video_model(core, video_model)
    await asyncio.to_thread(_find_video_model, core, video_model, name)
    if payload.alias is not None:
        raise HTTPException(status_code=422, detail="视频模型不支持 alias")
    if payload.note is not None:
        await asyncio.to_thread(store.set_note, video_model.value, name, payload.note)
    return {"ok": True}
