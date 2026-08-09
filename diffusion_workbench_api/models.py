from __future__ import annotations

import asyncio
import json
import struct
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from diffusion_workbench_core import Mode, ResourceKind

from .dependencies import get_core, get_model_info_store
from .jobs import ModeParam

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
