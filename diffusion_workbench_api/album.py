from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image

from diffusion_workbench_core import read_generation_metadata

from .dependencies import get_album_manager
from .files import _public_metadata

router = APIRouter(prefix="/api/v1")

# 内置目录(输出目录)的固定 id,不允许改名/删除
BUILTIN_DIR_ID = "output"


@dataclass
class AlbumEntry:
    relpath: str
    name: str
    mtime_ns: int
    size_bytes: int
    width: int
    height: int


class AlbumIndex:
    """单个目录的图片索引,按文件扫描而非 jobs 表。

    图片可能被用户删除/移动,所以每次请求都重新 stat 目录;只有 PNG 尺寸
    解析结果按 (mtime_ns, size) 做内存缓存,文件变化或消失自动失效。
    """

    def __init__(self, output_dir: Path):
        self._root = Path(output_dir).resolve()
        self._lock = threading.Lock()
        self._size_cache: dict[Path, tuple[int, int, int, int]] = {}

    @property
    def root(self) -> Path:
        return self._root

    def scan(self) -> list[AlbumEntry]:
        """全部图片按修改时间倒序(同刻按路径名倒序保证稳定)。"""
        with self._lock:
            entries: list[AlbumEntry] = []
            seen: set[Path] = set()
            if self._root.is_dir():
                for path in self._root.rglob("*.png"):
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    seen.add(path)
                    cached = self._size_cache.get(path)
                    if (
                        cached
                        and cached[0] == stat.st_mtime_ns
                        and cached[1] == stat.st_size
                    ):
                        _, _, width, height = cached
                    else:
                        try:
                            with Image.open(path) as image:
                                width, height = image.size
                        except Exception:
                            continue  # 写入中或损坏的文件本次跳过
                        self._size_cache[path] = (
                            stat.st_mtime_ns,
                            stat.st_size,
                            width,
                            height,
                        )
                    entries.append(
                        AlbumEntry(
                            relpath=path.relative_to(self._root).as_posix(),
                            name=path.name,
                            mtime_ns=stat.st_mtime_ns,
                            size_bytes=stat.st_size,
                            width=width,
                            height=height,
                        )
                    )
            for path in list(self._size_cache):
                if path not in seen:
                    del self._size_cache[path]
        entries.sort(key=lambda entry: (entry.mtime_ns, entry.relpath), reverse=True)
        return entries

    def resolve(self, relpath: str) -> Path:
        path = (self._root / relpath).resolve()
        if not path.is_relative_to(self._root):
            raise HTTPException(status_code=400, detail="invalid image path")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="图片不存在或已被移动")
        return path


class AlbumDirStore:
    """用户添加的相册目录列表,JSON 持久化在 cache 目录(内置 output 不入库)。"""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._dirs: list[dict[str, str]] = []
        if self._path.is_file():
            try:
                stored = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = []
            if isinstance(stored, list):
                for item in stored:
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("id"), str)
                        and isinstance(item.get("name"), str)
                        and isinstance(item.get("path"), str)
                        and item["id"] != BUILTIN_DIR_ID
                    ):
                        self._dirs.append(
                            {"id": item["id"], "name": item["name"], "path": item["path"]}
                        )

    def list(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(entry) for entry in self._dirs]

    def get(self, dir_id: str) -> dict[str, str] | None:
        with self._lock:
            for entry in self._dirs:
                if entry["id"] == dir_id:
                    return dict(entry)
            return None

    def add(self, name: str, path: Path) -> dict[str, str]:
        with self._lock:
            existing = {entry["id"] for entry in self._dirs}
            dir_id = uuid.uuid4().hex[:8]
            while dir_id in existing or dir_id == BUILTIN_DIR_ID:
                dir_id = uuid.uuid4().hex[:8]
            entry = {"id": dir_id, "name": name, "path": str(path)}
            self._dirs.append(entry)
            self._save()
            return dict(entry)

    def rename(self, dir_id: str, name: str) -> dict[str, str] | None:
        with self._lock:
            for entry in self._dirs:
                if entry["id"] == dir_id:
                    entry["name"] = name
                    self._save()
                    return dict(entry)
            return None

    def remove(self, dir_id: str) -> bool:
        with self._lock:
            for position, entry in enumerate(self._dirs):
                if entry["id"] == dir_id:
                    del self._dirs[position]
                    self._save()
                    return True
            return False

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._dirs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)


class AlbumManager:
    """多目录相册:内置 output + 用户添加的目录,每个目录一个 AlbumIndex。"""

    def __init__(self, output_dir: Path, store: AlbumDirStore):
        self._output = Path(output_dir).resolve()
        self._store = store
        self._lock = threading.Lock()
        self._indexes: dict[str, AlbumIndex] = {}

    def dirs(self) -> list[dict]:
        """目录列表:内置 output 永远在最前,显示名为「默认」,不暴露绝对路径。"""
        entries = [{"id": BUILTIN_DIR_ID, "name": "默认", "builtin": True}]
        entries.extend(
            {
                "id": entry["id"],
                "name": entry["name"],
                "path": entry["path"],
                "builtin": False,
            }
            for entry in self._store.list()
        )
        return entries

    def index_for(self, dir_id: str) -> AlbumIndex:
        if dir_id == BUILTIN_DIR_ID:
            root = self._output
        else:
            entry = self._store.get(dir_id)
            if entry is None:
                raise HTTPException(status_code=404, detail="相册目录不存在")
            root = Path(entry["path"])
            if not root.is_dir():
                raise HTTPException(status_code=410, detail="目录已不存在或不可访问")
        with self._lock:
            index = self._indexes.get(dir_id)
            if index is None:
                index = AlbumIndex(root)
                self._indexes[dir_id] = index
            return index

    def add_dir(self, name: str, path: Path) -> dict:
        entry = self._store.add(name, path)
        return {"id": entry["id"], "name": entry["name"], "path": entry["path"], "builtin": False}

    def rename_dir(self, dir_id: str, name: str) -> dict:
        if dir_id == BUILTIN_DIR_ID:
            raise HTTPException(status_code=400, detail="内置目录不能改名")
        entry = self._store.rename(dir_id, name)
        if entry is None:
            raise HTTPException(status_code=404, detail="相册目录不存在")
        return {"id": entry["id"], "name": entry["name"], "path": entry["path"], "builtin": False}

    def remove_dir(self, dir_id: str) -> None:
        """从列表移除目录(不删除磁盘上的任何文件)。"""
        if dir_id == BUILTIN_DIR_ID:
            raise HTTPException(status_code=400, detail="内置目录不能删除")
        if not self._store.remove(dir_id):
            raise HTTPException(status_code=404, detail="相册目录不存在")
        with self._lock:
            self._indexes.pop(dir_id, None)


@router.get("/album/dirs")
async def list_album_dirs(manager: AlbumManager = Depends(get_album_manager)):
    return {"dirs": await asyncio.to_thread(manager.dirs)}


@router.post("/album/dirs", status_code=201)
async def add_album_dir(payload: dict, manager: AlbumManager = Depends(get_album_manager)):
    name = str(payload.get("name") or "").strip()
    raw_path = str(payload.get("path") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="目录名称不能为空")
    if not raw_path:
        raise HTTPException(status_code=422, detail="目录位置不能为空")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(status_code=422, detail="目录不存在或不是文件夹")
    return await asyncio.to_thread(manager.add_dir, name, path)


@router.put("/album/dirs/{dir_id}")
async def rename_album_dir(
    dir_id: str, payload: dict, manager: AlbumManager = Depends(get_album_manager)
):
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="目录名称不能为空")
    return await asyncio.to_thread(manager.rename_dir, dir_id, name)


@router.delete("/album/dirs/{dir_id}")
async def remove_album_dir(dir_id: str, manager: AlbumManager = Depends(get_album_manager)):
    await asyncio.to_thread(manager.remove_dir, dir_id)
    return {"deleted": dir_id}


@router.get("/album")
async def list_album(
    limit: int = Query(default=60, ge=1, le=200),
    cursor: str | None = None,
    dir: str = Query(default=BUILTIN_DIR_ID),
    manager: AlbumManager = Depends(get_album_manager),
):
    index = await asyncio.to_thread(manager.index_for, dir)
    entries = await asyncio.to_thread(index.scan)
    start = 0
    if cursor is not None:
        try:
            cursor_mtime_text, cursor_relpath = cursor.rsplit("|", 1)
            cursor_mtime = int(cursor_mtime_text)
        except ValueError:
            raise HTTPException(status_code=422, detail="无效的分页 cursor") from None
        for position, entry in enumerate(entries):
            if (entry.mtime_ns, entry.relpath) < (cursor_mtime, cursor_relpath):
                start = position
                break
        else:
            start = len(entries)
    page = entries[start : start + limit]
    next_cursor = None
    if page and start + limit < len(entries):
        last = page[-1]
        next_cursor = f"{last.mtime_ns}|{last.relpath}"
    return {
        "images": [
            {
                "id": entry.relpath,
                "name": entry.name,
                "mtime_ns": entry.mtime_ns,
                "size_bytes": entry.size_bytes,
                "width": entry.width,
                "height": entry.height,
            }
            for entry in page
        ],
        "next_cursor": next_cursor,
    }


@router.get("/album/image/{relpath:path}/metadata")
async def album_image_metadata(
    relpath: str,
    dir: str = Query(default=BUILTIN_DIR_ID),
    manager: AlbumManager = Depends(get_album_manager),
):
    index = await asyncio.to_thread(manager.index_for, dir)
    path = await asyncio.to_thread(index.resolve, relpath)
    try:
        metadata = await asyncio.to_thread(read_generation_metadata, path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if metadata is None:
        raise HTTPException(status_code=404, detail="图片不包含生成元数据")
    return _public_metadata(metadata)


@router.get("/album/image/{relpath:path}")
async def album_image(
    relpath: str,
    dir: str = Query(default=BUILTIN_DIR_ID),
    manager: AlbumManager = Depends(get_album_manager),
):
    index = await asyncio.to_thread(manager.index_for, dir)
    path = await asyncio.to_thread(index.resolve, relpath)
    return FileResponse(path, media_type="image/png")


@router.delete("/album/image/{relpath:path}")
async def delete_album_image(
    relpath: str,
    dir: str = Query(default=BUILTIN_DIR_ID),
    manager: AlbumManager = Depends(get_album_manager),
):
    """从本机删除目录中的图片(路径校验与读取一致,越界拒绝)。"""
    index = await asyncio.to_thread(manager.index_for, dir)
    path = await asyncio.to_thread(index.resolve, relpath)
    await asyncio.to_thread(path.unlink)
    return {"deleted": relpath}
