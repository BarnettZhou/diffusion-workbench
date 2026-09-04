from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image

from diffusion_workbench_core import read_comfyui_metadata, read_generation_metadata

from .dependencies import get_album_manager, get_core
from .files import _public_metadata


def _public_video_metadata(job) -> dict:
    """脱敏的视频生成参数:只含生成参数与资源文件名,不含绝对路径。"""
    return {
        "video_model": job.video_model.value,
        "generation_type": job.generation_type,
        "prompt": job.prompt,
        "negative_prompt": job.negative_prompt,
        "width": job.width,
        "height": job.height,
        "duration_seconds": job.duration_seconds,
        "fps": job.fps,
        "length": job.length,
        "steps": job.steps,
        "seed": job.seed,
        "cfg": job.cfg,
        "shift": job.shift,
        "denoise": job.denoise,
        "latent_multiplier": job.latent_multiplier,
        "sampler": job.sampler,
        "scheduler": job.scheduler,
        "model_name": job.model_path.name,
        "vae_name": job.vae_path.name,
    }

router = APIRouter(prefix="/api/v1")

# 内置目录(输出目录)的固定 id,不允许改名/删除
BUILTIN_DIR_ID = "output"


def _validate_subdir(subdir: str) -> str:
    """校验子目录参数:只允许根目录下单级目录名,拒绝越界路径。"""
    normalized = subdir.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) != 1 or parts[0] in (".", ".."):
        raise HTTPException(status_code=422, detail="无效的子目录")
    return parts[0]


@dataclass
class AlbumEntry:
    relpath: str
    name: str
    mtime_ns: int
    size_bytes: int
    width: int
    height: int
    kind: str  # "image" | "video"


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
                for path in (*self._root.rglob("*.png"), *self._root.rglob("*.mp4")):
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    seen.add(path)
                    is_video = path.suffix.lower() == ".mp4"
                    if is_video:
                        # mp4 不做 PIL 尺寸解析,宽高记 0
                        width, height = 0, 0
                    else:
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
                            kind="video" if is_video else "image",
                        )
                    )
            for path in list(self._size_cache):
                if path not in seen:
                    del self._size_cache[path]
        entries.sort(key=lambda entry: (entry.mtime_ns, entry.relpath), reverse=True)
        return entries

    def subdirs(self) -> list[str]:
        """根目录下第一级子目录名(通常按日期分目录),按名称倒序(新的在前)。"""
        if not self._root.is_dir():
            return []
        names = [
            child.name
            for child in self._root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]
        names.sort(reverse=True)
        return names

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

    def __init__(
        self,
        output_dir: Path,
        store: AlbumDirStore,
        poster_dir: Path | None = None,
        thumbnail_dir: Path | None = None,
    ):
        self._output = Path(output_dir).resolve()
        self._store = store
        self._poster_dir = Path(poster_dir) if poster_dir is not None else None
        self._thumbnail_dir = Path(thumbnail_dir) if thumbnail_dir is not None else None
        # 缩略图总像素上限(宽×高)。按面积等比缩放:无论源图比例,新图总像素
        # 都尽量接近且不超过 max_pixels(直接约束面积,而不是约束最长边)。
        # 缓存键含 max_pixels,以后调整上限旧缓存自动失效。
        self._thumbnail_max_pixels = 160_000
        # JPEG 质量;80~85 是肉眼难辨损失的常用阈值,缩略图小尺寸下更省
        self._thumbnail_quality = 85
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

    def poster_for(self, dir_id: str, relpath: str) -> Path:
        """视频封面(首帧 JPEG):ffmpeg 抽取,按 (目录, 路径, mtime, size) 缓存。

        iOS(WebKit)不会为 preload=metadata 的 <video> 渲染首帧,相册封面只能
        用服务端生成的静态图。缓存键含 mtime 与大小,文件被替换后自动失效。
        """
        if self._poster_dir is None:
            raise HTTPException(status_code=503, detail="未配置视频封面缓存目录")
        index = self.index_for(dir_id)
        path = index.resolve(relpath)
        if path.suffix.lower() != ".mp4":
            raise HTTPException(status_code=404, detail="只有视频才有封面")
        stat = path.stat()
        key = f"{index.root}|{relpath}|{stat.st_mtime_ns}|{stat.st_size}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        cache_path = self._poster_dir / f"{digest}.jpg"
        if cache_path.is_file():
            return cache_path
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise HTTPException(
                status_code=503, detail="服务器未安装 ffmpeg,无法生成视频封面"
            )
        self._poster_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._poster_dir / f".{digest}.{uuid.uuid4().hex[:8]}.tmp.jpg"
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(temporary),
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0 or not temporary.is_file():
                detail = result.stderr.decode("utf-8", "replace").strip()
                raise HTTPException(
                    status_code=422,
                    detail=f"视频封面生成失败:{detail or '无法解码首帧'}",
                )
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return cache_path

    def thumbnail_for(self, dir_id: str, relpath: str) -> Path:
        """图片缩略图:首访懒生成,按 (目录, 路径, mtime, size, 缩放参数) 缓存。

        相册网格每页 60+ 张,直接下发原始 PNG 会浪费带宽与内存;缩略图限定
        200×200 像素(等比,总像素 ≤ 40_000)并转 JPEG 缓存到 ``thumbnail_dir``。
        源文件被替换后 mtime/size 变化 → 缓存键变化 → 自动重新生成。
        仅支持图片(PNG/JPEG/WebP),视频走 poster 端点,其他格式返回 404。
        """
        if self._thumbnail_dir is None:
            raise HTTPException(status_code=503, detail="未配置缩略图缓存目录")
        index = self.index_for(dir_id)
        path = index.resolve(relpath)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise HTTPException(status_code=404, detail="只有图片才有缩略图")
        stat = path.stat()
        # 缩放参数进缓存键,以后调整 max_pixels / quality 时旧缓存自然失效
        key = (
            f"{index.root}|{relpath}|{stat.st_mtime_ns}|{stat.st_size}|"
            f"{self._thumbnail_max_pixels}|{self._thumbnail_quality}"
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        cache_path = self._thumbnail_dir / f"{digest}.jpg"
        if cache_path.is_file():
            return cache_path
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._thumbnail_dir / f".{digest}.{uuid.uuid4().hex[:8]}.tmp.jpg"
        try:
            try:
                with Image.open(path) as image:
                    # RGBA → 白色背景合成后再转 RGB,避免 JPEG 丢弃 alpha 通道
                    # 后出现黑边;纯 RGB 图像 PIL 直接走原通道
                    if image.mode in ("RGBA", "LA") or (
                        image.mode == "P" and "transparency" in image.info
                    ):
                        background = Image.new("RGB", image.size, (255, 255, 255))
                        rgba = image.convert("RGBA")
                        background.paste(rgba, mask=rgba.split()[-1])
                        image = background
                    else:
                        image = image.convert("RGB")
                    # 按面积等比缩放:scale = √(max_pixels / 源像素),
                    # 新图总像素 ≤ max_pixels 且尽量接近,不再受「最长边」约束
                    # (thumbnail 那种方式对非正方形图会远低于 max_pixels)
                    width, height = image.size
                    source_pixels = width * height
                    if source_pixels > self._thumbnail_max_pixels:
                        scale = (self._thumbnail_max_pixels / source_pixels) ** 0.5
                        new_width = max(1, int(width * scale))
                        new_height = max(1, int(height * scale))
                        image = image.resize(
                            (new_width, new_height),
                            Image.Resampling.LANCZOS,
                        )
                    image.save(
                        temporary,
                        format="JPEG",
                        quality=self._thumbnail_quality,
                        optimize=True,
                    )
            except (OSError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"缩略图生成失败:{exc}",
                ) from exc
            if not temporary.is_file():
                raise HTTPException(status_code=422, detail="缩略图生成失败:无法写入")
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return cache_path


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


@router.get("/album/subdirs")
async def list_album_subdirs(
    dir: str = Query(default=BUILTIN_DIR_ID),
    manager: AlbumManager = Depends(get_album_manager),
):
    index = await asyncio.to_thread(manager.index_for, dir)
    return {"subdirs": await asyncio.to_thread(index.subdirs)}


@router.get("/album")
async def list_album(
    limit: int = Query(default=60, ge=1, le=200),
    cursor: str | None = None,
    dir: str = Query(default=BUILTIN_DIR_ID),
    subdir: str | None = None,
    manager: AlbumManager = Depends(get_album_manager),
):
    index = await asyncio.to_thread(manager.index_for, dir)
    entries = await asyncio.to_thread(index.scan)
    # subdir 限定只看某个一级子目录,目录内为深度查找(含更深层级)
    if subdir is not None:
        prefix = _validate_subdir(subdir) + "/"
        entries = [entry for entry in entries if entry.relpath.startswith(prefix)]
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
                "kind": entry.kind,
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
    core=Depends(get_core),
):
    index = await asyncio.to_thread(manager.index_for, dir)
    path = await asyncio.to_thread(index.resolve, relpath)
    # mp4 没有内嵌元数据,参数来自 jobs 表中的视频任务记录(按输出文件反查)
    if path.suffix.lower() == ".mp4":
        try:
            job = await asyncio.to_thread(
                core.find_video_job_by_output, path.parent.name, path.name
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="视频没有对应的生成记录") from None
        return _public_video_metadata(job)
    try:
        metadata = await asyncio.to_thread(read_generation_metadata, path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if metadata is not None:
        return _public_metadata(metadata)
    # 非本应用 PNG 回退解析 ComfyUI 原生 prompt 块(已是公开形状,直接返回)
    comfyui_metadata = await asyncio.to_thread(read_comfyui_metadata, path)
    if comfyui_metadata is None:
        raise HTTPException(status_code=404, detail="图片不包含生成元数据")
    return comfyui_metadata


@router.get("/album/image/{relpath:path}/poster")
async def album_image_poster(
    relpath: str,
    dir: str = Query(default=BUILTIN_DIR_ID),
    manager: AlbumManager = Depends(get_album_manager),
):
    poster = await asyncio.to_thread(manager.poster_for, dir, relpath)
    return FileResponse(poster, media_type="image/jpeg")


@router.get("/album/image/{relpath:path}/thumbnail")
async def album_image_thumbnail(
    relpath: str,
    dir: str = Query(default=BUILTIN_DIR_ID),
    manager: AlbumManager = Depends(get_album_manager),
):
    """图片缩略图(200×200 等比,JPEG 质量 85),首访懒生成后缓存。"""
    thumbnail = await asyncio.to_thread(manager.thumbnail_for, dir, relpath)
    return FileResponse(thumbnail, media_type="image/jpeg")


@router.get("/album/image/{relpath:path}")
async def album_image(
    relpath: str,
    dir: str = Query(default=BUILTIN_DIR_ID),
    manager: AlbumManager = Depends(get_album_manager),
):
    index = await asyncio.to_thread(manager.index_for, dir)
    path = await asyncio.to_thread(index.resolve, relpath)
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "image/png"
    return FileResponse(path, media_type=media_type)


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


@router.post("/album/batch-delete")
async def batch_delete_album_images(
    payload: dict,
    manager: AlbumManager = Depends(get_album_manager),
):
    """批量删除:逐个解析并删除,单个失败不影响其余;已不存在的文件视为删除成功(幂等)。"""
    dir = str(payload.get("dir") or BUILTIN_DIR_ID)
    relpaths = payload.get("relpaths")
    if not isinstance(relpaths, list) or not relpaths:
        raise HTTPException(status_code=422, detail="relpaths 必须是非空列表")
    if len(relpaths) > 200:
        raise HTTPException(status_code=422, detail="单次最多删除 200 个文件")
    if not all(isinstance(item, str) and item for item in relpaths):
        raise HTTPException(status_code=422, detail="relpaths 必须是字符串列表")
    index = await asyncio.to_thread(manager.index_for, dir)
    deleted: list[str] = []
    failed: list[dict[str, str]] = []
    for relpath in relpaths:
        try:
            path = await asyncio.to_thread(index.resolve, relpath)
        except HTTPException as exc:
            if exc.status_code == 404:
                deleted.append(relpath)  # 已被删除/移动,目标已达成
            else:
                failed.append({"relpath": relpath, "reason": str(exc.detail)})
            continue
        try:
            await asyncio.to_thread(path.unlink)
        except OSError as exc:
            failed.append({"relpath": relpath, "reason": str(exc)})
            continue
        deleted.append(relpath)
    return {"deleted": deleted, "failed": failed}
