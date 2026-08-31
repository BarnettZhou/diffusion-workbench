import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .domain import (
    GenerationSettings,
    JobRecord,
    Mode,
    ModelLoader,
    ResourceKind,
    UpscaleSettings,
    VideoGenerationSettings,
    VideoJobRecord,
    VideoModel,
)


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS aliases (
    mode TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    alias TEXT NOT NULL,
    PRIMARY KEY (mode, kind, path),
    UNIQUE (mode, kind, alias)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    batch_id TEXT,
    status TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    duration_seconds REAL,
    output_path TEXT NOT NULL,
    upscaled_output_path TEXT,
    output_date TEXT NOT NULL,
    daily_index INTEGER NOT NULL,
    mode TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    vae TEXT NOT NULL,
    text_encoder TEXT NOT NULL,
    audio_vae TEXT,
    sampler TEXT NOT NULL,
    scheduler TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    cfg REAL NOT NULL,
    model_loader TEXT NOT NULL DEFAULT 'components',
    upscale_json TEXT NOT NULL DEFAULT '{}',
    job_kind TEXT NOT NULL DEFAULT 'image',
    input_image TEXT,
    secondary_input_image TEXT,
    grounding_px INTEGER,
    ref_boost REAL,
    reference_image TEXT,
    reference_inputs TEXT,
    video_duration_seconds INTEGER,
    fps INTEGER,
    frame_count INTEGER,
    denoise REAL NOT NULL DEFAULT 1.0,
    shift REAL NOT NULL DEFAULT 8.0,
    latent_multiplier REAL NOT NULL DEFAULT 1.0,
    error TEXT,
    UNIQUE (output_date, mode, daily_index)
);
"""


class JobStore:
    def __init__(self, database: Path, output_dir: Path):
        self.database = Path(database)
        self.output_dir = Path(output_dir)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connection() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
            }
            if "negative_prompt" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN negative_prompt TEXT NOT NULL DEFAULT ''"
                )
            if "upscaled_output_path" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN upscaled_output_path TEXT"
                )
            if "upscale_json" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN upscale_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "model_loader" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN model_loader TEXT NOT NULL DEFAULT 'components'"
                )
            migrations = {
                "job_kind": "TEXT NOT NULL DEFAULT 'image'",
                "input_image": "TEXT",
                "reference_image": "TEXT",
                "reference_inputs": "TEXT",
                "video_duration_seconds": "INTEGER",
                "fps": "INTEGER",
                "frame_count": "INTEGER",
                "denoise": "REAL NOT NULL DEFAULT 1.0",
                "shift": "REAL NOT NULL DEFAULT 8.0",
                "latent_multiplier": "REAL NOT NULL DEFAULT 1.0",
                "audio_vae": "TEXT",
                # Krea2 图像编辑专用列；旧库通过 ALTER TABLE 增量添加。
                "grounding_px": "INTEGER",
                "ref_boost": "REAL",
                # 双图编辑的第二张输入图；仅 edit-krea2 任务非空。
                "secondary_input_image": "TEXT",
                "text_encoder_source": "TEXT NOT NULL DEFAULT 'local'",
                "remote_text_encoder_id": "TEXT",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE jobs ADD COLUMN {name} {declaration}"
                    )

    def recover_incomplete_jobs(self) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'cancelled',
                    completed_at = COALESCE(completed_at, ?),
                    error = COALESCE(error, 'Workbench 上次退出时任务未完成')
                WHERE status IN ('queued', 'running')
                """,
                (datetime.now().astimezone().isoformat(),),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get_aliases(self, mode: Mode, kind: ResourceKind) -> dict[Path, str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT path, alias FROM aliases WHERE mode = ? AND kind = ?",
                (mode.value, kind.value),
            ).fetchall()
        return {Path(row["path"]): row["alias"] for row in rows}

    def set_alias(self, mode: Mode, kind: ResourceKind, path: Path, alias: str) -> None:
        alias = alias.strip()
        if not alias:
            raise ValueError("alias 不能为空")
        try:
            with self._lock, self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO aliases(mode, kind, path, alias) VALUES (?, ?, ?, ?)
                    ON CONFLICT(mode, kind, path) DO UPDATE SET alias = excluded.alias
                    """,
                    (mode.value, kind.value, str(path.resolve()), alias),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"alias {alias!r} 已被当前 {mode.value} {kind.value} 使用"
            ) from exc

    def prune_aliases(
        self, mode: Mode, kind: ResourceKind, live_paths: set[Path]
    ) -> None:
        """删除索引中文件已不存在于磁盘的别名记录(模型被删除后自动清理)。"""
        resolved = [str(path.resolve()) for path in live_paths]
        with self._lock, self._connection() as connection:
            if resolved:
                placeholders = ",".join("?" for _ in resolved)
                connection.execute(
                    f"""
                    DELETE FROM aliases
                    WHERE mode = ? AND kind = ? AND path NOT IN ({placeholders})
                    """,
                    (mode.value, kind.value, *resolved),
                )
            else:
                connection.execute(
                    "DELETE FROM aliases WHERE mode = ? AND kind = ?",
                    (mode.value, kind.value),
                )

    def create_jobs(
        self,
        settings: GenerationSettings,
        count: int,
        submitted_at: datetime | None = None,
    ) -> list[JobRecord]:
        settings.validate()
        if count <= 0:
            raise ValueError("任务数量必须大于 0")
        submitted_at = submitted_at or datetime.now().astimezone()
        output_date = submitted_at.date().isoformat()
        submitted_iso = submitted_at.isoformat()
        batch_id = str(uuid.uuid4()) if count > 1 else None
        jobs = []
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(MAX(daily_index), 0) AS value FROM jobs WHERE output_date = ? AND mode = ?",
                (output_date, settings.mode.value),
            ).fetchone()
            persisted_index = int(row["value"])
            file_index = self._highest_output_index(
                output_date, settings.mode.value, ".png"
            )
            next_index = max(persisted_index, file_index) + 1
            for offset in range(count):
                daily_index = next_index + offset
                job_id = str(uuid.uuid4())
                output_path = (
                    self.output_dir
                    / output_date
                    / f"{settings.mode.value}-{daily_index:05d}.png"
                ).resolve()
                upscaled_output_path = (
                    output_path.with_name(f"{output_path.stem}-upscale.png")
                    if settings.upscale.enabled
                    else None
                )
                is_edit = settings.mode == Mode.KREA2_EDIT
                input_image_value = (
                    str(settings.input_image.resolve())
                    if settings.input_image is not None
                    else None
                )
                secondary_input_image_value = (
                    str(settings.secondary_input_image.resolve())
                    if settings.secondary_input_image is not None
                    else None
                )
                # 仅编辑模式写入实际值；其他 mode 落 NULL，避免误传 768/1.0 默认值。
                grounding_px_value = int(settings.grounding_px) if is_edit else None
                ref_boost_value = float(settings.ref_boost) if is_edit else None
                # 仅 krea2-rebalance 模式写入参考图 JSON；其他 mode 落 NULL。
                reference_inputs_value = (
                    _rebalance_reference_json(settings)
                    if settings.mode == Mode.KREA2_REBALANCE
                    else None
                )
                values = (
                    job_id,
                    batch_id,
                    "queued",
                    submitted_iso,
                    str(output_path),
                    str(upscaled_output_path) if upscaled_output_path else None,
                    output_date,
                    daily_index,
                    settings.mode.value,
                    settings.prompt,
                    settings.negative_prompt,
                    str(settings.model.path.resolve()),
                    str(settings.vae.path.resolve()) if settings.vae else "",
                    str(settings.text_encoder.resolve()) if settings.text_encoder else "",
                    settings.sampler,
                    settings.scheduler,
                    settings.width,
                    settings.height,
                    settings.steps,
                    settings.seed,
                    settings.cfg,
                    settings.model_loader.value,
                    json.dumps(
                        settings.upscale.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "image",
                    input_image_value,
                    secondary_input_image_value,
                    grounding_px_value,
                    ref_boost_value,
                    reference_inputs_value,
                    settings.text_encoder_source,
                    settings.remote_text_encoder_id,
                )
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, batch_id, status, submitted_at, output_path, upscaled_output_path, output_date,
                        daily_index, mode, prompt, negative_prompt, model, vae, text_encoder, sampler,
                        scheduler, width, height, steps, seed, cfg, model_loader, upscale_json,
                        job_kind, input_image, secondary_input_image, grounding_px, ref_boost, reference_inputs,
                        text_encoder_source, remote_text_encoder_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                jobs.append(self.get_job(job_id, connection=connection))
        return jobs

    def create_video_jobs(
        self,
        settings: VideoGenerationSettings,
        count: int,
        submitted_at: datetime | None = None,
    ) -> list[VideoJobRecord]:
        settings.validate()
        if count <= 0:
            raise ValueError("任务数量必须大于 0")
        submitted_at = submitted_at or datetime.now().astimezone()
        output_date = submitted_at.date().isoformat()
        submitted_iso = submitted_at.isoformat()
        batch_id = str(uuid.uuid4()) if count > 1 else None
        jobs = []
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(MAX(daily_index), 0) AS value FROM jobs "
                "WHERE output_date = ? AND mode = ?",
                (output_date, settings.video_model.value),
            ).fetchone()
            persisted_index = int(row["value"])
            file_index = self._highest_output_index(
                output_date, settings.video_model.value, ".mp4"
            )
            next_index = max(persisted_index, file_index) + 1
            for offset in range(count):
                daily_index = next_index + offset
                job_id = str(uuid.uuid4())
                output_path = (
                    self.output_dir
                    / output_date
                    / f"{settings.video_model.value}-{daily_index:05d}.mp4"
                ).resolve()
                values = (
                    job_id,
                    batch_id,
                    "queued",
                    submitted_iso,
                    str(output_path),
                    output_date,
                    daily_index,
                    settings.video_model.value,
                    settings.prompt,
                    settings.negative_prompt,
                    str(settings.model.path.resolve()),
                    str(settings.vae.path.resolve()),
                    str(settings.text_encoder.resolve()),
                    str(settings.audio_vae.resolve()) if settings.audio_vae else None,
                    settings.sampler,
                    settings.scheduler,
                    settings.width,
                    settings.height,
                    settings.steps,
                    settings.seed,
                    settings.cfg,
                    "components",
                    "{}",
                    "video",
                    (
                        str(settings.input_image.resolve())
                        if settings.input_image is not None
                        else None
                    ),
                    _reference_inputs_json(settings),
                    settings.duration_seconds,
                    settings.fps,
                    settings.length,
                    settings.denoise,
                    settings.shift,
                    settings.latent_multiplier,
                )
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, batch_id, status, submitted_at, output_path, output_date,
                        daily_index, mode, prompt, negative_prompt, model, vae,
                        text_encoder, audio_vae, sampler, scheduler, width, height, steps, seed,
                        cfg, model_loader, upscale_json, job_kind, input_image, reference_inputs,
                        video_duration_seconds, fps, frame_count, denoise, shift,
                        latent_multiplier
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    values,
                )
                jobs.append(self.get_video_job(job_id, connection=connection))
        return jobs

    def _highest_output_index(
        self, output_date: str, prefix_value: str, extension: str
    ) -> int:
        directory = self.output_dir / output_date
        if not directory.is_dir():
            return 0
        prefix = f"{prefix_value}-"
        highest = 0
        for path in directory.glob(f"{prefix}*{extension}"):
            index_text = path.stem.removeprefix(prefix)
            if index_text.isdigit():
                highest = max(highest, int(index_text))
        return highest

    def get_job(
        self, job_id: str, connection: sqlite3.Connection | None = None
    ) -> JobRecord:
        if connection is None:
            with self._connection() as owned:
                row = owned.execute(
                    "SELECT * FROM jobs WHERE id = ? AND job_kind = 'image'",
                    (job_id,),
                ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ? AND job_kind = 'image'", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_from_row(row)

    def get_video_job(
        self, job_id: str, connection: sqlite3.Connection | None = None
    ) -> VideoJobRecord:
        if connection is None:
            with self._connection() as owned:
                row = owned.execute(
                    "SELECT * FROM jobs WHERE id = ? AND job_kind = 'video'",
                    (job_id,),
                ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ? AND job_kind = 'video'", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._video_job_from_row(row)

    def find_video_job_by_output(self, date_dir: str, name: str) -> VideoJobRecord:
        """按输出文件(日期目录名 + 文件名)反查视频任务,供相册视频元数据使用。

        LIKE 只做粗筛(文件名可能含通配符),精确匹配在 Python 侧完成。
        """
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE job_kind = 'video' AND output_path LIKE ?",
                (f"%{name}",),
            ).fetchall()
        for row in rows:
            job = self._video_job_from_row(row)
            if job.output_path.name == name and job.output_path.parent.name == date_dir:
                return job
        raise KeyError(f"{date_dir}/{name}")

    def _get_any_job(self, job_id: str):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row["job_kind"] == "video":
            return self._video_job_from_row(row)
        return self._job_from_row(row)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        mode: Mode | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[JobRecord], str | None]:
        """Keyset 分页读取任务，按 submitted_at DESC, id DESC 排序。

        cursor 编码为 ``submitted_at|id``，由上一页最后一条记录生成。
        返回 (jobs, next_cursor)；next_cursor 为 None 表示没有更多记录。
        """
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        clauses = ["job_kind = 'image'"]
        values: list = []
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if mode is not None:
            clauses.append("mode = ?")
            values.append(mode.value)
        if cursor is not None:
            try:
                cursor_submitted, cursor_id = cursor.rsplit("|", 1)
            except ValueError:
                raise ValueError(f"无效的分页 cursor: {cursor!r}") from None
            clauses.append("(submitted_at, id) < (?, ?)")
            values.extend([cursor_submitted, cursor_id])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs {where} "
                "ORDER BY submitted_at DESC, id DESC LIMIT ?",
                (*values, limit + 1),
            ).fetchall()
        jobs = [self._job_from_row(row) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit and jobs:
            last = jobs[-1]
            next_cursor = f"{last.submitted_at.isoformat()}|{last.id}"
        return jobs, next_cursor

    def mark_running(self, job_id: str, seed: int, started_at: datetime) -> JobRecord:
        self._update_job(
            job_id,
            "UPDATE jobs SET status = 'running', seed = ?, started_at = ? WHERE id = ?",
            (seed, started_at.isoformat(), job_id),
        )
        return self._get_any_job(job_id)

    def mark_completed(
        self, job_id: str, completed_at: datetime, duration_seconds: float
    ) -> None:
        self._update_job(
            job_id,
            "UPDATE jobs SET status = 'completed', completed_at = ?, duration_seconds = ?, error = NULL WHERE id = ?",
            (completed_at.isoformat(), duration_seconds, job_id),
        )

    def mark_failed(
        self, job_id: str, completed_at: datetime, duration_seconds: float, error: str
    ) -> None:
        self._update_job(
            job_id,
            "UPDATE jobs SET status = 'failed', completed_at = ?, duration_seconds = ?, error = ? WHERE id = ?",
            (completed_at.isoformat(), duration_seconds, error, job_id),
        )

    def mark_cancelled(
        self,
        job_id: str,
        completed_at: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        completed_at = completed_at or datetime.now().astimezone()
        self._update_job(
            job_id,
            "UPDATE jobs SET status = 'cancelled', completed_at = ?, duration_seconds = ? WHERE id = ?",
            (completed_at.isoformat(), duration_seconds, job_id),
        )

    def _update_job(self, job_id: str, sql: str, values: tuple) -> None:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(sql, values)
            if cursor.rowcount != 1:
                raise KeyError(job_id)

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        parse_time = lambda value: datetime.fromisoformat(value) if value else None
        return JobRecord(
            id=row["id"],
            batch_id=row["batch_id"],
            status=row["status"],
            submitted_at=parse_time(row["submitted_at"]),
            started_at=parse_time(row["started_at"]),
            completed_at=parse_time(row["completed_at"]),
            duration_seconds=row["duration_seconds"],
            output_path=Path(row["output_path"]),
            upscaled_output_path=(
                Path(row["upscaled_output_path"])
                if row["upscaled_output_path"]
                else None
            ),
            mode=Mode(row["mode"]),
            prompt=row["prompt"],
            negative_prompt=row["negative_prompt"],
            model_path=Path(row["model"]),
            vae_path=Path(row["vae"]) if row["vae"] else None,
            text_encoder_path=(Path(row["text_encoder"]) if row["text_encoder"] else None),
            text_encoder_source=row["text_encoder_source"] if "text_encoder_source" in row.keys() else "local",
            remote_text_encoder_id=row["remote_text_encoder_id"] if "remote_text_encoder_id" in row.keys() else None,
            sampler=row["sampler"],
            scheduler=row["scheduler"],
            width=row["width"],
            height=row["height"],
            steps=row["steps"],
            seed=row["seed"],
            cfg=row["cfg"],
            error=row["error"],
            upscale=UpscaleSettings.from_dict(json.loads(row["upscale_json"])),
            model_loader=ModelLoader(row["model_loader"]),
            input_image_path=(
                Path(row["input_image"]) if row["input_image"] else None
            ),
            secondary_input_image_path=(
                Path(row["secondary_input_image"])
                if row["secondary_input_image"]
                else None
            ),
            grounding_px=(
                int(row["grounding_px"]) if row["grounding_px"] is not None else None
            ),
            ref_boost=(
                float(row["ref_boost"]) if row["ref_boost"] is not None else None
            ),
            **_parse_image_reference_inputs(row),
        )

    @staticmethod
    def _video_job_from_row(row: sqlite3.Row) -> VideoJobRecord:
        parse_time = lambda value: datetime.fromisoformat(value) if value else None
        return VideoJobRecord(
            id=row["id"],
            batch_id=row["batch_id"],
            status=row["status"],
            submitted_at=parse_time(row["submitted_at"]),
            started_at=parse_time(row["started_at"]),
            completed_at=parse_time(row["completed_at"]),
            elapsed_seconds=row["duration_seconds"],
            output_path=Path(row["output_path"]),
            video_model=VideoModel(row["mode"]),
            prompt=row["prompt"],
            negative_prompt=row["negative_prompt"],
            input_image_path=(
                Path(row["input_image"]) if row["input_image"] else None
            ),
            **_parse_video_reference_inputs(row),
            model_path=Path(row["model"]),
            vae_path=Path(row["vae"]),
            text_encoder_path=Path(row["text_encoder"]),
            audio_vae_path=Path(row["audio_vae"]) if row["audio_vae"] else None,
            sampler=row["sampler"],
            scheduler=row["scheduler"],
            width=row["width"],
            height=row["height"],
            duration_seconds=row["video_duration_seconds"],
            fps=row["fps"],
            length=row["frame_count"],
            steps=row["steps"],
            seed=row["seed"],
            cfg=row["cfg"],
            denoise=row["denoise"],
            shift=row["shift"],
            latent_multiplier=row["latent_multiplier"],
            error=row["error"],
        )



def _rebalance_reference_json(settings: GenerationSettings) -> str | None:
    """把 krea2-rebalance 的参考图与 token 档位序列化为 JSON；无参考图时写 NULL。"""
    if not settings.reference_images:
        return None
    payload = {
        "images": [str(path.resolve()) for path in settings.reference_images],
        "tokens": list(settings.reference_image_tokens),
    }
    return json.dumps(payload)


def _parse_image_reference_inputs(row: sqlite3.Row) -> dict:
    """从 reference_inputs JSON 还原图片任务 JobRecord 的参考图字段。

    非 krea2-rebalance 任务该列为 NULL 或视频格式 JSON，解析失败一律回退为空 tuple。
    """
    raw = row["reference_inputs"]
    data = None
    if raw:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            data = None
    if not isinstance(data, dict):
        data = {}
    return {
        "reference_image_paths": tuple(
            Path(value) for value in data.get("images") or ()
        ),
        "reference_image_tokens": tuple(
            str(value) for value in data.get("tokens") or ()
        ),
    }


def _reference_inputs_json(settings: VideoGenerationSettings) -> str | None:
    """把尾帧与三类参考输入序列化为 JSON；全部为空时写 NULL。"""
    payload = {
        "last_frame_image": (
            str(settings.last_frame_image.resolve())
            if settings.last_frame_image is not None
            else None
        ),
        "images": [str(path.resolve()) for path in settings.reference_images],
        "videos": [str(path.resolve()) for path in settings.reference_videos],
        "audios": [str(path.resolve()) for path in settings.reference_audios],
    }
    if not payload["last_frame_image"] and not any(
        payload[key] for key in ("images", "videos", "audios")
    ):
        return None
    return json.dumps(payload)


def _parse_video_reference_inputs(row: sqlite3.Row) -> dict:
    """从 reference_inputs JSON 还原 VideoJobRecord 的参考输入字段。

    旧行没有 JSON 内容时回落到单张 reference_image 列，保证历史 r2v 任务可读。
    """
    raw = row["reference_inputs"]
    data = None
    if raw:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            data = None
    if not isinstance(data, dict):
        data = {}
    images = tuple(data.get("images") or ())
    if not images and row["reference_image"]:
        images = (row["reference_image"],)
    last_frame = data.get("last_frame_image")
    return {
        "last_frame_image_path": Path(last_frame) if last_frame else None,
        "reference_image_paths": tuple(Path(value) for value in images),
        "reference_video_paths": tuple(
            Path(value) for value in data.get("videos") or ()
        ),
        "reference_audio_paths": tuple(
            Path(value) for value in data.get("audios") or ()
        ),
    }
