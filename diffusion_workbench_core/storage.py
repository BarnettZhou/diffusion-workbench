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
    video_duration_seconds INTEGER,
    fps INTEGER,
    frame_count INTEGER,
    denoise REAL NOT NULL DEFAULT 1.0,
    shift REAL NOT NULL DEFAULT 8.0,
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
                "video_duration_seconds": "INTEGER",
                "fps": "INTEGER",
                "frame_count": "INTEGER",
                "denoise": "REAL NOT NULL DEFAULT 1.0",
                "shift": "REAL NOT NULL DEFAULT 8.0",
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
                )
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, batch_id, status, submitted_at, output_path, upscaled_output_path, output_date,
                        daily_index, mode, prompt, negative_prompt, model, vae, text_encoder, sampler,
                        scheduler, width, height, steps, seed, cfg, model_loader, upscale_json,
                        job_kind
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    settings.duration_seconds,
                    settings.fps,
                    settings.length,
                    settings.denoise,
                    settings.shift,
                )
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, batch_id, status, submitted_at, output_path, output_date,
                        daily_index, mode, prompt, negative_prompt, model, vae,
                        text_encoder, sampler, scheduler, width, height, steps, seed,
                        cfg, model_loader, upscale_json, job_kind, input_image,
                        video_duration_seconds, fps, frame_count, denoise, shift
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            model_path=Path(row["model"]),
            vae_path=Path(row["vae"]),
            text_encoder_path=Path(row["text_encoder"]),
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
            error=row["error"],
        )
