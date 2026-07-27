import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .domain import GenerationSettings, JobRecord, Mode, ResourceKind


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
            file_index = self._highest_output_index(output_date, settings.mode)
            next_index = max(persisted_index, file_index) + 1
            for offset in range(count):
                daily_index = next_index + offset
                job_id = str(uuid.uuid4())
                output_path = (
                    self.output_dir
                    / output_date
                    / f"{settings.mode.value}-{daily_index:05d}.png"
                ).resolve()
                values = (
                    job_id,
                    batch_id,
                    "queued",
                    submitted_iso,
                    str(output_path),
                    output_date,
                    daily_index,
                    settings.mode.value,
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
                )
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, batch_id, status, submitted_at, output_path, output_date,
                        daily_index, mode, prompt, negative_prompt, model, vae, text_encoder, sampler,
                        scheduler, width, height, steps, seed, cfg
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                jobs.append(self.get_job(job_id, connection=connection))
        return jobs

    def _highest_output_index(self, output_date: str, mode: Mode) -> int:
        directory = self.output_dir / output_date
        if not directory.is_dir():
            return 0
        prefix = f"{mode.value}-"
        highest = 0
        for path in directory.glob(f"{prefix}*.png"):
            index_text = path.stem.removeprefix(prefix)
            if index_text.isdigit():
                highest = max(highest, int(index_text))
        return highest

    def get_job(
        self, job_id: str, connection: sqlite3.Connection | None = None
    ) -> JobRecord:
        if connection is None:
            with self._connection() as owned:
                row = owned.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        else:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_from_row(row)

    def mark_running(self, job_id: str, seed: int, started_at: datetime) -> JobRecord:
        self._update_job(
            job_id,
            "UPDATE jobs SET status = 'running', seed = ?, started_at = ? WHERE id = ?",
            (seed, started_at.isoformat(), job_id),
        )
        return self.get_job(job_id)

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
            mode=Mode(row["mode"]),
            prompt=row["prompt"],
            negative_prompt=row["negative_prompt"],
            model_path=Path(row["model"]),
            vae_path=Path(row["vae"]),
            text_encoder_path=Path(row["text_encoder"]),
            sampler=row["sampler"],
            scheduler=row["scheduler"],
            width=row["width"],
            height=row["height"],
            steps=row["steps"],
            seed=row["seed"],
            cfg=row["cfg"],
            error=row["error"],
        )
