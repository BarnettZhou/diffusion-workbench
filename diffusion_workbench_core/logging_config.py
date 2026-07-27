import copy
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5


class CoreLogManager:
    """Own one Core instance's rotating diagnostic log and file handle."""

    def __init__(self, database: Path):
        self.path = Path(database).with_suffix(".log").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.Logger(
            "diffusion_workbench_core.instance", level=logging.INFO
        )
        self.logger.propagate = False
        self._handler = RotatingFileHandler(
            self.path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        self._handler.setFormatter(CoreLogFormatter())
        self.logger.addHandler(self._handler)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.logger.removeHandler(self._handler)
        self._handler.close()


class CoreLogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            "[%(levelname)s] %(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        rendered = copy.copy(record)
        if rendered.levelname == "WARNING":
            rendered.levelname = "WARN"
        return super().format(rendered)


_NULL_LOGGER = logging.getLogger("diffusion_workbench_core.silent")
_NULL_LOGGER.addHandler(logging.NullHandler())
_NULL_LOGGER.propagate = False


def silent_logger() -> logging.Logger:
    return _NULL_LOGGER


def worker_output_level(line: str) -> int:
    normalized = line.lstrip().upper()
    if normalized.startswith(
        ("ERROR", "[ERROR]", "CRITICAL", "[CRITICAL]", "FATAL", "[FATAL]", "TRACEBACK")
    ):
        return logging.ERROR
    if normalized.startswith(("WARNING", "[WARNING]", "WARN", "[WARN]")):
        return logging.WARNING
    return logging.INFO
