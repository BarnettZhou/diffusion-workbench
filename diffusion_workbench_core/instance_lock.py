import os
from pathlib import Path


class InstanceLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+b")
        if self._file.seek(0, os.SEEK_END) == 0:
            self._file.write(b"\0")
            self._file.flush()
        self._file.seek(0)
        try:
            self._lock()
        except OSError as exc:
            self._file.close()
            raise RuntimeError(
                f"已有 diffusion-workbench 使用数据库 {self.path.with_suffix('')}"
            ) from exc
        self._closed = False

    def _lock(self) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._closed = True
