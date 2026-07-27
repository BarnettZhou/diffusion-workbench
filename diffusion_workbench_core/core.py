from pathlib import Path

from .catalog import ResourceCatalog
from .config import WorkbenchConfig, load_config
from .controller import GenerationController
from .domain import GenerationSettings, Mode, ResourceKind
from .instance_lock import InstanceLock
from .persistent_runtime import PersistentComfyRuntime
from .storage import JobStore


class WorkbenchCore:
    def __init__(self, config: WorkbenchConfig):
        self.config = config
        lock_path = config.database.with_suffix(config.database.suffix + ".lock")
        self._instance_lock = InstanceLock(lock_path)
        self.runtime = None
        try:
            self.store = JobStore(config.database, config.output_dir)
            self.store.recover_incomplete_jobs()
            self.catalog = ResourceCatalog(config, self.store)
            self.runtime = PersistentComfyRuntime(config)
            self.controller = GenerationController(self.runtime, self.store)
        except Exception:
            if self.runtime is not None:
                self.runtime.close()
            self._instance_lock.close()
            raise

    @classmethod
    def from_config(cls, path: str | Path) -> "WorkbenchCore":
        return cls(load_config(path))

    def list_resources(self, mode: Mode, kind: ResourceKind):
        return self.catalog.list(mode, kind)

    def set_alias(self, mode: Mode, kind: ResourceKind, path: Path, alias: str) -> None:
        self.catalog.set_alias(mode, kind, path, alias)

    def submit(self, settings: GenerationSettings, count: int):
        resources = self.config.resources[settings.mode]
        if settings.text_encoder.resolve() != resources.text_encoder.resolve():
            raise ValueError(
                f"{settings.mode.value} text encoder 固定为 {resources.text_encoder}"
            )
        if settings.clip_type != resources.clip_type:
            raise ValueError(
                f"{settings.mode.value} clip type 固定为 {resources.clip_type}"
            )
        model_paths = {
            item.path.resolve()
            for item in self.catalog.list(settings.mode, ResourceKind.DIFFUSION)
        }
        if settings.model.path.resolve() not in model_paths:
            raise ValueError(
                f"model 不属于 {settings.mode.value} 配置的 diffusion 目录"
            )
        vae_paths = {
            item.path.resolve()
            for item in self.catalog.list(settings.mode, ResourceKind.VAE)
        }
        if settings.vae.path.resolve() not in vae_paths:
            raise ValueError(f"VAE 不属于 {settings.mode.value} 配置的 vae 目录")
        return self.controller.submit(settings, count)

    def stop(self) -> None:
        self.controller.stop()

    def runtime_status(self) -> dict:
        return self.controller.status()

    def set_event_sink(self, sink) -> None:
        self.controller.set_event_sink(sink)

    def set_preview_enabled(self, enabled: bool) -> None:
        """Enable or disable base64 JPEG preview events for future sampling steps."""

        self.runtime.set_preview_enabled(enabled)

    def shutdown(self) -> None:
        try:
            self.controller.shutdown()
        finally:
            self._instance_lock.close()
