from pathlib import Path

from .catalog import ResourceCatalog
from .config import WorkbenchConfig, load_config
from .controller import GenerationController
from .domain import GenerationSettings, Mode, ModelLoader, ResourceKind, UpscaleMethod
from .instance_lock import InstanceLock
from .logging_config import CoreLogManager
from .persistent_runtime import PersistentComfyRuntime
from .storage import JobStore


class WorkbenchCore:
    def __init__(self, config: WorkbenchConfig):
        self.config = config
        self.log_path = config.database.with_suffix(".log").resolve()
        lock_path = config.database.with_suffix(config.database.suffix + ".lock")
        self._instance_lock = None
        self._log_manager = None
        self.runtime = None
        try:
            self._instance_lock = InstanceLock(lock_path)
            self._log_manager = CoreLogManager(config.database)
            logger = self._log_manager.logger
            logger.info(
                "core starting config=%s database=%s output_dir=%s",
                config.path,
                config.database,
                config.output_dir,
            )
            self.store = JobStore(config.database, config.output_dir)
            self.store.recover_incomplete_jobs()
            self.catalog = ResourceCatalog(config, self.store)
            self.runtime = PersistentComfyRuntime(config, logger=logger)
            self.controller = GenerationController(
                self.runtime, self.store, logger=logger
            )
            logger.info("core ready log_file=%s", self.log_path)
        except Exception:
            if self._log_manager is not None:
                self._log_manager.logger.exception("core startup failed")
            if self.runtime is not None:
                self.runtime.close()
            if self._instance_lock is not None:
                self._instance_lock.close()
            if self._log_manager is not None:
                self._log_manager.close()
            raise

    @classmethod
    def from_config(cls, path: str | Path) -> "WorkbenchCore":
        return cls(load_config(path))

    def list_resources(self, mode: Mode, kind: ResourceKind):
        return self.catalog.list(mode, kind)

    def set_alias(self, mode: Mode, kind: ResourceKind, path: Path, alias: str) -> None:
        self.catalog.set_alias(mode, kind, path, alias)

    def list_upscale_models(self):
        return self.catalog.list_upscale_models()

    def submit(self, settings: GenerationSettings, count: int):
        resources = self.config.resources.get(settings.mode)
        if resources is None:
            raise ValueError(f"未配置 mode: {settings.mode.value}")
        if settings.model_loader != resources.model_loader:
            raise ValueError(f"{settings.mode.value} model loader 配置不匹配")
        if resources.model_loader == ModelLoader.COMPONENTS:
            if (
                settings.text_encoder is None
                or resources.text_encoder is None
                or settings.text_encoder.resolve() != resources.text_encoder.resolve()
            ):
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
                f"model 不属于 {settings.mode.value} 配置的 diffusion 目录或文件路径"
            )
        if resources.model_loader == ModelLoader.COMPONENTS:
            vae_paths = {
                item.path.resolve()
                for item in self.catalog.list(settings.mode, ResourceKind.VAE)
            }
            if settings.vae is None or settings.vae.path.resolve() not in vae_paths:
                raise ValueError(f"VAE 不属于 {settings.mode.value} 配置的 vae 目录")
        elif settings.vae is not None or settings.text_encoder is not None:
            raise ValueError(f"{settings.mode.value} checkpoint 已内嵌 VAE 和文本编码器")
        if settings.upscale.enabled and settings.upscale.method == UpscaleMethod.UPSCALE_MODEL:
            upscale_model_paths = {
                item.path.resolve() for item in self.catalog.list_upscale_models()
            }
            if (
                settings.upscale.model is None
                or settings.upscale.model.path.resolve() not in upscale_model_paths
            ):
                raise ValueError("放大模型不属于配置的 upscaling.models 目录")
        return self.controller.submit(settings, count)

    def stop(self) -> None:
        self.controller.stop()

    def skip_current(self) -> str | None:
        return self.controller.skip_current()

    def get_job(self, job_id: str):
        return self.store.get_job(job_id)

    def list_jobs(self, *, status=None, mode=None, limit=50, cursor=None):
        return self.store.list_jobs(status=status, mode=mode, limit=limit, cursor=cursor)

    def runtime_status(self) -> dict:
        return self.controller.status()

    def set_event_sink(self, sink) -> None:
        self.controller.set_event_sink(sink)

    def set_preview_enabled(self, enabled: bool) -> None:
        """Enable or disable base64 JPEG preview events for future sampling steps."""

        if self._log_manager is not None:
            self._log_manager.logger.info("preview enabled=%s", bool(enabled))
        self.runtime.set_preview_enabled(enabled)

    def shutdown(self) -> None:
        logger = self._log_manager.logger if self._log_manager is not None else None
        if logger is not None:
            logger.info("core shutting down")
        try:
            self.controller.shutdown()
        finally:
            if self._instance_lock is not None:
                self._instance_lock.close()
            if logger is not None:
                logger.info("core stopped")
            if self._log_manager is not None:
                self._log_manager.close()
