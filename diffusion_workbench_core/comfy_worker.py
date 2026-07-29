"""Long-lived headless ComfyUI worker. Run only with ComfyUI's Python."""

import argparse
import base64
import gc
import json
import logging
import os
import platform
import queue
import sys
import threading
import time
import traceback
from io import BytesIO
from pathlib import Path

try:
    from .domain import (
        SAMPLERS,
        SCHEDULERS,
        UpscaleMethod,
        UpscaleSettings,
        validate_sampling,
    )
    from .png_metadata import (
        ResourceFingerprintCache,
        build_generation_metadata,
        create_png_info,
    )
except ImportError:  # The Comfy worker runs this module as a standalone script.
    from domain import (
        SAMPLERS,
        SCHEDULERS,
        UpscaleMethod,
        UpscaleSettings,
        validate_sampling,
    )
    from png_metadata import (
        ResourceFingerprintCache,
        build_generation_metadata,
        create_png_info,
    )


EVENT_PREFIX = "DWB_EVENT="


def emit(payload: dict) -> None:
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    return parser


def sampling_progress_payload(
    step: int,
    total_steps: int,
    started_at: float,
    previous_step_at: float,
    now: float,
) -> dict:
    completed = int(step) + 1
    total = int(total_steps)
    elapsed = max(now - started_at, 0.0)
    average = elapsed / completed if completed else 0.0
    step_seconds = max(now - previous_step_at, 0.0)
    return {
        "step": completed,
        "total": total,
        "elapsed_seconds": round(elapsed, 4),
        "step_seconds": round(step_seconds, 4),
        "seconds_per_step": round(average, 4),
        "steps_per_second": round(1.0 / average, 4) if average > 0 else None,
        "eta_seconds": round(average * max(total - completed, 0), 4),
    }


def encode_preview_image(previewer, x0) -> dict:
    image = previewer.decode_latent_to_preview(x0)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=False)
    return {
        "mime_type": "image/jpeg",
        "encoding": "base64",
        "width": image.width,
        "height": image.height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def resolve_upscale_settings(command: dict) -> dict:
    """把 inherit 参数解析为当前任务实际执行值。"""

    settings = UpscaleSettings.from_dict(command.get("upscale"))
    settings.validate()
    resolved = settings.to_dict()
    if not settings.enabled or settings.method != UpscaleMethod.LATENT_HIRES:
        return resolved
    for key in ("cfg", "sampler", "scheduler", "seed"):
        inherited = resolved[key] is None
        resolved[f"{key}_inherited"] = inherited
        if inherited:
            resolved[key] = command[key]
    return resolved


class ComfyWorker:
    def __init__(self, comfy_root: Path):
        comfy_root = comfy_root.resolve()
        sys.path.insert(0, str(comfy_root))
        os.chdir(comfy_root)

        import comfy.options

        comfy.options.args_parsing = False
        import folder_paths
        from utils.extra_config import load_extra_path_config

        extra_paths = comfy_root / "extra_model_paths.yaml"
        if extra_paths.is_file():
            load_extra_path_config(str(extra_paths))

        logging.getLogger("xformers").setLevel(logging.ERROR)
        import comfy.model_management
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import comfyui_version
        import latent_preview
        import nodes
        import torch
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader
        from PIL import Image

        self.available_samplers = frozenset(comfy.samplers.KSampler.SAMPLERS)
        self.available_schedulers = frozenset(comfy.samplers.KSampler.SCHEDULERS)
        if "euler" not in self.available_samplers:
            raise RuntimeError("当前 ComfyUI 不支持默认 sampler: euler")
        if "simple" not in self.available_schedulers:
            raise RuntimeError("当前 ComfyUI 不支持默认 scheduler: simple")

        self.folder_paths = folder_paths
        self.model_management = comfy.model_management
        self.comfy_sample = comfy.sample
        self.comfy_utils = comfy.utils
        self.latent_preview = latent_preview
        self.nodes = nodes
        self.torch = torch
        self.Image = Image
        self.UpscaleModelLoader = UpscaleModelLoader
        self.runtime_versions = {
            "comfyui": str(comfyui_version.__version__),
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        }
        self.resource_fingerprints = ResourceFingerprintCache()
        self.model = None
        self.model_path: Path | None = None
        self.clip = None
        self.clip_path: Path | None = None
        self.clip_type: str | None = None
        self.vae = None
        self.vae_path: Path | None = None
        self.mode: str | None = None
        self.upscale_model = None
        self.upscale_model_path: Path | None = None

    def generate(self, command: dict) -> dict:
        self._validate(command)
        self._validate_runtime_sampling(command)
        with self.torch.inference_mode():
            return self._generate(command)

    def _generate(self, command: dict) -> dict:
        command = dict(command)
        command["upscale"] = resolve_upscale_settings(command)
        job_id = command["job_id"]
        requested_mode = command["mode"]
        if self.mode is not None and self.mode != requested_mode:
            self.release()
        self.mode = requested_mode
        model_path = Path(command["model_path"]).resolve()
        vae_path = Path(command["vae_path"]).resolve()
        text_encoder_path = Path(command["text_encoder_path"]).resolve()
        clip_type = command["clip_type"]
        load_started = time.perf_counter()
        stage_total = int(command["steps"])
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "loading_model",
                "total": stage_total,
            }
        )
        loaded_model = self._ensure_model(model_path)
        loaded_clip = self._ensure_clip(text_encoder_path, clip_type)
        loaded_vae = self._ensure_vae(vae_path)
        resource_metadata = {
            "diffusion_model": self.resource_fingerprints.describe(model_path),
            "vae": self.resource_fingerprints.describe(vae_path),
            "text_encoder": self.resource_fingerprints.describe(text_encoder_path),
        }
        previewer = self._create_previewer() if command.get("preview_enabled") else None

        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "prompt",
                "total": stage_total,
            }
        )
        positive = self.nodes.CLIPTextEncode().encode(self.clip, command["prompt"])[0]
        if command["mode"] != "zib" and float(command["cfg"]) == 1.0:
            negative = positive
        else:
            negative = self.nodes.CLIPTextEncode().encode(
                self.clip, command.get("negative_prompt", "")
            )[0]
        load_seconds = time.perf_counter() - load_started

        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "latent",
                "total": stage_total,
            }
        )
        width, height = int(command["width"]), int(command["height"])
        latent = {
            "samples": self.torch.zeros(
                [1, 16, height // 8, width // 8],
                device=self.model_management.intermediate_device(),
                dtype=self.model_management.intermediate_dtype(),
            ),
            "downscale_ratio_spacial": 8,
        }
        self.torch.cuda.reset_peak_memory_stats()

        def make_sampling_callback(stage: str, started_at: float):
            previous_step_at = started_at

            def sampling_callback(step, x0, _x, total_steps):
                nonlocal previous_step_at, previewer
                now = time.perf_counter()
                progress = sampling_progress_payload(
                    step, total_steps, started_at, previous_step_at, now
                )
                previous_step_at = now
                emit(
                    {
                        "type": "step_progress",
                        "job_id": job_id,
                        "stage": stage,
                        **progress,
                    }
                )
                if previewer is not None:
                    try:
                        emit(
                            {
                                "type": "preview_image",
                                "job_id": job_id,
                                "stage": stage,
                                "step": progress["step"],
                                "total": progress["total"],
                                **encode_preview_image(previewer, x0),
                            }
                        )
                    except Exception:
                        logging.exception("生成 latent 预览失败，当前任务将停止输出预览")
                        previewer = None

            return sampling_callback

        sampling_started = time.perf_counter()
        sampling_callback = make_sampling_callback("sampling", sampling_started)

        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "sampling",
                "total": int(command["steps"]),
            }
        )
        samples = self._sample(command, latent, positive, negative, sampling_callback)
        sampling_seconds = time.perf_counter() - sampling_started
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "vae",
                "total": stage_total,
            }
        )
        vae_started = time.perf_counter()
        images = self.nodes.VAEDecode().decode(self.vae, samples)[0]
        vae_seconds = time.perf_counter() - vae_started
        original_generation_seconds = sampling_seconds + vae_seconds

        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "saving",
                "total": stage_total,
            }
        )
        output_path = Path(command["output_path"]).resolve()
        peak_allocated = self.torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = self.torch.cuda.max_memory_reserved() / 1024**3
        performance = {
            "load_seconds": round(load_seconds, 3),
            "sampling_seconds": round(sampling_seconds, 3),
            "vae_seconds": round(vae_seconds, 3),
            "generation_seconds": round(original_generation_seconds, 3),
            "cuda_peak_allocated_gib": round(peak_allocated, 3),
            "cuda_peak_reserved_gib": round(peak_reserved, 3),
        }
        original_metadata = build_generation_metadata(
            command,
            self.runtime_versions,
            performance,
            resource_metadata,
            artifact_kind="original",
            artifact_size=(width, height),
        )
        self._save_image(images, output_path, original_metadata, job_id)
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "saved",
                "total": stage_total,
            }
        )
        upscale_seconds = 0.0
        upscaled_output_path = None
        upscale = command["upscale"]
        if upscale["enabled"]:
            redraw_steps = (
                int(upscale["steps"] - upscale["start_step"])
                if upscale["method"] == UpscaleMethod.LATENT_HIRES
                else None
            )
            emit(
                {
                    "type": "stage_progress",
                    "job_id": job_id,
                    "stage": "upscale_preparing",
                    "total": redraw_steps,
                }
            )
            upscale_started = time.perf_counter()
            if upscale["method"] == UpscaleMethod.RESIZE:
                upscaled_images = self.nodes.ImageScaleBy().upscale(
                    images, upscale["interpolation"], float(upscale["scale"])
                )[0]
            elif upscale["method"] == UpscaleMethod.UPSCALE_MODEL:
                upscale_model_path = Path(upscale["model_path"]).resolve()
                self._ensure_upscale_model(upscale_model_path)
                resource_metadata["upscale_model"] = (
                    self.resource_fingerprints.describe(upscale_model_path)
                )
                upscaled_images = self._upscale_with_model(
                    images,
                    tile=int(upscale["tile"]),
                    overlap=int(upscale["overlap"]),
                )
                native_scale = float(self.upscale_model.scale)
                post_scale = float(upscale["scale"]) / native_scale
                if abs(post_scale - 1.0) > 1e-6:
                    upscaled_images = self.nodes.ImageScaleBy().upscale(
                        upscaled_images, upscale["interpolation"], post_scale
                    )[0]
            else:
                upscaled_latent = self.nodes.LatentUpscaleBy().upscale(
                    samples,
                    upscale["interpolation"],
                    float(upscale["scale"]),
                )[0]
                emit(
                    {
                        "type": "stage_progress",
                        "job_id": job_id,
                        "stage": "upscale_sampling",
                        "total": redraw_steps,
                    }
                )
                redraw_started = time.perf_counter()
                upscaled_samples = self._sample(
                    command,
                    upscaled_latent,
                    positive,
                    negative,
                    make_sampling_callback("upscale_sampling", redraw_started),
                    sampling=upscale,
                    start_step=int(upscale["start_step"]),
                )
                emit(
                    {
                        "type": "stage_progress",
                        "job_id": job_id,
                        "stage": "upscale_decoding",
                        "total": redraw_steps,
                    }
                )
                upscaled_images = self.nodes.VAEDecode().decode(
                    self.vae, upscaled_samples
                )[0]
            upscale_seconds = time.perf_counter() - upscale_started
            emit(
                {
                    "type": "stage_progress",
                    "job_id": job_id,
                    "stage": "upscale_saving",
                    "total": redraw_steps,
                }
            )
            upscaled_output_path = Path(command["upscaled_output_path"]).resolve()
            target_width = int(upscaled_images.shape[2])
            target_height = int(upscaled_images.shape[1])
            upscaled_performance = {
                **performance,
                "upscale_seconds": round(upscale_seconds, 3),
                "generation_seconds": round(
                    original_generation_seconds + upscale_seconds, 3
                ),
                "cuda_peak_allocated_gib": round(
                    self.torch.cuda.max_memory_allocated() / 1024**3, 3
                ),
                "cuda_peak_reserved_gib": round(
                    self.torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
            }
            upscaled_metadata = build_generation_metadata(
                command,
                self.runtime_versions,
                upscaled_performance,
                resource_metadata,
                artifact_kind="upscaled",
                artifact_size=(target_width, target_height),
            )
            self._save_image(
                upscaled_images,
                upscaled_output_path,
                upscaled_metadata,
                job_id,
            )
            emit(
                {
                    "type": "stage_progress",
                    "job_id": job_id,
                    "stage": "upscale_saved",
                    "total": redraw_steps,
                }
            )
        return {
            "type": "result",
            "job_id": job_id,
            "model_path": str(model_path),
            "vae_path": str(vae_path),
            "output_path": str(output_path),
            "upscaled_output_path": (
                str(upscaled_output_path) if upscaled_output_path else None
            ),
            "loaded_model": loaded_model,
            "loaded_clip": loaded_clip,
            "loaded_vae": loaded_vae,
            **performance,
            "upscale_seconds": round(upscale_seconds, 3),
            "generation_seconds": round(
                original_generation_seconds + upscale_seconds, 3
            ),
            "cuda_peak_allocated_gib": round(
                self.torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "cuda_peak_reserved_gib": round(
                self.torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        }

    def _sample(
        self,
        command,
        latent,
        positive,
        negative,
        callback,
        *,
        sampling=None,
        start_step=None,
    ):
        sampling = sampling or command
        latent_image = self.comfy_sample.fix_empty_latent_channels(
            self.model,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        seed = int(sampling["seed"])
        noise = self.comfy_sample.prepare_noise(latent_image, seed, None)
        sampled = self.comfy_sample.sample(
            self.model,
            noise,
            int(sampling["steps"]),
            float(sampling["cfg"]),
            sampling["sampler"],
            sampling["scheduler"],
            positive,
            negative,
            latent_image,
            denoise=1.0,
            start_step=start_step,
            force_full_denoise=start_step is not None,
            callback=callback,
            disable_pbar=True,
            seed=seed,
        )
        output = latent.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = sampled
        return output

    def _save_image(self, images, output_path: Path, metadata: dict, job_id: str) -> None:
        pixels = images[0].detach().cpu().clamp(0, 1).mul(255).byte().numpy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{job_id}.tmp")
        try:
            self.Image.fromarray(pixels).save(
                temporary_path,
                format="PNG",
                pnginfo=create_png_info(metadata),
            )
            os.link(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _ensure_upscale_model(self, path: Path) -> bool:
        if self.upscale_model is not None and self.upscale_model_path == path:
            return False
        self.upscale_model = None
        self.upscale_model_path = None
        gc.collect()
        name = self._register_exact("upscale_models", path)
        self.upscale_model = self.UpscaleModelLoader().load_model(name)[0]
        self.upscale_model_path = path
        return True

    def _upscale_with_model(self, images, tile: int, overlap: int):
        device = self.model_management.get_torch_device()
        model = self.upscale_model
        memory_required = self.model_management.module_size(model.model)
        memory_required += (
            (tile * tile * 3)
            * images.element_size()
            * max(float(model.scale), 1.0)
            * 384.0
        )
        memory_required += images.nelement() * images.element_size()
        self.model_management.free_memory(memory_required, device)
        model.to(device)
        input_images = images.movedim(-1, -3).to(device)
        output_device = self.model_management.intermediate_device()
        try:
            while True:
                try:
                    steps = input_images.shape[0] * self.comfy_utils.get_tiled_scale_steps(
                        input_images.shape[3],
                        input_images.shape[2],
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                    )
                    progress = self.comfy_utils.ProgressBar(steps)
                    result = self.comfy_utils.tiled_scale(
                        input_images,
                        lambda value: model(value.float()),
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                        upscale_amount=model.scale,
                        pbar=progress,
                        output_device=output_device,
                    )
                    break
                except Exception as exc:
                    self.model_management.raise_non_oom(exc)
                    tile //= 2
                    if tile < 128:
                        raise
        finally:
            model.to("cpu")
        return self.torch.clamp(result.movedim(-3, -1), min=0, max=1.0).to(
            self.model_management.intermediate_dtype()
        )

    def _ensure_model(self, path: Path) -> bool:
        if self.model is not None and self.model_path == path:
            return False
        self._unload_gpu()
        self.model = None
        self.model_path = None
        gc.collect()
        name = self._register_exact("diffusion_models", path)
        self.model = self.nodes.UNETLoader().load_unet(name, "default")[0]
        self.model_path = path
        return True

    def _create_previewer(self):
        latent_format = self.model.model.latent_format
        factors = latent_format.latent_rgb_factors
        if factors is None:
            return None
        return self.latent_preview.Latent2RGBPreviewer(
            factors,
            latent_format.latent_rgb_factors_bias,
            latent_format.latent_rgb_factors_reshape,
        )

    def _ensure_clip(self, path: Path, clip_type: str) -> bool:
        if self.clip is not None and self.clip_path == path and self.clip_type == clip_type:
            return False
        self._unload_gpu()
        self.clip = None
        self.clip_path = None
        self.clip_type = None
        gc.collect()
        name = self._register_exact("text_encoders", path)
        self.clip = self.nodes.CLIPLoader().load_clip(name, clip_type)[0]
        self.clip_path = path
        self.clip_type = clip_type
        return True

    def _ensure_vae(self, path: Path) -> bool:
        if self.vae is not None and self.vae_path == path:
            return False
        self._unload_gpu()
        self.vae = None
        self.upscale_model = None
        self.vae_path = None
        gc.collect()
        name = self._register_exact("vae", path)
        self.vae = self.nodes.VAELoader().load_vae(name)[0]
        self.vae_path = path
        return True

    def _register_exact(self, category: str, path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {category}: {path}")
        self.folder_paths.add_model_folder_path(
            category, str(path.parent), is_default=True
        )
        name = path.name
        resolved = Path(self.folder_paths.get_full_path_or_raise(category, name))
        if not resolved.samefile(path):
            raise RuntimeError(
                f"ComfyUI {category} 路径解析不一致: resolved={resolved} expected={path}"
            )
        return name

    def _unload_gpu(self) -> None:
        self.model_management.unload_all_models()
        self.model_management.soft_empty_cache(force=True)

    def cancel(self) -> None:
        self.model_management.interrupt_current_processing()

    def release(self) -> None:
        self._unload_gpu()
        self.model = None
        self.clip = None
        self.vae = None
        self.model_path = self.clip_path = self.vae_path = None
        self.upscale_model_path = None
        self.clip_type = None
        self.mode = None
        gc.collect()
        self.model_management.cleanup_models_gc()
        self.model_management.cleanup_models()
        self.model_management.soft_empty_cache(force=True)

    @staticmethod
    def _validate(command: dict) -> None:
        if command.get("mode") not in {"zit", "krea2", "zib"}:
            raise ValueError("mode 必须是 zit、krea2 或 zib")
        cfg = float(command.get("cfg", 0))
        steps = int(command["steps"])
        validate_sampling(
            steps, cfg, command.get("sampler"), command.get("scheduler")
        )
        width, height = int(command["width"]), int(command["height"])
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("图片宽高必须为正数且是 16 的倍数")
        upscale = UpscaleSettings.from_dict(command.get("upscale"))
        upscale.validate()
        if upscale.enabled and not command.get("upscaled_output_path"):
            raise ValueError("启用放大时必须提供 upscaled_output_path")

    def _validate_runtime_sampling(self, command: dict) -> None:
        sampler = command["sampler"]
        scheduler = command["scheduler"]
        if sampler not in self.available_samplers:
            raise RuntimeError(f"当前 ComfyUI 不支持 sampler: {sampler}")
        if scheduler not in self.available_schedulers:
            raise RuntimeError(f"当前 ComfyUI 不支持 scheduler: {scheduler}")
        upscale = resolve_upscale_settings(command)
        if upscale["enabled"] and upscale["method"] == UpscaleMethod.LATENT_HIRES:
            if upscale["sampler"] not in self.available_samplers:
                raise RuntimeError(
                    f"当前 ComfyUI 不支持放大 sampler: {upscale['sampler']}"
                )
            if upscale["scheduler"] not in self.available_schedulers:
                raise RuntimeError(
                    f"当前 ComfyUI 不支持放大 scheduler: {upscale['scheduler']}"
                )


def main() -> None:
    args = build_parser().parse_args()
    try:
        worker = ComfyWorker(args.comfy_root)
    except Exception as exc:
        emit({"type": "startup_error", "error": f"{type(exc).__name__}: {exc}"})
        raise
    emit({"type": "ready"})

    commands: queue.Queue[dict] = queue.Queue()
    active_lock = threading.RLock()
    active_job_id: str | None = None

    def read_commands() -> None:
        nonlocal active_job_id
        try:
            for line in sys.stdin:
                try:
                    command = json.loads(line)
                except Exception as exc:
                    commands.put(
                        {
                            "type": "invalid",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                command_type = command.get("type")
                if command_type == "cancel":
                    with active_lock:
                        should_cancel = active_job_id == command.get("job_id")
                    if should_cancel:
                        worker.cancel()
                    continue
                if command_type == "shutdown":
                    with active_lock:
                        should_cancel = active_job_id is not None
                    if should_cancel:
                        worker.cancel()
                commands.put(command)
        finally:
            commands.put({"type": "shutdown"})

    threading.Thread(
        target=read_commands,
        name="worker-command-reader",
        daemon=True,
    ).start()

    while True:
        command = commands.get()
        command_type = command.get("type")
        if command_type == "shutdown":
            worker.release()
            emit({"type": "stopped"})
            return
        if command_type != "generate":
            emit(
                {
                    "type": "error",
                    "job_id": command.get("job_id"),
                    "error": command.get("error", "未知 Worker 命令"),
                }
            )
            continue
        with active_lock:
            active_job_id = command.get("job_id")
        try:
            payload = worker.generate(command)
        except worker.model_management.InterruptProcessingException:
            payload = {
                "type": "cancelled",
                "job_id": command.get("job_id"),
                "model_path": str(worker.model_path) if worker.model_path else None,
            }
        except Exception as exc:
            traceback_text = traceback.format_exc()
            try:
                worker.release()
            except Exception as cleanup_exc:
                logging.exception("生成失败后的资源清理也失败: %s", cleanup_exc)
            payload = {
                "type": "error",
                "job_id": command.get("job_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback_text,
            }
        finally:
            with active_lock:
                active_job_id = None
        emit(payload)


if __name__ == "__main__":
    main()
