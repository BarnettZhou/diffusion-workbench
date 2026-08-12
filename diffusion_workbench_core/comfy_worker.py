"""Long-lived headless ComfyUI worker. Run only with ComfyUI's Python."""

import argparse
import base64
import gc
import importlib.util
import json
import logging
import math
import os
import platform
import queue
import sys
import threading
import time
import traceback
from fractions import Fraction
from io import BytesIO
from pathlib import Path

try:
    from .domain import (
        SAMPLERS,
        SCHEDULERS,
        Mode,
        ModelLoader,
        UpscaleMethod,
        UpscaleSettings,
        VideoModel,
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
        Mode,
        ModelLoader,
        UpscaleMethod,
        UpscaleSettings,
        VideoModel,
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
        self.comfy_root = comfy_root
        sys.path.insert(0, str(comfy_root))
        os.chdir(comfy_root)

        import comfy.options

        comfy.options.args_parsing = False
        import comfy.cli_args

        # Headless Worker 不解析 main.py 参数；统一使用 PyTorch attention，避免继承整合包里
        # 与当前 PyTorch/CUDA ABI 不匹配的 xformers 二进制。
        comfy.cli_args.args.disable_xformers = True
        comfy.cli_args.args.use_pytorch_cross_attention = True
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
        import numpy
        from comfy_api.latest import InputImpl, Types
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
            MiniMaxH3SigmaShift,
        )
        from comfy_extras.nodes_wan import WanImageToVideo, Wan22ImageToVideoLatent
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader
        from PIL import Image, ImageOps

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
        self.ImageOps = ImageOps
        self.numpy = numpy
        self.InputImpl = InputImpl
        self.Types = Types
        self.ModelSamplingSD3 = ModelSamplingSD3
        self.VAEDecodeAudio = VAEDecodeAudio
        self.MiniMaxH3ImageToVideo = MiniMaxH3ImageToVideo
        self.MiniMaxH3ReferenceToVideo = MiniMaxH3ReferenceToVideo
        self.MiniMaxH3SigmaShift = MiniMaxH3SigmaShift
        self.WanImageToVideo = WanImageToVideo
        self.Wan22ImageToVideoLatent = Wan22ImageToVideoLatent
        self.UpscaleModelLoader = UpscaleModelLoader
        self.runtime_versions = {
            "comfyui": str(comfyui_version.__version__),
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        }
        self.resource_fingerprints = ResourceFingerprintCache()
        self.gguf_unet_loader_class = None
        self.model = None
        self.model_path: Path | None = None
        self.model_high = None
        self.model_low = None
        self.model_high_path: Path | None = None
        self.model_low_path: Path | None = None
        self.clip = None
        self.clip_path: Path | None = None
        self.clip_type: str | None = None
        self.vae = None
        self.vae_path: Path | None = None
        self.audio_vae = None
        self.audio_vae_path: Path | None = None
        self.mode: str | None = None
        self.checkpoint_path: Path | None = None
        self.upscale_model = None
        self.upscale_model_path: Path | None = None

    def generate(self, command: dict) -> dict:
        if command.get("type") == "generate_video":
            self._validate_video(command)
        else:
            self._validate(command)
        self._validate_runtime_sampling(command)
        with self.torch.inference_mode():
            if command.get("type") == "generate_video":
                return self._generate_video(command)
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
        model_loader = ModelLoader(command.get("model_loader", ModelLoader.COMPONENTS))
        vae_path = Path(command["vae_path"]).resolve() if command.get("vae_path") else None
        text_encoder_path = (
            Path(command["text_encoder_path"]).resolve()
            if command.get("text_encoder_path")
            else None
        )
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
        if model_loader == ModelLoader.CHECKPOINT:
            loaded_model = self._ensure_checkpoint(model_path)
            loaded_clip = loaded_vae = loaded_model
            resource_metadata = {
                "diffusion_model": self.resource_fingerprints.describe(model_path)
            }
        else:
            assert vae_path is not None and text_encoder_path is not None
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
        latent_format = self.model.model.latent_format
        latent_channels = int(latent_format.latent_channels)
        downscale_ratio = int(latent_format.spacial_downscale_ratio)
        latent = {
            "samples": self.torch.zeros(
                [1, latent_channels, height // downscale_ratio, width // downscale_ratio],
                device=self.model_management.intermediate_device(),
                dtype=self.model_management.intermediate_dtype(),
            ),
            "downscale_ratio_spacial": downscale_ratio,
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
            "vae_path": str(vae_path) if vae_path else None,
            "output_path": str(output_path),
            "upscaled_output_path": (
                str(upscaled_output_path) if upscaled_output_path else None
            ),
            "loaded_model": loaded_model,
            "loaded_clip": loaded_clip,
            "loaded_vae": loaded_vae,
            "loaded_resources": self.resource_status(),
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

    def _generate_video(self, command: dict) -> dict:
        command = dict(command)
        if VideoModel(command["video_model"]) == VideoModel.MINIMAX_H3:
            return self._generate_minimax_h3(command)
        job_id = command["job_id"]
        requested_mode = command["video_model"]
        if self.mode is not None and self.mode != requested_mode:
            self.release()
        self.mode = requested_mode
        model_path = Path(command["model_path"]).resolve()
        model_high_path = Path(command.get("model_high_path") or model_path).resolve()
        model_low_path = (
            Path(command["model_low_path"]).resolve()
            if command.get("model_low_path")
            else None
        )
        vae_path = Path(command["vae_path"]).resolve()
        text_encoder_path = Path(command["text_encoder_path"]).resolve()
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
        if model_low_path is not None:
            loaded_model = self._ensure_video_models(model_high_path, model_low_path)
            high_model = self.model_high
            low_model = self.model_low
        else:
            loaded_model = self._ensure_model(model_path)
            high_model = self.model
            low_model = None
        loaded_clip = self._ensure_clip(text_encoder_path, "wan")
        loaded_vae = self._ensure_vae(vae_path)
        resource_metadata = {
            "diffusion_model": self.resource_fingerprints.describe(model_path),
            "vae": self.resource_fingerprints.describe(vae_path),
            "text_encoder": self.resource_fingerprints.describe(text_encoder_path),
        }
        if model_low_path is not None:
            resource_metadata["diffusion_model_high"] = self.resource_fingerprints.describe(
                model_high_path
            )
            resource_metadata["diffusion_model_low"] = self.resource_fingerprints.describe(
                model_low_path
            )
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "prompt",
                "total": stage_total,
            }
        )
        positive = self.nodes.CLIPTextEncode().encode(self.clip, command["prompt"])[0]
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
        start_image = self._load_input_image(command.get("input_image_path"))
        if low_model is not None:
            positive, negative, latent = self.WanImageToVideo().execute(
                positive,
                negative,
                self.vae,
                int(command["width"]),
                int(command["height"]),
                int(command["length"]),
                1,
                start_image=start_image,
            )
        else:
            latent = self.Wan22ImageToVideoLatent().execute(
                self.vae,
                int(command["width"]),
                int(command["height"]),
                int(command["length"]),
                1,
                start_image=start_image,
            )[0]
        latent_multiplier = float(command.get("latent_multiplier", 1.0))
        if latent_multiplier != 1.0:
            latent = dict(latent)
            latent["samples"] = latent["samples"] * latent_multiplier
        sampling_model = self.ModelSamplingSD3().patch(
            high_model, float(command.get("shift", 8.0))
        )[0]
        self.torch.cuda.reset_peak_memory_stats()

        sampling_started = time.perf_counter()
        previous_step_at = sampling_started
        sampling_step_offset = 0

        def sampling_callback(step, x0, _x, total_steps):
            nonlocal previous_step_at, sampling_step_offset
            now = time.perf_counter()
            progress = sampling_progress_payload(
                step, total_steps, sampling_started, previous_step_at, now
            )
            progress["step"] = min(int(progress["step"]) + sampling_step_offset, stage_total)
            progress["total"] = stage_total
            previous_step_at = now
            emit(
                {
                    "type": "step_progress",
                    "job_id": job_id,
                    "stage": "sampling",
                    **progress,
                }
            )

        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "sampling",
                "total": stage_total,
            }
        )
        if low_model is None:
            samples = self._sample(
                command,
                latent,
                positive,
                negative,
                sampling_callback,
                model=sampling_model,
                denoise=float(command.get("denoise", 1.0)),
            )
        else:
            split_step = max(1, int(command["steps"]) // 2)
            high_sampling = self.ModelSamplingSD3().patch(
                high_model, float(command.get("shift", 8.0))
            )[0]
            low_sampling = self.ModelSamplingSD3().patch(
                low_model, float(command.get("shift", 8.0))
            )[0]
            samples = self._sample(
                command,
                latent,
                positive,
                negative,
                sampling_callback,
                model=high_sampling,
                denoise=float(command.get("denoise", 1.0)),
                start_step=0,
                end_step=split_step,
                force_full_denoise=False,
            )
            sampling_step_offset = split_step
            samples = self._sample(
                command,
                samples,
                positive,
                negative,
                sampling_callback,
                model=low_sampling,
                denoise=float(command.get("denoise", 1.0)),
                start_step=split_step,
                end_step=int(command["steps"]),
                disable_noise=True,
                force_full_denoise=True,
            )
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
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "video_encoding",
                "total": stage_total,
            }
        )
        output_path = Path(command["output_path"]).resolve()
        performance = {
            "load_seconds": round(load_seconds, 3),
            "sampling_seconds": round(sampling_seconds, 3),
            "vae_seconds": round(vae_seconds, 3),
            "generation_seconds": round(sampling_seconds + vae_seconds, 3),
            "cuda_peak_allocated_gib": round(
                self.torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "cuda_peak_reserved_gib": round(
                self.torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        }
        metadata = {
            "schema_version": 1,
            "generator": {
                "name": "diffusion-workbench",
                "version": command.get("workbench_version"),
            },
            "parameters": {
                "mode": command["video_model"],
                "generation_type": "i2v" if command.get("input_image_path") else "t2v",
                "prompt": command["prompt"],
                "negative_prompt": command.get("negative_prompt", ""),
                "width": int(command["width"]),
                "height": int(command["height"]),
                "duration_seconds": int(command["duration_seconds"]),
                "fps": int(command["fps"]),
                "length": int(command["length"]),
                "steps": int(command["steps"]),
                "seed": int(command["seed"]),
                "cfg": float(command["cfg"]),
                "sampler": command["sampler"],
                "scheduler": command["scheduler"],
                "denoise": float(command.get("denoise", 1.0)),
                "shift": float(command.get("shift", 8.0)),
                "latent_multiplier": float(command.get("latent_multiplier", 1.0)),
                "input_image": command.get("input_image_path"),
            },
            "resources": {
                **resource_metadata,
                "clip_type": "wan",
            },
            "runtime": self.runtime_versions,
            "performance": performance,
        }
        self._save_video(images, output_path, int(command["fps"]), metadata, job_id)
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "video_saved",
                "total": stage_total,
            }
        )
        return {
            "type": "result",
            "job_id": job_id,
            "model_path": str(model_path),
            "output_path": str(output_path),
            "loaded_model": loaded_model,
            "loaded_clip": loaded_clip,
            "loaded_vae": loaded_vae,
            "loaded_resources": self.resource_status(),
            **performance,
        }

    def _generate_minimax_h3(self, command: dict) -> dict:
        """使用 ComfyUI 原生 FL2VA 节点生成带立体声音频的 MiniMax H3 视频。"""
        job_id = command["job_id"]
        requested_mode = command["video_model"]
        if self.mode is not None and self.mode != requested_mode:
            self.release()
        self.mode = requested_mode
        model_path = Path(command["model_path"]).resolve()
        vae_path = Path(command["vae_path"]).resolve()
        audio_vae_path = Path(command["audio_vae_path"]).resolve()
        text_encoder_path = Path(command["text_encoder_path"]).resolve()
        stage_total = int(command["steps"])
        load_started = time.perf_counter()
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "loading_model",
                "total": stage_total,
            }
        )
        loaded_model = self._ensure_model(model_path)
        loaded_clip = self._ensure_clip(text_encoder_path, "minimax")
        loaded_vae = self._ensure_vae(vae_path)
        loaded_audio_vae = self._ensure_audio_vae(audio_vae_path)
        resource_metadata = {
            "diffusion_model": self.resource_fingerprints.describe(model_path),
            "vae": self.resource_fingerprints.describe(vae_path),
            "audio_vae": self.resource_fingerprints.describe(audio_vae_path),
            "text_encoder": self.resource_fingerprints.describe(text_encoder_path),
        }
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "prompt",
                "total": stage_total,
            }
        )
        first_frame = self._load_input_image(command.get("input_image_path"))
        reference_image = self._load_input_image(command.get("reference_image_path"))
        if reference_image is not None:
            positive, latent = self.MiniMaxH3ReferenceToVideo.execute(
                self.clip, self.vae, self.audio_vae, command["prompt"],
                int(command["width"]), int(command["height"]), int(command["length"]),
                ref_images={"ref_image_1": reference_image},
            )
        else:
            positive, latent = self.MiniMaxH3ImageToVideo.execute(
                self.clip, self.vae, command["prompt"],
                int(command["width"]), int(command["height"]), int(command["length"]),
                first_frame=first_frame,
            )
        sampling_model = self.MiniMaxH3SigmaShift.execute(
            self.model,
            float(command.get("shift", 12.0)),
            float(command.get("audio_shift", 3.0)),
        )[0]
        load_seconds = time.perf_counter() - load_started
        self.torch.cuda.reset_peak_memory_stats()
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "sampling",
                "total": stage_total,
            }
        )
        sampling_started = time.perf_counter()
        previous_step_at = sampling_started

        def sampling_callback(step, x0, _x, total_steps):
            nonlocal previous_step_at
            now = time.perf_counter()
            progress = sampling_progress_payload(
                step, total_steps, sampling_started, previous_step_at, now
            )
            previous_step_at = now
            emit(
                {
                    "type": "step_progress",
                    "job_id": job_id,
                    "stage": "sampling",
                    **progress,
                }
            )

        samples = self._sample(
            command,
            latent,
            positive,
            positive,
            sampling_callback,
            model=sampling_model,
        )
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
        audio = self.VAEDecodeAudio.execute(self.audio_vae, samples)[0]
        vae_seconds = time.perf_counter() - vae_started
        performance = {
            "load_seconds": round(load_seconds, 3),
            "sampling_seconds": round(sampling_seconds, 3),
            "vae_seconds": round(vae_seconds, 3),
            "generation_seconds": round(sampling_seconds + vae_seconds, 3),
            "cuda_peak_allocated_gib": round(
                self.torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "cuda_peak_reserved_gib": round(
                self.torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        }
        metadata = self._video_metadata(command, resource_metadata, performance)
        output_path = Path(command["output_path"]).resolve()
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "video_encoding",
                "total": stage_total,
            }
        )
        self._save_video(
            images, output_path, int(command["fps"]), metadata, job_id, audio=audio
        )
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "video_saved",
                "total": stage_total,
            }
        )
        return {
            "type": "result",
            "job_id": job_id,
            "model_path": str(model_path),
            "output_path": str(output_path),
            "loaded_model": loaded_model,
            "loaded_clip": loaded_clip,
            "loaded_vae": loaded_vae,
            "loaded_audio_vae": loaded_audio_vae,
            "loaded_resources": self.resource_status(),
            **performance,
        }

    def _video_metadata(self, command: dict, resource_metadata: dict, performance: dict) -> dict:
        return {
            "schema_version": 1,
            "generator": {
                "name": "diffusion-workbench",
                "version": command.get("workbench_version"),
            },
            "parameters": {
                "mode": command["video_model"],
                "generation_type": "r2v" if command.get("reference_image_path") else "i2v" if command.get("input_image_path") else "t2v",
                "prompt": command["prompt"],
                "negative_prompt": command.get("negative_prompt", ""),
                "width": int(command["width"]),
                "height": int(command["height"]),
                "duration_seconds": int(command["duration_seconds"]),
                "fps": int(command["fps"]),
                "length": int(command["length"]),
                "steps": int(command["steps"]),
                "seed": int(command["seed"]),
                "cfg": float(command["cfg"]),
                "sampler": command["sampler"],
                "scheduler": command["scheduler"],
                "denoise": float(command.get("denoise", 1.0)),
                "shift": float(command.get("shift", 8.0)),
                "audio_shift": float(command.get("audio_shift", 3.0)),
                "latent_multiplier": float(command.get("latent_multiplier", 1.0)),
                "input_image": command.get("input_image_path"),
                "reference_image": command.get("reference_image_path"),
            },
            "resources": {
                **resource_metadata,
                "clip_type": command["clip_type"],
            },
            "runtime": self.runtime_versions,
            "performance": performance,
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
        model=None,
        denoise=1.0,
        end_step=None,
        disable_noise=False,
        force_full_denoise=None,
    ):
        sampling = sampling or command
        model = model or self.model
        latent_image = self.comfy_sample.fix_empty_latent_channels(
            model,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        seed = int(sampling["seed"])
        if disable_noise:
            noise = self.torch.zeros(
                latent_image.size(),
                dtype=latent_image.dtype,
                layout=latent_image.layout,
                device="cpu",
            )
        else:
            noise = self.comfy_sample.prepare_noise(
                latent_image, seed, latent.get("batch_index")
            )
        sampled = self.comfy_sample.sample(
            model,
            noise,
            int(sampling["steps"]),
            float(sampling["cfg"]),
            sampling["sampler"],
            sampling["scheduler"],
            positive,
            negative,
            latent_image,
            noise_mask=latent.get("noise_mask"),
            denoise=float(denoise),
            start_step=start_step,
            last_step=end_step,
            disable_noise=disable_noise,
            force_full_denoise=(start_step is not None if force_full_denoise is None else force_full_denoise),
            callback=callback,
            disable_pbar=True,
            seed=seed,
        )
        output = latent.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = sampled
        return output

    def _load_input_image(self, path: str | None):
        if not path:
            return None
        image_path = Path(path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"找不到输入图片: {image_path}")
        with self.Image.open(image_path) as source:
            image = self.ImageOps.exif_transpose(source).convert("RGB")
        try:
            pixels = self.numpy.array(image, dtype=self.numpy.float32) / 255.0
            return self.torch.from_numpy(pixels)[None, ...]
        finally:
            image.close()

    def _save_video(
        self, images, output_path: Path, fps: int, metadata: dict, job_id: str, audio=None
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{job_id}.tmp.mp4")
        components = self.Types.VideoComponents(
            images=images,
            frame_rate=Fraction(fps),
            audio=audio,
        )
        video = self.InputImpl.VideoFromComponents(components, bit_depth=8)
        try:
            video.save_to(
                str(temporary_path),
                format=self.Types.VideoContainer.MP4,
                codec=self.Types.VideoCodec.H264,
                metadata={"diffusion_workbench": metadata},
            )
            os.link(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

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
        self.model = self._load_diffusion_model(path)
        self.model_path = path
        return True

    def _ensure_video_models(self, high_path: Path, low_path: Path) -> bool:
        if (
            self.model_high is not None
            and self.model_low is not None
            and self.model_high_path == high_path
            and self.model_low_path == low_path
        ):
            self.model = self.model_high
            self.model_path = high_path
            return False
        self._unload_gpu()
        self.model = self.model_high = self.model_low = None
        self.model_path = self.model_high_path = self.model_low_path = None
        gc.collect()
        self.model_high = self._load_diffusion_model(high_path)
        self.model_low = self._load_diffusion_model(low_path)
        self.model = self.model_high
        self.model_path = self.model_high_path = high_path
        self.model_low_path = low_path
        return True

    def _load_diffusion_model(self, path: Path):
        name = self._register_exact("diffusion_models", path)
        if path.suffix.casefold() != ".gguf":
            return self.nodes.UNETLoader().load_unet(name, "default")[0]
        loader_class = self._ensure_gguf_unet_loader()
        return loader_class().load_unet(name)[0]

    def _ensure_gguf_unet_loader(self):
        if self.gguf_unet_loader_class is not None:
            return self.gguf_unet_loader_class
        node_root = self.comfy_root / "custom_nodes" / "ComfyUI-GGUF"
        entrypoint = node_root / "__init__.py"
        if not entrypoint.is_file():
            raise RuntimeError(
                f"加载 GGUF diffusion 模型需要安装 ComfyUI-GGUF: {node_root}"
            )
        module_name = "diffusion_workbench_comfyui_gguf"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(
                module_name,
                entrypoint,
                submodule_search_locations=[str(node_root)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"无法加载 ComfyUI-GGUF 节点: {entrypoint}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        try:
            loader_class = module.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError("ComfyUI-GGUF 未提供 UnetLoaderGGUF 节点") from exc
        self.gguf_unet_loader_class = loader_class
        return loader_class

    def _ensure_checkpoint(self, path: Path) -> bool:
        if (
            self.checkpoint_path == path
            and self.model is not None
            and self.clip is not None
            and self.vae is not None
        ):
            return False
        self._unload_gpu()
        self.model = self.clip = self.vae = None
        self.model_path = self.clip_path = self.vae_path = None
        self.checkpoint_path = None
        gc.collect()
        name = self._register_exact("checkpoints", path)
        loader = self.nodes.CheckpointLoaderSimple()
        self.model, self.clip, self.vae = loader.load_checkpoint(name)
        self.model_path = path
        self.checkpoint_path = path
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

    def _ensure_audio_vae(self, path: Path) -> bool:
        if self.audio_vae is not None and self.audio_vae_path == path:
            return False
        self._unload_gpu()
        self.audio_vae = None
        self.audio_vae_path = None
        gc.collect()
        name = self._register_exact("vae", path)
        self.audio_vae = self.nodes.VAELoader().load_vae(name)[0]
        self.audio_vae_path = path
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
        self.model_high = None
        self.model_low = None
        self.clip = None
        self.vae = None
        self.audio_vae = None
        self.upscale_model = None
        self.model_path = self.clip_path = self.vae_path = None
        self.audio_vae_path = None
        self.model_high_path = self.model_low_path = None
        self.checkpoint_path = None
        self.upscale_model_path = None
        self.clip_type = None
        self.mode = None
        gc.collect()
        self.model_management.cleanup_models_gc()
        self.model_management.cleanup_models()
        self.model_management.soft_empty_cache(force=True)

    def resource_status(self) -> dict:
        return {
            "workload": self.mode,
            "model": str(self.model_path) if self.model_path else None,
            "vae": str(self.vae_path) if self.vae_path else None,
            "text_encoder": str(self.clip_path) if self.clip_path else None,
            "audio_vae": str(self.audio_vae_path) if self.audio_vae_path else None,
            "clip_type": self.clip_type,
        }

    @staticmethod
    def _validate(command: dict) -> None:
        try:
            Mode(command.get("mode"))
        except ValueError:
            raise ValueError(f"不支持 mode: {command.get('mode')}") from None
        if "model_loader" in command:
            model_loader = ModelLoader(command["model_loader"])
            if model_loader == ModelLoader.CHECKPOINT:
                if command.get("vae_path") or command.get("text_encoder_path"):
                    raise ValueError("checkpoint loader 不接受外置 VAE 或文本编码器")
            elif not command.get("vae_path") or not command.get("text_encoder_path"):
                raise ValueError("components loader 必须提供 VAE 和文本编码器")
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

    def _validate_video(self, command: dict) -> None:
        try:
            video_model = VideoModel(command.get("video_model"))
        except ValueError:
            raise ValueError(f"不支持视频模型: {command.get('video_model')}") from None
        expected_clip_type = (
            "minimax" if video_model == VideoModel.MINIMAX_H3 else "wan"
        )
        if command.get("clip_type") != expected_clip_type:
            raise ValueError(
                f"{video_model.value} clip_type 必须为 {expected_clip_type}"
            )
        width, height = int(command["width"]), int(command["height"])
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("视频宽高必须为正数且是 16 的倍数")
        duration = int(command["duration_seconds"])
        fps = int(command["fps"])
        length = int(command["length"])
        if duration <= 0 or fps <= 0 or fps > 120:
            raise ValueError("视频时长必须为正数，帧率必须在 1 到 120 之间")
        if video_model == VideoModel.MINIMAX_H3:
            if width % 32 or height % 32:
                raise ValueError("MiniMax H3 视频宽高必须是 32 的倍数")
            expected_length = max(5, round(duration * 24))
            while expected_length % 17 != 5:
                expected_length += 1
            if fps != 24:
                raise ValueError("MiniMax H3 帧率固定为 24")
            if length != expected_length or length % 17 != 5:
                raise ValueError("MiniMax H3 length 必须按 17n + 5 对齐")
            if float(command["cfg"]) != 1.0:
                raise ValueError("MiniMax H3 CFG 固定为 1")
            if not command.get("audio_vae_path"):
                raise ValueError("MiniMax H3 必须提供音频 VAE")
            has_reference = bool(command.get("reference_image_path"))
            model_path = command.get("model_path", "")
            model_is_ref2va = "ref2va" in Path(model_path).name.casefold()
            if model_is_ref2va != has_reference:
                raise ValueError("MiniMax H3 Ref2VA 必须使用 ref2va 模型和参考图")
            if has_reference and command.get("input_image_path"):
                raise ValueError("MiniMax H3 首帧输入与参考图不能同时提供")
        else:
            if length != duration * fps + 1:
                raise ValueError("视频 length 必须等于 duration_seconds * fps + 1")
            if (length - 1) % 4:
                raise ValueError("视频总帧数必须满足 length = 4n + 1")
        if float(command.get("denoise", 1.0)) != 1.0:
            raise ValueError("视频 denoise 固定为 1")
        if video_model == VideoModel.WAN22_I2V_14B:
            if not command.get("model_high_path") or not command.get("model_low_path"):
                raise ValueError("Wan I2V-14B 必须同时提供 high_noise 和 low_noise 模型")
            if not command.get("input_image_path"):
                raise ValueError("Wan I2V-14B 必须提供输入图片")
        shift = float(command.get("shift", 8.0))
        if not math.isfinite(shift) or not 0.0 <= shift <= 100.0:
            raise ValueError("shift 必须在 0 到 100 之间")
        latent_multiplier = float(command.get("latent_multiplier", 1.0))
        if not math.isfinite(latent_multiplier) or latent_multiplier <= 0:
            raise ValueError("latent_multiplier 必须是大于 0 的有限数值")
        validate_sampling(
            int(command["steps"]),
            float(command["cfg"]),
            command.get("sampler"),
            command.get("scheduler"),
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
        if command_type == "release":
            try:
                worker.release()
                emit({"type": "released"})
            except Exception as exc:
                emit({"type": "error", "job_id": None, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if command_type not in {"generate", "generate_video"}:
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
            resources_released = False
            # 视频任务通常同时持有 high/low 两个大模型。取消后立即释放，
            # 避免下一个图片任务切模时继承残留的显存状态。
            if command.get("type") == "generate_video":
                try:
                    worker.release()
                    resources_released = True
                except Exception:
                    logging.exception("视频任务取消后的资源清理失败")
            payload = {
                "type": "cancelled",
                "job_id": command.get("job_id"),
                "model_path": str(worker.model_path) if worker.model_path else None,
                "resources_released": resources_released,
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
