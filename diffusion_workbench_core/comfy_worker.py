"""Long-lived headless ComfyUI worker. Run only with ComfyUI's Python."""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path


EVENT_PREFIX = "DWB_EVENT="


def emit(payload: dict) -> None:
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    return parser


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
        import nodes
        import torch
        from PIL import Image

        if "euler" not in comfy.samplers.KSampler.SAMPLERS:
            raise RuntimeError("当前 ComfyUI 不支持 Euler sampler")
        if "simple" not in comfy.samplers.KSampler.SCHEDULERS:
            raise RuntimeError("当前 ComfyUI 不支持 simple scheduler")

        self.folder_paths = folder_paths
        self.model_management = comfy.model_management
        self.comfy_sample = comfy.sample
        self.nodes = nodes
        self.torch = torch
        self.Image = Image
        self.model = None
        self.model_path: Path | None = None
        self.clip = None
        self.clip_path: Path | None = None
        self.clip_type: str | None = None
        self.vae = None
        self.vae_path: Path | None = None
        self.mode: str | None = None

    def generate(self, command: dict) -> dict:
        self._validate(command)
        with self.torch.inference_mode():
            return self._generate(command)

    def _generate(self, command: dict) -> dict:
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
        emit({"type": "stage_progress", "job_id": job_id, "stage": "loading_model"})
        loaded_model = self._ensure_model(model_path)
        loaded_clip = self._ensure_clip(text_encoder_path, clip_type)
        loaded_vae = self._ensure_vae(vae_path)

        emit({"type": "stage_progress", "job_id": job_id, "stage": "prompt"})
        positive = self.nodes.CLIPTextEncode().encode(self.clip, command["prompt"])[0]
        negative = self.nodes.ConditioningZeroOut().zero_out(positive)[0]
        load_seconds = time.perf_counter() - load_started

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

        def sampling_callback(step, _x0, _x, total_steps):
            emit(
                {
                    "type": "step_progress",
                    "job_id": job_id,
                    "step": int(step) + 1,
                    "total": int(total_steps),
                }
            )

        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "sampling",
                "total": int(command["steps"]),
            }
        )
        generation_started = time.perf_counter()
        samples = self._sample(command, latent, positive, negative, sampling_callback)
        emit({"type": "stage_progress", "job_id": job_id, "stage": "vae"})
        images = self.nodes.VAEDecode().decode(self.vae, samples)[0]
        generation_seconds = time.perf_counter() - generation_started

        pixels = images[0].detach().cpu().clamp(0, 1).mul(255).byte().numpy()
        output_path = Path(command["output_path"]).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{job_id}.tmp")
        try:
            self.Image.fromarray(pixels).save(temporary_path, format="PNG")
            os.link(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        emit({"type": "stage_progress", "job_id": job_id, "stage": "saved"})
        return {
            "type": "result",
            "job_id": job_id,
            "model_path": str(model_path),
            "vae_path": str(vae_path),
            "output_path": str(output_path),
            "loaded_model": loaded_model,
            "loaded_clip": loaded_clip,
            "loaded_vae": loaded_vae,
            "load_seconds": round(load_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "cuda_peak_allocated_gib": round(
                self.torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "cuda_peak_reserved_gib": round(
                self.torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        }

    def _sample(self, command, latent, positive, negative, callback):
        latent_image = self.comfy_sample.fix_empty_latent_channels(
            self.model,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        seed = int(command["seed"])
        noise = self.comfy_sample.prepare_noise(latent_image, seed, None)
        sampled = self.comfy_sample.sample(
            self.model,
            noise,
            int(command["steps"]),
            1.0,
            "euler",
            "simple",
            positive,
            negative,
            latent_image,
            denoise=1.0,
            callback=callback,
            disable_pbar=True,
            seed=seed,
        )
        output = latent.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = sampled
        return output

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

    def release(self) -> None:
        self._unload_gpu()
        self.model = None
        self.clip = None
        self.vae = None
        self.model_path = self.clip_path = self.vae_path = None
        self.clip_type = None
        self.mode = None
        gc.collect()
        self.model_management.cleanup_models_gc()
        self.model_management.cleanup_models()
        self.model_management.soft_empty_cache(force=True)

    @staticmethod
    def _validate(command: dict) -> None:
        if command.get("mode") not in {"zit", "krea2"}:
            raise ValueError("mode 必须是 zit 或 krea2")
        if command.get("sampler") != "euler" or command.get("scheduler") != "simple":
            raise ValueError("当前只支持 Euler + simple")
        if float(command.get("cfg", 0)) != 1.0:
            raise ValueError("CFG 固定为 1")
        width, height = int(command["width"]), int(command["height"])
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("图片宽高必须为正数且是 16 的倍数")
        steps = int(command["steps"])
        if not 8 <= steps <= 20:
            raise ValueError("steps 必须在 8 到 20 之间")


def main() -> None:
    args = build_parser().parse_args()
    try:
        worker = ComfyWorker(args.comfy_root)
    except Exception as exc:
        emit({"type": "startup_error", "error": f"{type(exc).__name__}: {exc}"})
        raise
    emit({"type": "ready"})
    for line in sys.stdin:
        command = {}
        try:
            command = json.loads(line)
            if command.get("type") == "shutdown":
                worker.release()
                emit({"type": "stopped"})
                return
            if command.get("type") != "generate":
                raise ValueError("未知 Worker 命令")
            emit(worker.generate(command))
        except Exception as exc:
            traceback_text = traceback.format_exc()
            try:
                worker.release()
            except Exception as cleanup_exc:
                logging.exception("生成失败后的资源清理也失败: %s", cleanup_exc)
            emit(
                {
                    "type": "error",
                    "job_id": command.get("job_id") if isinstance(command, dict) else None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback_text,
                }
            )


if __name__ == "__main__":
    main()
