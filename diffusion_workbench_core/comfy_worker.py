"""Long-lived headless ComfyUI worker. Run only with ComfyUI's Python."""

import argparse
import base64
import gc
import hashlib
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
from collections import OrderedDict
from fractions import Fraction
from io import BytesIO
from pathlib import Path

try:
    from .domain import (
        SAMPLERS,
        SCHEDULERS,
        MINIMAX_H3_MODELS,
        Mode,
        ModelLoader,
        UpscaleMethod,
        UpscaleSettings,
        VideoModel,
        is_h3_ref2va_model_name,
        validate_sampling,
        H3_MAX_PIXELS,
        H3_MAX_SIZE,
        H3_MIN_SIZE,
        H3_REF2VA_MAX_AUDIOS,
        H3_REF2VA_MAX_IMAGES,
        H3_REF2VA_MAX_TOTAL,
        H3_REF2VA_MAX_VIDEOS,
        REBALANCE_MAX_REFERENCE_IMAGES,
        REBALANCE_TOKEN_TIERS,
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
        MINIMAX_H3_MODELS,
        Mode,
        ModelLoader,
        UpscaleMethod,
        UpscaleSettings,
        VideoModel,
        is_h3_ref2va_model_name,
        validate_sampling,
        H3_MAX_PIXELS,
        H3_MAX_SIZE,
        H3_MIN_SIZE,
        H3_REF2VA_MAX_AUDIOS,
        H3_REF2VA_MAX_IMAGES,
        H3_REF2VA_MAX_TOTAL,
        H3_REF2VA_MAX_VIDEOS,
        REBALANCE_MAX_REFERENCE_IMAGES,
        REBALANCE_TOKEN_TIERS,
    )
    from png_metadata import (
        ResourceFingerprintCache,
        build_generation_metadata,
        create_png_info,
    )


EVENT_PREFIX = "DWB_EVENT="

# MiniMax H3 Turbo 采样配方（参考社区 lightx2v turbo 工作流）：
# 蒸馏 LoRA 常驻 strength=0.5；euler + BetaSamplingScheduler(alpha, beta)
# 生成 sigma 序列，再用 ExtendIntermediateSigmas 在低 sigma 区间[0, 0.8]
# 的每个间隔内按线性插入 3 个中间步。该配方替代裸模型的 MiniMaxH3SigmaShift。
H3_TURBO_LORA_STRENGTH = 0.5
H3_TURBO_BETA_ALPHA = 0.79
H3_TURBO_BETA_BETA = 0.5
H3_TURBO_EXTEND_STEPS = 3
H3_TURBO_EXTEND_START_SIGMA = 0.8
H3_TURBO_EXTEND_END_SIGMA = 0.0


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


class _LruTensorCache:
    """按 LRU 淘汰的有限容量缓存，用于输入图片解码结果与图片 conditioning。

    键为任意可哈希元组；值附带估算字节数，超过单条目或总预算时直接不缓存。
    """

    def __init__(self, max_items: int, max_bytes: int):
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict = OrderedDict()
        self._total_bytes = 0

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key, value, size_bytes: int) -> bool:
        size_bytes = int(size_bytes)
        if size_bytes < 0 or size_bytes > self.max_bytes:
            return False
        existing = self._entries.pop(key, None)
        if existing is not None:
            self._total_bytes -= existing[1]
        while self._entries and (
            len(self._entries) >= self.max_items
            or self._total_bytes + size_bytes > self.max_bytes
        ):
            _, (_, evicted_bytes) = self._entries.popitem(last=False)
            self._total_bytes -= evicted_bytes
        self._entries[key] = (value, size_bytes)
        self._total_bytes += size_bytes
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._total_bytes = 0


def _is_tensor_like(value) -> bool:
    return (
        hasattr(value, "nelement")
        and hasattr(value, "element_size")
        and hasattr(value, "clone")
    )


def _clone_conditioning_structure(value):
    """结构化深拷贝 conditioning：张量 clone，dict/list/tuple 递归复制，其余对象原样共享。"""

    if _is_tensor_like(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_conditioning_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_conditioning_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_conditioning_structure(item) for item in value)
    return value


def _estimate_conditioning_bytes(value) -> int:
    """估算 conditioning 中张量占用的字节数；无法识别的结构按 0 计。"""

    if _is_tensor_like(value):
        return int(value.nelement()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_estimate_conditioning_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_conditioning_bytes(item) for item in value)
    return 0


def _image_tensor_fingerprint(tensor) -> tuple:
    """对归一化后的 CPU RGB tensor 计算内容指纹：形状 + SHA-256 摘要。

    指纹基于 EXIF 纠正后的实际像素，因此同一文件内容或 EXIF 方向变化必然
    产生不同指纹。
    """

    shape = tuple(int(dim) for dim in tensor.shape)
    digest = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
    return (shape, digest)


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
        import comfy.model_prefetch
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import comfyui_version
        import latent_preview
        import nodes
        import torch
        import numpy
        try:
            from comfy_aimdo import model_vbar
        except ImportError:  # 非 aimdo 加速构建的 ComfyUI 没有 vbar 水位管理
            model_vbar = None
        from comfy_api.latest import InputImpl, Types
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_custom_sampler import (
            BasicGuider,
            BetaSamplingScheduler,
            ExtendIntermediateSigmas,
            KSamplerSelect,
            RandomNoise,
        )
        from comfy_extras.nodes_minimax_h3 import (
            EmptyMiniMaxH3LatentAV,
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
        self.model_prefetch = comfy.model_prefetch
        self.model_vbar = model_vbar
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
        self.BasicGuider = BasicGuider
        self.BetaSamplingScheduler = BetaSamplingScheduler
        self.ExtendIntermediateSigmas = ExtendIntermediateSigmas
        self.KSamplerSelect = KSamplerSelect
        self.RandomNoise = RandomNoise
        self.MiniMaxH3ImageToVideo = MiniMaxH3ImageToVideo
        self.MiniMaxH3ReferenceToVideo = MiniMaxH3ReferenceToVideo
        self.EmptyMiniMaxH3LatentAV = EmptyMiniMaxH3LatentAV
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
        # Krea2Edit 自定义节点包懒加载缓存（仅 edit-krea2 任务触发）。
        self.krea2edit_module = None
        # Conditioning-Rebalance 自定义节点包懒加载缓存（仅 krea2-rebalance 任务触发）。
        self.rebalance_module = None
        self.model = None
        self.model_path: Path | None = None
        self.model_high = None
        self.model_low = None
        self.model_high_path: Path | None = None
        self.model_low_path: Path | None = None
        self.clip = None
        self.clip_path: Path | None = None
        self.clip_type: str | None = None
        # 仅缓存不含图像/视频条件的文本 conditioning，避免重复执行昂贵的文本编码。
        self._conditioning_cache: dict[tuple, tuple] = {}
        # 输入图片解码缓存：键为 (绝对路径, 文件大小, mtime_ns)，值为归一化
        # CPU RGB tensor 与内容指纹，避免相同文件重复读取/EXIF 转换/PIL 解码。
        self._image_cache = _LruTensorCache(max_items=4, max_bytes=256 * 1024 * 1024)
        # 图片 conditioning 缓存：只缓存完整 positive，latent 仍每个任务独立创建。
        self._image_conditioning_cache = _LruTensorCache(
            max_items=2, max_bytes=512 * 1024 * 1024
        )
        self.vae = None
        self.vae_path: Path | None = None
        self.audio_vae = None
        self.audio_vae_path: Path | None = None
        self.mode: str | None = None
        self.checkpoint_path: Path | None = None
        self.upscale_model = None
        self.upscale_model_path: Path | None = None

    def _comfy_execution_cleanup(self) -> None:
        """对齐 ComfyUI execution.py 每个 prompt 结束后的全局收尾。

        headless Worker 直接调用节点、不经过 execution.py，这些清理从未执行；
        缺失时 CROSS_STEP_STATE / PREFETCH_QUEUES 等全局状态会挂住已卸载模型的
        mmap 与 host 缓冲（实测 minimax 32B 文本编码器约 15GB 的 mmap 无法关闭），
        即使 release() 丢弃了全部模型引用也回收不了对应 RAM。
        """
        self.model_management.reset_cast_buffers()
        self.model_prefetch.cleanup_prefetch_queues()
        if self.model_vbar is not None:
            self.model_vbar.vbars_reset_watermark_limits()

    def generate(self, command: dict) -> dict:
        if command.get("type") == "generate_video":
            self._validate_video(command)
        else:
            self._validate(command)
        self._validate_runtime_sampling(command)
        with self.torch.inference_mode():
            try:
                if command.get("type") == "generate_video":
                    return self._generate_video(command)
                return self._generate(command)
            finally:
                self._comfy_execution_cleanup()

    def describe_image(self, command: dict) -> dict:
        """图片反推：复用 krea2 的 Qwen3-VL clip，把图片与系统提示词转成英文描述文本。

        同步命令，不产图片也不落盘；走 ComfyUI 核心 TextGenerate 同款路径
        （clip.tokenize(image=...) → clip.generate → clip.decode）。
        """
        job_id = command["job_id"]
        text_encoder_path = Path(command["text_encoder_path"]).resolve()
        prompt = str(command["prompt"])
        max_length = max(1, int(command.get("max_length", 2048)))
        seed = int(command.get("seed", 0))
        with self.torch.inference_mode():
            try:
                # 与 _generate 开头的 mode 检查一致：从其他功能切来时先整体释放旧资源；
                # mode 记为 krea2，保证之后切去别的模式时 Qwen3-VL 会被 release 掉。
                cache_mode = "krea2"
                if self.mode is not None and self.mode != cache_mode:
                    self.release()
                self.mode = cache_mode
                load_started = time.perf_counter()
                self._ensure_clip(text_encoder_path, "krea2")
                load_seconds = time.perf_counter() - load_started
                image = self._load_input_image(command.get("image_path"))
                self.torch.cuda.reset_peak_memory_stats()
                infer_started = time.perf_counter()
                tokens = self.clip.tokenize(prompt, image=image, min_length=1)
                generated_ids = self.clip.generate(
                    tokens,
                    do_sample=True,
                    max_length=max_length,
                    temperature=0.7,
                    top_k=64,
                    top_p=0.95,
                    min_p=0.05,
                    repetition_penalty=1.05,
                    seed=seed,
                )
                caption = self.clip.decode(generated_ids)
                infer_seconds = time.perf_counter() - infer_started
            finally:
                self._comfy_execution_cleanup()
        return {
            "type": "result",
            "job_id": job_id,
            "caption": caption,
            "loaded_resources": self.resource_status(),
            "load_seconds": round(load_seconds, 3),
            "infer_seconds": round(infer_seconds, 3),
            "cuda_peak_allocated_gib": round(
                self.torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
        }

    def _generate(self, command: dict) -> dict:
        command = dict(command)
        command["upscale"] = resolve_upscale_settings(command)
        job_id = command["job_id"]
        requested_mode = command["mode"]
        is_edit = requested_mode == "edit-krea2"
        is_rebalance = requested_mode == "krea2-rebalance"
        # edit-krea2 与 krea2-rebalance 都复用 krea2 的 diffusion/clip/vae 缓存，避免来回切换重载。
        cache_mode = "krea2" if is_edit or is_rebalance else requested_mode
        if self.mode is not None and self.mode != cache_mode:
            self.release()
        self.mode = cache_mode
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
        cfg_value = float(command["cfg"])
        if is_edit:
            # 编辑模式不走纯文本编码：用 Krea2EditGroundedEncode 把指令与源图一起
            # 经 Qwen3-VL 编码，得到 image-grounded 的 conditioning。
            # 双图编辑时第二张图经 image_b 传入，由节点内部拼入 grounded prompt。
            source_image = self._load_input_image(command.get("input_image_path"))
            source_image_b = self._load_input_image(
                command.get("secondary_input_image_path")
            )
            edit_module = self._ensure_krea2edit_nodes()
            grounded = edit_module.NODE_CLASS_MAPPINGS["Krea2EditGroundedEncode"]()
            positive = grounded.encode(
                self.clip,
                command["prompt"],
                image=source_image,
                image_b=source_image_b,
                grounding_px=int(command.get("grounding_px", 768)),
                system_prompt="",
            )[0]
            negative_prompt = command.get("negative_prompt", "")
            if cfg_value == 1.0:
                # CFG 1 复用 positive，与普通图任务的 reuse_negative_at_cfg_one 一致
                negative = positive
            elif not negative_prompt:
                # 对应原 workflow：negative 端用 ConditioningZeroOut(conditioning=positive)
                negative = self.nodes.ConditioningZeroOut().zero_out(positive)[0]
            else:
                negative = grounded.encode(
                    self.clip,
                    negative_prompt,
                    image=source_image,
                    image_b=source_image_b,
                    grounding_px=int(command.get("grounding_px", 768)),
                    system_prompt="",
                )[0]
        elif is_rebalance:
            # 参考图重排不走 Krea2Edit 节点：用 Krea2EncodeRebalance 把提示词与
            # 1-4 张参考图一起经 Qwen3-VL 编码为 conditioning，模型与采样流程不变。
            rebalance_module = self._ensure_rebalance_nodes()
            encoder = rebalance_module.NODE_CLASS_MAPPINGS["Krea2EncodeRebalance"]()
            reference_paths = command.get("reference_image_paths") or []
            reference_tokens = list(command.get("reference_image_tokens") or ())
            if not reference_tokens:
                reference_tokens = ["normal"] * len(reference_paths)
            encode_kwargs = {}
            for slot, (path, tier) in enumerate(
                zip(reference_paths, reference_tokens), start=1
            ):
                encode_kwargs[f"image{slot}"] = self._load_input_image(path)
                encode_kwargs[f"image{slot}_tokens"] = tier
            positive = encoder.main(command["prompt"], self.clip, **encode_kwargs)[0]
            negative_prompt = command.get("negative_prompt", "")
            if cfg_value == 1.0:
                # CFG 1 复用 positive，与普通图任务的 reuse_negative_at_cfg_one 一致
                negative = positive
            else:
                negative = self.nodes.CLIPTextEncode().encode(
                    self.clip, negative_prompt
                )[0]
        else:
            positive, negative = self._encode_text_conditioning(
                command["prompt"],
                command.get("negative_prompt", ""),
                cfg=cfg_value,
                cache_scope=f"image:{command['mode']}",
                reuse_negative_at_cfg_one=command["mode"] != "zib",
            )
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

        sampling_model = self.model
        if is_edit:
            # 按参考 workflow：先对源图做 VAEEncode 拿到 source_latent，再走
            # Krea2EditModelPatch 注入 krea2_edit in-context forward。
            # 注意：fit_mode="fit" 且接了 vae+source_image+target_latent 时，patch 节点
            # 内部会按目标网格从 source_image 重新编码（pixel path），这里产出的
            # source_latent 实际不会被采样使用。因此先把源图缩放到输出尺寸再编码，
            # 避免对几千像素的大图做一次结果被丢弃的全分辨率 VAEEncode。
            edit_module = self._ensure_krea2edit_nodes()
            encode_image = self._downscale_image_to_target(source_image, width, height)
            source_latent = self.nodes.VAEEncode().encode(self.vae, encode_image)[0]
            # 双图编辑：第二图同样先缩放到输出尺寸再编码，避免全分辨率 VAEEncode。
            source_latent_b = None
            if source_image_b is not None:
                encode_image_b = self._downscale_image_to_target(
                    source_image_b, width, height
                )
                source_latent_b = self.nodes.VAEEncode().encode(
                    self.vae, encode_image_b
                )[0]
            edit_lora_path = Path(command["edit_lora_path"]).resolve()
            lora_name = self._register_exact("loras", edit_lora_path)
            lora_model = self.nodes.LoraLoaderModelOnly().load_lora_model_only(
                self.model, lora_name, 1.0
            )[0]
            resource_metadata["edit_lora"] = self.resource_fingerprints.describe(
                edit_lora_path
            )
            patcher = edit_module.NODE_CLASS_MAPPINGS["Krea2EditModelPatch"]()
            sampling_model = patcher.patch(
                lora_model,
                source_latent,
                source_latent_b=source_latent_b,
                ref_boost=float(command.get("ref_boost", 1.0)),
                ref_boost_a=1.0,
                fit_mode="fit",
                vae=self.vae,
                source_image=source_image,
                source_image_b=source_image_b,
                target_latent=latent,
            )[0]

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
        samples = self._sample(
            command,
            latent,
            positive,
            negative,
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
        if VideoModel(command["video_model"]) in MINIMAX_H3_MODELS:
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
        positive, negative = self._encode_text_conditioning(
            command["prompt"],
            command.get("negative_prompt", ""),
            cfg=float(command["cfg"]),
            cache_scope=f"video:{command['video_model']}",
            reuse_negative_at_cfg_one=False,
        )
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
        first_frame, first_frame_fingerprint = self._load_input_image_cached(
            command.get("input_image_path")
        )
        last_frame, last_frame_fingerprint = self._load_input_image_cached(
            command.get("last_frame_image_path")
        )
        width = int(command["width"])
        height = int(command["height"])
        length = int(command["length"])
        if VideoModel(command["video_model"]) == VideoModel.MINIMAX_H3_REF2VA:
            # Ref2VA 多参考输入（图片/视频/音频），不与 FL2VA 共用缓存条目。
            positive = self._get_h3_ref2va_conditioning(command)
            # 采样 latent 必须每个任务独立创建，不复用缓存。
            latent = self.EmptyMiniMaxH3LatentAV.execute(width, height, length)[0]
        elif first_frame is None and last_frame is None:
            # 纯文生视频没有图像条件，可以复用文本 conditioning；latent 每次重新创建。
            positive = self._encode_h3_text_conditioning(command["prompt"])
            latent = self.EmptyMiniMaxH3LatentAV.execute(width, height, length)[0]
        else:
            positive = self._get_h3_fl2va_conditioning(
                command,
                first_frame,
                first_frame_fingerprint,
                last_frame,
                last_frame_fingerprint,
            )
            latent = self.EmptyMiniMaxH3LatentAV.execute(width, height, length)[0]
        is_turbo = VideoModel(command["video_model"]) == VideoModel.MINIMAX_H3_TURBO
        turbo_sampler = None
        turbo_sigmas = None
        if is_turbo:
            turbo_lora_path = command.get("turbo_lora_path")
            if not turbo_lora_path:
                raise ValueError("minimax-h3-turbo 缺少 turbo_lora 配置")
            turbo_lora_path = Path(turbo_lora_path).resolve()
            lora_name = self._register_exact("loras", turbo_lora_path)
            # LoRA 只 patch 本次采样使用的模型副本，不污染缓存的基础模型
            sampling_model = self.nodes.LoraLoaderModelOnly().load_lora_model_only(
                self.model, lora_name, H3_TURBO_LORA_STRENGTH
            )[0]
            resource_metadata["turbo_lora"] = self.resource_fingerprints.describe(
                turbo_lora_path
            )
            turbo_sampler = self.KSamplerSelect.get_sampler("euler")[0]
            turbo_sigmas = self.BetaSamplingScheduler.get_sigmas(
                sampling_model,
                int(command["steps"]),
                H3_TURBO_BETA_ALPHA,
                H3_TURBO_BETA_BETA,
            )[0]
            turbo_sigmas = self.ExtendIntermediateSigmas.extend(
                turbo_sigmas,
                H3_TURBO_EXTEND_STEPS,
                H3_TURBO_EXTEND_START_SIGMA,
                H3_TURBO_EXTEND_END_SIGMA,
                "linear",
            )[0]
            # 扩展后的实际采样步数与进度条 total 保持一致
            sampling_total = int(turbo_sigmas.shape[-1]) - 1
        else:
            sampling_model = self.MiniMaxH3SigmaShift.execute(
                self.model,
                float(command.get("shift", 12.0)),
                float(command.get("audio_shift", 3.0)),
            )[0]
            sampling_total = stage_total
        load_seconds = time.perf_counter() - load_started
        self.torch.cuda.reset_peak_memory_stats()
        emit(
            {
                "type": "stage_progress",
                "job_id": job_id,
                "stage": "sampling",
                "total": sampling_total,
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

        if is_turbo:
            samples = self._sample_h3_turbo(
                command,
                latent,
                positive,
                sampling_callback,
                model=sampling_model,
                sampler=turbo_sampler,
                sigmas=turbo_sigmas,
            )
        else:
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

    def _encode_h3_text_conditioning(self, prompt: str):
        """缓存纯文生视频的 H3 文本 conditioning；图像条件路径不调用此缓存。"""

        cache = getattr(self, "_conditioning_cache", None)
        if cache is None:
            cache = self._conditioning_cache = {}
        clip_identity = (str(self.clip_path) if self.clip_path else None, self.clip_type)
        key = (clip_identity, "h3-t2va", prompt)
        cached = cache.get(key)
        if cached is not None:
            return cached
        # 与 ComfyUI 原生 MiniMaxH3ImageToVideo 在无首帧时保持相同的空图像参数。
        tokens = self.clip.tokenize(prompt, images=[])
        conditioning = self.clip.encode_from_tokens_scheduled(tokens)
        cache[key] = conditioning
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        return conditioning

    def _image_caches(self) -> tuple[_LruTensorCache, _LruTensorCache]:
        """返回 (图片解码缓存, 图片条件缓存)；首次访问时惰性创建。"""

        image_cache = getattr(self, "_image_cache", None)
        if image_cache is None:
            image_cache = self._image_cache = _LruTensorCache(
                max_items=4, max_bytes=256 * 1024 * 1024
            )
        conditioning_cache = getattr(self, "_image_conditioning_cache", None)
        if conditioning_cache is None:
            conditioning_cache = self._image_conditioning_cache = _LruTensorCache(
                max_items=2, max_bytes=512 * 1024 * 1024
            )
        return image_cache, conditioning_cache

    def _clear_conditioning_caches(self) -> None:
        """模型/VAE/text encoder 切换或资源释放时清空全部 conditioning 与图片缓存。"""

        self._conditioning_cache.clear()
        image_cache, conditioning_cache = self._image_caches()
        image_cache.clear()
        conditioning_cache.clear()

    def _get_h3_fl2va_conditioning(
        self,
        command: dict,
        first_frame,
        first_frame_fingerprint,
        last_frame=None,
        last_frame_fingerprint=None,
    ):
        """H3 FL2VA 首尾帧条件：命中缓存或调用原生 MiniMaxH3ImageToVideo；只缓存 positive。"""

        key = (
            "h3-fl2va",
            str(self.model_path),
            str(self.vae_path),
            str(self.audio_vae_path),
            str(self.clip_path),
            self.clip_type,
            command["prompt"],
            int(command["width"]),
            int(command["height"]),
            int(command["length"]),
            first_frame_fingerprint,
            last_frame_fingerprint,
        )
        _, conditioning_cache = self._image_caches()
        cached = self._lookup_image_conditioning(
            conditioning_cache, key, kind="h3-fl2va"
        )
        if cached is not None:
            return cached
        # 原生节点返回的 latent 与 EmptyMiniMaxH3LatentAV 一致（均来自内部
        # _empty_av_latent），因此这里丢弃 latent，由主流程每个任务重新创建。
        positive, _latent = self.MiniMaxH3ImageToVideo.execute(
            self.clip,
            self.vae,
            command["prompt"],
            int(command["width"]),
            int(command["height"]),
            int(command["length"]),
            first_frame=first_frame,
            last_frame=last_frame,
        )
        self._cache_image_conditioning(conditioning_cache, key, positive, kind="h3-fl2va")
        return positive

    def _get_h3_ref2va_conditioning(self, command: dict):
        """H3 Ref2VA 多参考输入条件：命中缓存或调用原生 MiniMaxH3ReferenceToVideo；只缓存 positive。"""

        # 当前未向客户端暴露，始终取原生默认值 "match"；未来暴露该参数时无需改键结构。
        ref_image_size = command.get("ref_image_size", "match")
        image_paths = list(command.get("reference_image_paths") or ())
        video_paths = list(command.get("reference_video_paths") or ())
        audio_paths = list(command.get("reference_audio_paths") or ())
        key = (
            "h3-ref2va",
            str(self.model_path),
            str(self.vae_path),
            str(self.audio_vae_path),
            str(self.clip_path),
            self.clip_type,
            command["prompt"],
            int(command["width"]),
            int(command["height"]),
            int(command["length"]),
            ref_image_size,
            # 引用顺序与 prompt 中 <Picture N> 编号绑定，指纹必须按顺序进入键。
            tuple(self._media_fingerprint(path) for path in image_paths),
            tuple(self._media_fingerprint(path) for path in video_paths),
            tuple(self._media_fingerprint(path) for path in audio_paths),
        )
        _, conditioning_cache = self._image_caches()
        cached = self._lookup_image_conditioning(
            conditioning_cache, key, kind="h3-ref2va"
        )
        if cached is not None:
            return cached
        # 缓存未命中才真正解码参考输入，避免命中时白付视频/音频解码开销。
        ref_images = {}
        for index, path in enumerate(image_paths, start=1):
            image, _fingerprint = self._load_input_image_cached(path)
            ref_images[f"ref_image_{index}"] = image
        ref_videos = {}
        ref_video_audios = {}
        for index, path in enumerate(video_paths, start=1):
            frames, soundtrack = self._load_ref_video(path)
            ref_videos[f"ref_video_{index}"] = frames
            if soundtrack is not None:
                # 与原生节点约定一致：ref_video_audio_N 是 ref_video_N 的配对音轨
                ref_video_audios[f"ref_video_audio_{index}"] = soundtrack
        ref_audios = {}
        for index, path in enumerate(audio_paths, start=1):
            ref_audios[f"ref_audio_{index}"] = self._load_ref_audio(path)
        # latent 处理同 FL2VA：丢弃节点返回的 latent，由主流程每个任务重新创建。
        positive, _latent = self.MiniMaxH3ReferenceToVideo.execute(
            self.clip,
            self.vae,
            self.audio_vae,
            command["prompt"],
            int(command["width"]),
            int(command["height"]),
            int(command["length"]),
            ref_image_size=ref_image_size,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )
        self._cache_image_conditioning(conditioning_cache, key, positive, kind="h3-ref2va")
        return positive

    @staticmethod
    def _media_fingerprint(path: str):
        """参考输入的廉价指纹（路径+大小+mtime），不做全文 hash。"""
        resolved = Path(path).resolve()
        stat = resolved.stat()
        return (str(resolved), stat.st_size, stat.st_mtime_ns)

    # 原生节点要求参考视频为 24fps、2-15 秒
    _H3_REF_VIDEO_FPS = 24
    _H3_REF_VIDEO_MAX_FRAMES = 15 * 24

    def _load_ref_video(self, path: str):
        """加载参考视频：帧抽成 24fps、最长 15 秒；自带音轨一并返回（无音轨为 None）。"""
        from comfy_api.latest import InputImpl

        components = InputImpl.VideoFromFile(str(path)).get_components()
        frames = components.images
        frame_rate = float(components.frame_rate or 0)
        if frame_rate > 0 and abs(frame_rate - self._H3_REF_VIDEO_FPS) > 1e-6:
            step = frame_rate / self._H3_REF_VIDEO_FPS
            total = frames.shape[0]
            indices = [min(round(i * step), total - 1) for i in range(int(total / step))]
            frames = frames[indices]
        return frames[: self._H3_REF_VIDEO_MAX_FRAMES], components.audio

    def _load_ref_audio(self, path: str):
        """加载参考音频为 ComfyUI AUDIO 格式（与 nodes_audio.LoadAudio 一致）。"""
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

    def _lookup_image_conditioning(self, cache: _LruTensorCache, key, *, kind: str):
        """读取图片条件缓存并返回深拷贝；异常时按未命中处理，不影响生成。"""

        try:
            cached = cache.get(key)
            if cached is None:
                return None
            positive, size_bytes = cached
            # 缓存对象永不直接交给采样器，避免下游原地修改污染后续任务。
            positive = _clone_conditioning_structure(positive)
        except Exception:
            logging.warning(
                "图片条件缓存读取失败，按未命中处理: kind=%s", kind, exc_info=True
            )
            return None
        logging.info("图片条件缓存命中: kind=%s bytes=%d", kind, size_bytes)
        return positive

    def _cache_image_conditioning(
        self, cache: _LruTensorCache, key, positive, *, kind: str
    ) -> None:
        """写入图片条件缓存（保存深拷贝）；异常或超预算仅记录日志，任务继续无缓存运行。"""

        try:
            stored = _clone_conditioning_structure(positive)
            size_bytes = _estimate_conditioning_bytes(stored)
            if not cache.put(key, (stored, size_bytes), size_bytes):
                logging.info(
                    "图片条件超出缓存预算，跳过缓存: kind=%s bytes=%d", kind, size_bytes
                )
                return
        except Exception:
            logging.warning(
                "图片条件缓存写入失败，本次结果不进入缓存: kind=%s", kind, exc_info=True
            )
            return
        logging.info("图片条件缓存写入: kind=%s bytes=%d", kind, size_bytes)

    @staticmethod
    def _video_generation_type(command: dict) -> str:
        if (
            command.get("reference_image_paths")
            or command.get("reference_video_paths")
            or command.get("reference_audio_paths")
        ):
            return "r2v"
        if command.get("input_image_path") or command.get("last_frame_image_path"):
            return "i2v"
        return "t2v"

    def _video_metadata(self, command: dict, resource_metadata: dict, performance: dict) -> dict:
        parameters = {
            "mode": command["video_model"],
            "generation_type": self._video_generation_type(command),
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
            "last_frame_image": command.get("last_frame_image_path"),
            "reference_images": list(command.get("reference_image_paths") or ()),
            "reference_videos": list(command.get("reference_video_paths") or ()),
            "reference_audios": list(command.get("reference_audio_paths") or ()),
        }
        if command.get("turbo_lora_path"):
            # turbo 配方不走 sigma shift，记录实际生效的调度参数
            parameters["turbo_lora_strength"] = H3_TURBO_LORA_STRENGTH
            parameters["beta_alpha"] = H3_TURBO_BETA_ALPHA
            parameters["beta_beta"] = H3_TURBO_BETA_BETA
        return {
            "schema_version": 1,
            "generator": {
                "name": "diffusion-workbench",
                "version": command.get("workbench_version"),
            },
            "parameters": parameters,
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

    def _sample_h3_turbo(
        self,
        command,
        latent,
        positive,
        callback,
        *,
        model,
        sampler,
        sigmas,
    ):
        """H3 Turbo 配方采样：RandomNoise + BasicGuider(CFG=1) + euler + 外部 sigma 序列。

        等价于社区工作流的 SamplerCustomAdvanced 组合，这里直接驱动 guider.sample
        以便接入我们自己的步进回调；BasicGuider 即 CFG=1，无负面条件。
        """
        seed = int(command["seed"])
        guider = self.BasicGuider.get_guider(model, positive)[0]
        noise = self.RandomNoise.get_noise(seed)[0]
        latent = latent.copy()
        latent_image = self.comfy_sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        samples = guider.sample(
            noise.generate_noise(latent),
            latent_image,
            sampler,
            sigmas,
            denoise_mask=latent.get("noise_mask"),
            callback=callback,
            disable_pbar=True,
            seed=seed,
        )
        samples = samples.to(self.model_management.intermediate_device())
        output = latent.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = samples
        return output

    def _downscale_image_to_target(self, image, width: int, height: int):
        """把 (B,H,W,C) 图像缩放到目标像素网格；仅在源图超出目标尺寸时缩放。

        仅用于 edit-krea2 中结果会被丢弃的那次 VAEEncode（pixel path 会从原图按
        目标网格重新编码），因此直接缩放到精确的 (height, width)，不保持宽高比。
        """
        _b, h, w, _c = image.shape
        if h <= height and w <= width:
            return image
        resized = self.torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2).float(),
            size=(height, width),
            mode="bicubic",
            antialias=True,
        )
        return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    def _load_input_image(self, path: str | None):
        # 保持原有外部行为：空路径返回 None，文件不存在抛中文 FileNotFoundError。
        tensor, _fingerprint = self._load_input_image_cached(path)
        return tensor

    def _load_input_image_cached(self, path: str | None):
        """读取输入图片并返回 (归一化 RGB tensor, 内容指纹)；重复文件命中解码缓存。"""

        if not path:
            return None, None
        image_path = Path(path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"找不到输入图片: {image_path}")
        stat = image_path.stat()
        key = (str(image_path), stat.st_size, stat.st_mtime_ns)
        image_cache, _ = self._image_caches()
        cached = image_cache.get(key)
        if cached is not None:
            tensor, fingerprint = cached
            # 缓存副本只读，返回 clone 避免下游节点原地修改污染缓存。
            return tensor.clone(), fingerprint
        tensor = self._load_input_image_uncached(image_path)
        fingerprint = _image_tensor_fingerprint(tensor)
        size_bytes = int(tensor.nelement()) * int(tensor.element_size())
        image_cache.put(key, (tensor, fingerprint), size_bytes)
        return tensor.clone(), fingerprint

    def _load_input_image_uncached(self, image_path: Path):
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
        self._clear_conditioning_caches()
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
        self._clear_conditioning_caches()
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

    def _ensure_krea2edit_nodes(self):
        """加载 comfyui-krea2edit 自定义节点包并返回模块；缺文件/缺节点抛中文 RuntimeError。"""
        if self.krea2edit_module is not None:
            return self.krea2edit_module
        node_root = self.comfy_root / "custom_nodes" / "comfyui-krea2edit"
        entrypoint = node_root / "__init__.py"
        if not entrypoint.is_file():
            raise RuntimeError(
                f"krea2 编辑模式需要安装 comfyui-krea2edit 自定义节点: {node_root}"
            )
        module_name = "diffusion_workbench_comfyui_krea2edit"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(
                module_name,
                entrypoint,
                submodule_search_locations=[str(node_root)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"无法加载 comfyui-krea2edit 节点: {entrypoint}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        try:
            mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
            mappings["Krea2EditModelPatch"]
            mappings["Krea2EditGroundedEncode"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                "comfyui-krea2edit 未提供 Krea2EditModelPatch/Krea2EditGroundedEncode 节点"
            ) from exc
        self.krea2edit_module = module
        return module

    def _ensure_rebalance_nodes(self):
        """加载 ComfyUI-Conditioning-Rebalance 的 krea2 子模块并返回；缺文件/缺节点抛中文 RuntimeError。

        只按需加载 conditioning_rebalance 与 krea2 两个子模块，不执行包 __init__
        （避免引入 OmniNode 等无关子模块的额外依赖）。
        """
        if self.rebalance_module is not None:
            return self.rebalance_module
        node_root = self.comfy_root / "custom_nodes" / "ComfyUI-Conditioning-Rebalance"
        if not (node_root / "krea2.py").is_file():
            raise RuntimeError(
                "krea2-rebalance 模式需要安装 ComfyUI-Conditioning-Rebalance 自定义节点: "
                f"{node_root}"
                "（https://github.com/nova452/ComfyUI-Conditioning-Rebalance）"
            )
        package_name = "diffusion_workbench_conditioning_rebalance"
        krea2_module = sys.modules.get(f"{package_name}.krea2")
        if krea2_module is None:
            package_spec = importlib.util.spec_from_file_location(
                package_name,
                node_root / "__init__.py",
                submodule_search_locations=[str(node_root)],
            )
            if package_spec is None:
                raise RuntimeError(
                    f"无法加载 ComfyUI-Conditioning-Rebalance 节点包: {node_root}"
                )
            package = importlib.util.module_from_spec(package_spec)
            sys.modules[package_name] = package
            try:
                for submodule in ("conditioning_rebalance", "krea2"):
                    spec = importlib.util.spec_from_file_location(
                        f"{package_name}.{submodule}",
                        node_root / f"{submodule}.py",
                    )
                    if spec is None or spec.loader is None:
                        raise RuntimeError(
                            f"无法加载 ComfyUI-Conditioning-Rebalance 子模块: {submodule}"
                        )
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"{package_name}.{submodule}"] = module
                    setattr(package, submodule, module)
                    spec.loader.exec_module(module)
            except Exception:
                for submodule in ("conditioning_rebalance", "krea2"):
                    sys.modules.pop(f"{package_name}.{submodule}", None)
                sys.modules.pop(package_name, None)
                raise
            krea2_module = package.krea2
        try:
            krea2_module.NODE_CLASS_MAPPINGS["Krea2EncodeRebalance"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                "ComfyUI-Conditioning-Rebalance 未提供 Krea2EncodeRebalance 节点"
            ) from exc
        self.rebalance_module = krea2_module
        return krea2_module

    def _ensure_checkpoint(self, path: Path) -> bool:
        if (
            self.checkpoint_path == path
            and self.model is not None
            and self.clip is not None
            and self.vae is not None
        ):
            return False
        self._unload_gpu()
        self._clear_conditioning_caches()
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
        self._clear_conditioning_caches()
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
        self._clear_conditioning_caches()
        self.vae = None
        self.upscale_model = None
        self.vae_path = None
        gc.collect()
        name = self._register_exact("vae", path)
        self.vae = self.nodes.VAELoader().load_vae(name)[0]
        self.vae_path = path
        return True

    def _encode_text_conditioning(
        self,
        prompt: str,
        negative_prompt: str,
        *,
        cfg: float,
        cache_scope: str,
        reuse_negative_at_cfg_one: bool,
    ) -> tuple[object, object]:
        """编码并缓存纯文本 conditioning；带图像条件的节点不得调用此方法。"""

        clip_identity = (str(self.clip_path) if self.clip_path else None, self.clip_type)
        key = (
            clip_identity,
            cache_scope,
            prompt,
            negative_prompt,
            bool(reuse_negative_at_cfg_one and cfg == 1.0),
        )
        cache = getattr(self, "_conditioning_cache", None)
        if cache is None:
            cache = self._conditioning_cache = {}
        cached = cache.get(key)
        if cached is not None:
            return cached

        encoder = self.nodes.CLIPTextEncode()
        positive = encoder.encode(self.clip, prompt)[0]
        if reuse_negative_at_cfg_one and cfg == 1.0:
            negative = positive
        else:
            negative = encoder.encode(self.clip, negative_prompt)[0]
        cache[key] = (positive, negative)
        # 文本 embedding 可能较大，限制缓存规模，保留最近插入的条目。
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        return positive, negative

    def _ensure_audio_vae(self, path: Path) -> bool:
        if self.audio_vae is not None and self.audio_vae_path == path:
            return False
        self._unload_gpu()
        self._clear_conditioning_caches()
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
        self._clear_conditioning_caches()
        self.mode = None
        # 先清掉 ComfyUI 全局收尾状态（cross-step/prefetch/cast buffer），否则它们
        # 挂住的引用会让下面的 gc 无法回收刚丢弃的模型。
        self._comfy_execution_cleanup()
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
        if command.get("mode") == "edit-krea2":
            if not command.get("input_image_path"):
                raise ValueError("edit-krea2 模式必须提供 input_image_path")
            if not command.get("edit_lora_path"):
                raise ValueError("edit-krea2 模式必须提供 edit_lora_path")
            grounding_px = command.get("grounding_px")
            if (
                not isinstance(grounding_px, int)
                or isinstance(grounding_px, bool)
                or not 0 <= grounding_px <= 4096
            ):
                raise ValueError("edit-krea2 grounding_px 必须是 0 到 4096 的整数")
            ref_boost = command.get("ref_boost")
            if not isinstance(ref_boost, (int, float)) or isinstance(ref_boost, bool):
                raise ValueError("edit-krea2 ref_boost 必须是数值")
            import math as _math
            if not _math.isfinite(float(ref_boost)) or not 0 <= float(ref_boost) <= 1000:
                raise ValueError("edit-krea2 ref_boost 必须是 0 到 1000 的有限数值")
            if upscale.enabled and upscale.method == UpscaleMethod.LATENT_HIRES:
                raise ValueError("edit-krea2 模式仅支持 resize / upscale_model 放大")
        elif command.get("input_image_path") is not None:
            raise ValueError("仅 edit-krea2 模式支持输入图片编辑")
        elif command.get("secondary_input_image_path") is not None:
            raise ValueError("仅 edit-krea2 模式支持第二输入图片")
        if command.get("mode") == "krea2-rebalance":
            reference_paths = command.get("reference_image_paths") or []
            if not 1 <= len(reference_paths) <= REBALANCE_MAX_REFERENCE_IMAGES:
                raise ValueError(
                    "krea2-rebalance 模式必须提供 1 到 "
                    f"{REBALANCE_MAX_REFERENCE_IMAGES} 张参考图"
                )
            tokens = command.get("reference_image_tokens") or []
            if tokens and len(tokens) != len(reference_paths):
                raise ValueError(
                    "krea2-rebalance reference_image_tokens 数量必须与参考图数量一致"
                )
            for tier in tokens:
                if tier not in REBALANCE_TOKEN_TIERS:
                    raise ValueError(
                        "krea2-rebalance 参考图 token 档位必须是 "
                        f"{'/'.join(REBALANCE_TOKEN_TIERS)}: {tier}"
                    )
        elif command.get("reference_image_paths"):
            raise ValueError("仅 krea2-rebalance 模式支持参考图")

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
            "minimax" if video_model in MINIMAX_H3_MODELS else "wan"
        )
        if command.get("clip_type") != expected_clip_type:
            raise ValueError(
                f"{video_model.value} clip_type 必须为 {expected_clip_type}"
            )
        width, height = int(command["width"]), int(command["height"])
        if width < 16 or height < 16:
            raise ValueError("视频宽高必须至少为 16 像素")
        duration = int(command["duration_seconds"])
        fps = int(command["fps"])
        length = int(command["length"])
        if duration <= 0 or fps <= 0 or fps > 120:
            raise ValueError("视频时长必须为正数，帧率必须在 1 到 120 之间")
        if video_model in MINIMAX_H3_MODELS:
            if width < H3_MIN_SIZE or height < H3_MIN_SIZE or width % 32 or height % 32:
                raise ValueError("MiniMax H3 视频宽高必须是 32 的倍数")
            if width > H3_MAX_SIZE or height > H3_MAX_SIZE or width * height > H3_MAX_PIXELS:
                raise ValueError("MiniMax H3 画面不能超过 1344×768 的官方 Base 范围")
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
            reference_images = list(command.get("reference_image_paths") or ())
            reference_videos = list(command.get("reference_video_paths") or ())
            reference_audios = list(command.get("reference_audio_paths") or ())
            has_reference = bool(reference_images or reference_videos or reference_audios)
            model_path = command.get("model_path", "")
            model_is_ref2va = is_h3_ref2va_model_name(Path(model_path).name)
            if video_model in (
                VideoModel.MINIMAX_H3_FL2VA,
                VideoModel.MINIMAX_H3_TURBO,
            ):
                if model_is_ref2va:
                    raise ValueError(f"{video_model.value} 必须选择 fl2va 模型")
                if has_reference:
                    raise ValueError(f"{video_model.value} 不接受参考输入")
                if video_model == VideoModel.MINIMAX_H3_TURBO and not command.get(
                    "turbo_lora_path"
                ):
                    raise ValueError("MiniMax H3 Turbo 必须提供 turbo LoRA")
            else:
                if not model_is_ref2va:
                    raise ValueError("MiniMax H3 Ref2VA 必须选择 ref2va 模型")
                if command.get("input_image_path") or command.get("last_frame_image_path"):
                    raise ValueError("MiniMax H3 Ref2VA 不接受首帧/尾帧输入")
                if not has_reference:
                    raise ValueError("MiniMax H3 Ref2VA 至少需要一个参考输入")
                if len(reference_images) > H3_REF2VA_MAX_IMAGES:
                    raise ValueError("MiniMax H3 Ref2VA 参考图不能超过 9 张")
                if len(reference_videos) > H3_REF2VA_MAX_VIDEOS:
                    raise ValueError("MiniMax H3 Ref2VA 参考视频不能超过 3 段")
                if len(reference_audios) > H3_REF2VA_MAX_AUDIOS:
                    raise ValueError("MiniMax H3 Ref2VA 参考音频不能超过 3 段")
                if (
                    len(reference_images) + len(reference_videos) + len(reference_audios)
                    > H3_REF2VA_MAX_TOTAL
                ):
                    raise ValueError("MiniMax H3 Ref2VA 参考输入总数不能超过 12 个")
        else:
            if width % 16 or height % 16:
                raise ValueError("视频宽高必须是 16 的倍数")
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
        if command_type == "describe_image":
            # 图片反推：同步命令,失败时同样释放资源,避免残留半初始化的 clip。
            with active_lock:
                active_job_id = command.get("job_id")
            try:
                payload = worker.describe_image(command)
            except Exception as exc:
                traceback_text = traceback.format_exc()
                try:
                    worker.release()
                except Exception:
                    logging.exception("反推失败后的资源清理也失败")
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
