import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


SAMPLERS = (
    "euler",
    "dpmpp_2m_sde",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
)

VIDEO_MIN_SIZE = 16
VIDEO_MAX_SIZE = 4096
H3_MIN_SIZE = 32
H3_MAX_SIZE = 1344
H3_MAX_PIXELS = 768 * 1344
SCHEDULERS = (
    "simple",
    "sgm_uniform",
    "beta",
    "karras",
    "exponential",
    "ddim_uniform",
    "normal",
    "linear_quadratic",
    "kl_optimal",
)
MIN_STEPS = 1
MAX_STEPS = 100
IMAGE_UPSCALE_INTERPOLATIONS = (
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos",
)
LATENT_UPSCALE_INTERPOLATIONS = (
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "bislerp",
)


def validate_steps(steps: int) -> None:
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise ValueError(f"steps 必须在 {MIN_STEPS} 到 {MAX_STEPS} 之间")


def validate_cfg(cfg: float) -> None:
    if not math.isfinite(cfg) or cfg <= 0:
        raise ValueError("CFG 必须是大于 0 的有限数值")


def validate_sampling(steps: int, cfg: float, sampler: str, scheduler: str) -> None:
    validate_steps(steps)
    validate_cfg(cfg)
    if sampler not in SAMPLERS:
        raise ValueError(f"不支持 sampler: {sampler}")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"不支持 scheduler: {scheduler}")


class Mode(StrEnum):
    ZIT = "zit"
    KREA2 = "krea2"
    ZIB = "zib"
    SDXL = "sdxl"


class VideoModel(StrEnum):
    WAN22_TI2V_5B = "wan2.2-ti2v-5b"
    WAN22_I2V_14B = "wan2.2-i2v-14b"
    MINIMAX_H3 = "minimax-h3"


class ModelLoader(StrEnum):
    COMPONENTS = "components"
    CHECKPOINT = "checkpoint"


class ResourceKind(StrEnum):
    DIFFUSION = "diffusion"
    VAE = "vae"


class UpscaleMethod(StrEnum):
    RESIZE = "resize"
    UPSCALE_MODEL = "upscale_model"
    LATENT_HIRES = "latent_hires"


@dataclass(frozen=True)
class ResourceItem:
    index: int
    path: Path
    alias: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.alias} ({self.path.name})" if self.alias else self.path.name


@dataclass(frozen=True)
class UpscaleSettings:
    enabled: bool = False
    method: UpscaleMethod = UpscaleMethod.LATENT_HIRES
    scale: float = 2.0
    interpolation: str = "bislerp"
    model: ResourceItem | None = None
    tile: int = 512
    overlap: int = 32
    steps: int = 9
    start_step: int = 4
    cfg: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    seed: int | None = None

    @property
    def executed_steps(self) -> int:
        return self.steps - self.start_step

    def validate(self) -> None:
        if not math.isfinite(self.scale) or not 1 < self.scale <= 4:
            raise ValueError("放大倍数必须大于 1 且不超过 4")
        if self.seed is not None and self.seed < 0:
            raise ValueError("放大 seed 必须为 inherit 或非负整数")
        if self.method == UpscaleMethod.LATENT_HIRES:
            validate_steps(self.steps)
            if not 0 <= self.start_step < self.steps:
                raise ValueError("start-step 必须大于等于 0 且小于 steps")
            if self.cfg is not None:
                validate_cfg(self.cfg)
            if self.sampler is not None and self.sampler not in SAMPLERS:
                raise ValueError(f"不支持 sampler: {self.sampler}")
            if self.scheduler is not None and self.scheduler not in SCHEDULERS:
                raise ValueError(f"不支持 scheduler: {self.scheduler}")
            if self.interpolation not in LATENT_UPSCALE_INTERPOLATIONS:
                raise ValueError(
                    f"latent_hires 不支持 interpolation: {self.interpolation}"
                )
        else:
            if self.interpolation not in IMAGE_UPSCALE_INTERPOLATIONS:
                raise ValueError(
                    f"{self.method.value} 不支持 interpolation: {self.interpolation}"
                )
        if self.method == UpscaleMethod.UPSCALE_MODEL:
            if self.tile < 128 or self.tile > 1024 or self.tile % 32:
                raise ValueError("tile 必须在 128 到 1024 之间且是 32 的倍数")
            if self.overlap < 0 or self.overlap >= self.tile / 2:
                raise ValueError("overlap 必须大于等于 0 且小于 tile 的一半")
        if (
            self.enabled
            and self.method == UpscaleMethod.UPSCALE_MODEL
            and self.model is None
        ):
            raise ValueError("upscale_model 方法必须选择放大模型")

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "method": self.method.value,
            "scale": self.scale,
            "interpolation": self.interpolation,
            "model_path": str(self.model.path.resolve()) if self.model else None,
            "tile": self.tile,
            "overlap": self.overlap,
            "steps": self.steps,
            "start_step": self.start_step,
            "cfg": self.cfg,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: dict | None) -> "UpscaleSettings":
        if not value:
            return cls()
        model_path = value.get("model_path")
        return cls(
            enabled=bool(value.get("enabled", False)),
            method=UpscaleMethod(value.get("method", UpscaleMethod.LATENT_HIRES)),
            scale=float(value.get("scale", 2.0)),
            interpolation=str(value.get("interpolation", "bislerp")),
            model=(
                ResourceItem(index=0, path=Path(model_path))
                if model_path
                else None
            ),
            tile=int(value.get("tile", 512)),
            overlap=int(value.get("overlap", 32)),
            steps=int(value.get("steps", 9)),
            start_step=int(value.get("start_step", 4)),
            cfg=float(value["cfg"]) if value.get("cfg") is not None else None,
            sampler=value.get("sampler"),
            scheduler=value.get("scheduler"),
            seed=int(value["seed"]) if value.get("seed") is not None else None,
        )


@dataclass(frozen=True)
class GenerationSettings:
    mode: Mode
    model: ResourceItem
    vae: ResourceItem | None
    text_encoder: Path | None
    clip_type: str | None
    prompt: str
    negative_prompt: str = ""
    width: int = 576
    height: int = 576
    steps: int = 8
    seed: int = -1
    sampler: str = "euler"
    scheduler: str = "simple"
    cfg: float = 1.0
    upscale: UpscaleSettings = field(default_factory=UpscaleSettings)
    model_loader: ModelLoader = ModelLoader.COMPONENTS

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空")
        if self.model_loader == ModelLoader.COMPONENTS:
            if self.vae is None or self.text_encoder is None or not self.clip_type:
                raise ValueError("components loader 必须提供 VAE、文本编码器和 clip type")
        elif self.vae is not None or self.text_encoder is not None or self.clip_type is not None:
            raise ValueError("checkpoint loader 不接受外置 VAE、文本编码器或 clip type")
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("size 必须为正数且是 16 的倍数")
        if self.seed < -1:
            raise ValueError("seed 必须为 -1 或非负整数")
        validate_sampling(self.steps, self.cfg, self.sampler, self.scheduler)
        self.upscale.validate()


@dataclass(frozen=True)
class JobRecord:
    id: str
    batch_id: str | None
    status: str
    submitted_at: datetime
    output_path: Path
    mode: Mode
    prompt: str
    model_path: Path
    vae_path: Path | None
    text_encoder_path: Path | None
    sampler: str
    scheduler: str
    width: int
    height: int
    steps: int
    seed: int
    cfg: float
    negative_prompt: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    error: str | None = None
    upscaled_output_path: Path | None = None
    upscale: UpscaleSettings = field(default_factory=UpscaleSettings)
    model_loader: ModelLoader = ModelLoader.COMPONENTS


@dataclass(frozen=True)
class VideoGenerationSettings:
    video_model: VideoModel
    model: ResourceItem
    vae: ResourceItem
    text_encoder: Path
    prompt: str
    audio_vae: Path | None = None
    negative_prompt: str = ""
    input_image: Path | None = None
    reference_image: Path | None = None
    width: int = 704
    height: int = 960
    duration_seconds: int = 5
    fps: int = 24
    steps: int = 20
    seed: int = -1
    sampler: str = "uni_pc"
    scheduler: str = "simple"
    cfg: float = 5.0
    denoise: float = 1.0
    shift: float = 8.0
    latent_multiplier: float = 1.0

    @property
    def length(self) -> int:
        if self.video_model == VideoModel.MINIMAX_H3:
            if self.width % 32 or self.height % 32:
                raise ValueError("MiniMax H3 视频宽高必须是 32 的倍数")
            length = max(5, round(self.duration_seconds * 24))
            while length % 17 != 5:
                length += 1
            return length
        return self.duration_seconds * self.fps + 1

    @property
    def generation_type(self) -> str:
        if self.reference_image is not None:
            return "r2v"
        return "i2v" if self.input_image is not None else "t2v"

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空")
        if not isinstance(self.duration_seconds, int) or isinstance(
            self.duration_seconds, bool
        ):
            raise ValueError("视频时长必须是整数秒")
        if not isinstance(self.fps, int) or isinstance(self.fps, bool):
            raise ValueError("帧率必须是整数")
        if self.width < VIDEO_MIN_SIZE or self.height < VIDEO_MIN_SIZE:
            raise ValueError(f"视频宽高必须至少为 {VIDEO_MIN_SIZE} 像素")
        if self.duration_seconds <= 0:
            raise ValueError("视频时长必须是正整数")
        if not 1 <= self.fps <= 120:
            raise ValueError("帧率必须在 1 到 120 之间")
        if self.video_model == VideoModel.MINIMAX_H3:
            if self.width > H3_MAX_SIZE or self.height > H3_MAX_SIZE:
                raise ValueError(f"MiniMax H3 宽高不能超过 {H3_MAX_SIZE} 像素")
            if self.width * self.height > H3_MAX_PIXELS:
                raise ValueError("MiniMax H3 画面面积不能超过 768×1344")
            if self.fps != 24:
                raise ValueError("MiniMax H3 帧率固定为 24")
            if self.length % 17 != 5:
                raise ValueError("MiniMax H3 总帧数必须满足 length = 17n + 5")
            if self.cfg != 1.0:
                raise ValueError("MiniMax H3 CFG 固定为 1")
            if self.audio_vae is None:
                raise ValueError("MiniMax H3 必须配置音频 VAE")
            model_is_ref2va = "ref2va" in self.model.path.name.casefold()
            if model_is_ref2va != (self.reference_image is not None):
                raise ValueError("MiniMax H3 Ref2VA 必须使用 ref2va 模型和单张参考图；FL2VA 不接受参考图")
            if self.input_image is not None and self.reference_image is not None:
                raise ValueError("MiniMax H3 首帧输入与参考图不能同时提供")
        elif self.reference_image is not None:
            raise ValueError("参考图片只支持 MiniMax H3 Ref2VA")
        elif (self.length - 1) % 4:
            raise ValueError("视频总帧数必须满足 length = 4n + 1")
        if self.video_model == VideoModel.MINIMAX_H3:
            if self.width % 32 or self.height % 32:
                raise ValueError("MiniMax H3 视频宽高必须是 32 的倍数")
        elif self.width % 16 or self.height % 16:
            raise ValueError("视频宽高必须是 16 的倍数")
        if self.seed < -1:
            raise ValueError("seed 必须为 -1 或非负整数")
        if self.video_model == VideoModel.WAN22_I2V_14B and self.input_image is None:
            raise ValueError("Wan I2V-14B 必须提供输入图片")
        if self.denoise != 1.0:
            raise ValueError("视频 denoise 固定为 1")
        if not math.isfinite(self.shift) or not 0.0 <= self.shift <= 100.0:
            raise ValueError("shift 必须在 0 到 100 之间")
        if not math.isfinite(self.latent_multiplier) or self.latent_multiplier <= 0:
            raise ValueError("latent_multiplier 必须是大于 0 的有限数值")
        validate_sampling(self.steps, self.cfg, self.sampler, self.scheduler)


def resolve_video_diffusion_pair(
    video_model: VideoModel, selected: Path
) -> tuple[Path, Path | None]:
    """解析视频模型需要的 diffusion 文件;14B I2V 必须同时使用 high/low 两个模型。"""
    selected = selected.resolve()
    if video_model != VideoModel.WAN22_I2V_14B:
        return selected, None
    name = selected.name
    if "_high_noise_" in name:
        high = selected
        low = selected.with_name(name.replace("_high_noise_", "_low_noise_", 1))
    elif "_low_noise_" in name:
        low = selected
        high = selected.with_name(name.replace("_low_noise_", "_high_noise_", 1))
    else:
        raise ValueError("Wan I2V-14B diffusion 文件名必须包含 high_noise 或 low_noise")
    return high, low


@dataclass(frozen=True)
class VideoJobRecord:
    id: str
    batch_id: str | None
    status: str
    submitted_at: datetime
    output_path: Path
    video_model: VideoModel
    prompt: str
    model_path: Path
    vae_path: Path
    text_encoder_path: Path
    sampler: str
    scheduler: str
    width: int
    height: int
    duration_seconds: int
    fps: int
    length: int
    steps: int
    seed: int
    cfg: float
    audio_vae_path: Path | None = None
    denoise: float = 1.0
    shift: float = 8.0
    latent_multiplier: float = 1.0
    negative_prompt: str = ""
    input_image_path: Path | None = None
    reference_image_path: Path | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    elapsed_seconds: float | None = None
    error: str | None = None

    @property
    def generation_type(self) -> str:
        if self.reference_image_path is not None:
            return "r2v"
        return "i2v" if self.input_image_path is not None else "t2v"
