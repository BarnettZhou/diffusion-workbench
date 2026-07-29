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
    vae: ResourceItem
    text_encoder: Path
    clip_type: str
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

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空")
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
    vae_path: Path
    text_encoder_path: Path
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
