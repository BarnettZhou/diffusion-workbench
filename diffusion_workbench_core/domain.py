import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ResourceItem:
    index: int
    path: Path
    alias: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.alias} ({self.path.name})" if self.alias else self.path.name


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

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空")
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("size 必须为正数且是 16 的倍数")
        if self.seed < -1:
            raise ValueError("seed 必须为 -1 或非负整数")
        validate_sampling(self.steps, self.cfg, self.sampler, self.scheduler)


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
