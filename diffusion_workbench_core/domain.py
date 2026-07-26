from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from datetime import datetime


class Mode(StrEnum):
    ZIT = "zit"
    KREA2 = "krea2"


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
        if not 8 <= self.steps <= 20:
            raise ValueError("steps 必须在 8 到 20 之间")
        if self.seed < -1:
            raise ValueError("seed 必须为 -1 或非负整数")
        if self.sampler != "euler" or self.scheduler != "simple" or self.cfg != 1.0:
            raise ValueError("当前只支持 Euler + simple，CFG 固定为 1")


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
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    error: str | None = None
