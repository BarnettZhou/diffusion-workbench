from __future__ import annotations

from diffusion_workbench_core import (
    GenerationSettings,
    Mode,
    ModelLoader,
    ResourceKind,
    WorkbenchCore,
)
from diffusion_workbench_core.domain import UpscaleMethod, UpscaleSettings

from .schemas import CreateJobsRequest, UpscaleRequest


def resource_at(core: WorkbenchCore, mode: Mode, kind: ResourceKind, index: int):
    """只允许从 Core 当前资源列表中按 index 选择，拒绝任何客户端路径。"""
    for item in core.list_resources(mode, kind):
        if item.index == index:
            return item
    raise LookupError(f"resource index {index} not found")


def upscale_model_at(core: WorkbenchCore, index: int):
    """放大模型同样只能按 index 从服务端目录选择，不接受任何客户端路径。"""
    for item in core.list_upscale_models():
        if item.index == index:
            return item
    raise LookupError(f"upscale model index {index} not found")


def build_upscale_settings(
    core: WorkbenchCore, payload: UpscaleRequest | None
) -> UpscaleSettings:
    """把请求里的放大参数映射为 Core 的 UpscaleSettings 并交给 domain 校验。"""
    if payload is None or not payload.enabled:
        return UpscaleSettings()
    model = None
    if payload.method == UpscaleMethod.UPSCALE_MODEL.value:
        if payload.model_index is None:
            raise ValueError("upscale_model 方法必须选择放大模型")
        model = upscale_model_at(core, payload.model_index)
    settings = UpscaleSettings(
        enabled=True,
        method=UpscaleMethod(payload.method),
        scale=payload.scale,
        interpolation=payload.interpolation,
        model=model,
        tile=payload.tile,
        overlap=payload.overlap,
        steps=payload.steps,
        start_step=payload.start_step,
        cfg=payload.cfg,
        sampler=payload.sampler,
        scheduler=payload.scheduler,
        seed=payload.seed,
    )
    settings.validate()
    return settings


def submit_jobs(core: WorkbenchCore, payload: CreateJobsRequest):
    mode = Mode(payload.mode)
    model = resource_at(core, mode, ResourceKind.DIFFUSION, payload.model_index)
    upscale = build_upscale_settings(core, payload.upscale)
    fixed = core.config.resources[mode]
    if fixed.model_loader == ModelLoader.COMPONENTS:
        if payload.vae_index is None:
            raise ValueError(f"{mode.value} 必须选择 VAE")
        vae = resource_at(core, mode, ResourceKind.VAE, payload.vae_index)
    else:
        if payload.vae_index is not None:
            raise ValueError(f"{mode.value} checkpoint 已内嵌 VAE")
        vae = None
    settings = GenerationSettings(
        mode=mode,
        model=model,
        vae=vae,
        text_encoder=fixed.text_encoder,
        clip_type=fixed.clip_type,
        model_loader=fixed.model_loader,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        seed=payload.seed,
        cfg=payload.cfg,
        sampler=payload.sampler,
        scheduler=payload.scheduler,
        upscale=upscale,
    )
    return core.submit(settings, payload.count)
