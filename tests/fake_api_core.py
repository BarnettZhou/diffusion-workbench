from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from diffusion_workbench_core.config import ComfyConfig, ModeResources, VideoResources, WorkbenchConfig
from diffusion_workbench_core.domain import (
    JobRecord,
    Mode,
    ModelLoader,
    ResourceItem,
    ResourceKind,
    VideoJobRecord,
    VideoModel,
)


class FakeApiCore:
    """无 GPU 的 WorkbenchCore 替身，记录调用并返回内存 JobRecord。"""

    def __init__(self, root: Path):
        self.root = root
        self.config = WorkbenchConfig(
            path=root / "workbench.yaml",
            comfyui=ComfyConfig(root, root / "python.exe"),
            resources={
                Mode.ZIT: ModeResources(
                    (), (), root / "zit-te.safetensors", "stable_diffusion"
                ),
                Mode.KREA2: ModeResources((), (), root / "krea-te.safetensors", "krea2"),
                Mode.ZIB: ModeResources(
                    (), (), root / "zib-te.safetensors", "stable_diffusion"
                ),
                Mode.SDXL: ModeResources(
                    (), (), None, None, ModelLoader.CHECKPOINT
                ),
            },
            output_dir=root / "output",
            database=root / "jobs.sqlite3",
            worker_timeout_seconds=300,
            video_resources={
                VideoModel.WAN22_TI2V_5B: VideoResources(
                    (root / "wan-models",),
                    (root / "wan-vae",),
                    root / "wan-te.safetensors",
                ),
                VideoModel.WAN22_I2V_14B: VideoResources(
                    (root / "wan14-models",),
                    (root / "wan14-vae" / "wan_2.1_vae.safetensors",),
                    root / "wan14-te.safetensors",
                ),
                VideoModel.MINIMAX_H3: VideoResources(
                    (root / "h3-models",),
                    (root / "h3-vae",),
                    root / "h3-te.safetensors",
                    "minimax",
                    root / "h3-audio-vae.safetensors",
                ),
            },
        )
        self.items = {
            (Mode.ZIT, ResourceKind.DIFFUSION): [
                ResourceItem(1, root / "zit.safetensors"),
                ResourceItem(2, root / "zit2.safetensors"),
            ],
            (Mode.ZIT, ResourceKind.VAE): [ResourceItem(1, root / "zit-vae.safetensors")],
            (Mode.KREA2, ResourceKind.DIFFUSION): [
                ResourceItem(1, root / "krea.safetensors")
            ],
            (Mode.KREA2, ResourceKind.VAE): [ResourceItem(1, root / "krea-vae.safetensors")],
            (Mode.ZIB, ResourceKind.DIFFUSION): [
                ResourceItem(1, root / "zib.safetensors")
            ],
            (Mode.ZIB, ResourceKind.VAE): [ResourceItem(1, root / "zib-vae.safetensors")],
            (Mode.SDXL, ResourceKind.DIFFUSION): [
                ResourceItem(1, root / "sdxl.safetensors")
            ],
            (Mode.SDXL, ResourceKind.VAE): [],
        }
        self.submitted: list = []
        self.jobs: dict[str, JobRecord] = {}
        self.upscale_models = [
            ResourceItem(1, root / "upscale_models" / "4x-UltraSharp.pth"),
            ResourceItem(2, root / "upscale_models" / "2x-Lite.pth"),
        ]
        self.list_jobs_calls: list[dict] = []
        self.stopped = False
        self.closed = False
        self.sink = None
        self.preview_enabled = False
        self.skipped_id = None
        self.skip_error = None
        self.skip_calls = 0
        self.video_model_items = {
            VideoModel.WAN22_TI2V_5B: [
                ResourceItem(1, root / "wan-models" / "wan2.2-ti2v-5b.safetensors"),
            ],
            VideoModel.WAN22_I2V_14B: [
                ResourceItem(
                    1,
                    root
                    / "wan14-models"
                    / "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                ),
                ResourceItem(
                    2,
                    root
                    / "wan14-models"
                    / "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                ),
            ],
            VideoModel.MINIMAX_H3: [
                ResourceItem(1, root / "h3-models" / "minimax_h3_fl2va_int4.safetensors"),
                ResourceItem(2, root / "h3-models" / "minimax_h3_ref2va_int4.safetensors"),
            ],
        }
        self.video_vae_items = {
            VideoModel.WAN22_TI2V_5B: [
                ResourceItem(1, root / "wan-vae" / "wan-vae.safetensors"),
            ],
            VideoModel.WAN22_I2V_14B: [
                ResourceItem(
                    1, root / "wan14-vae" / "wan_2.1_vae.safetensors"
                ),
            ],
            VideoModel.MINIMAX_H3: [
                ResourceItem(1, root / "h3-vae" / "minimax_h3_video_vae_fp16.safetensors"),
            ],
        }
        self.video_submitted: list = []
        self.video_jobs: dict[str, VideoJobRecord] = {}
        self.loaded_resources = {}

    def list_resources(self, mode, kind):
        return self.items[(mode, kind)]

    def list_upscale_models(self):
        return self.upscale_models

    def set_alias(self, mode, kind, path, alias):
        items = self.items[(mode, kind)]
        if any(item.alias == alias and item.path != path for item in items):
            raise ValueError(f"alias {alias!r} 已被使用")
        self.items[(mode, kind)] = [
            ResourceItem(item.index, item.path, alias if item.path == path else item.alias)
            for item in items
        ]

    def make_job(self, settings, batch_id=None, status="queued") -> JobRecord:
        output_path = (
            self.config.output_dir
            / "2026-07-27"
            / f"{settings.mode.value}-{len(self.jobs) + 1:05d}.png"
        )
        return JobRecord(
            id=str(uuid.uuid4()),
            batch_id=batch_id,
            status=status,
            submitted_at=datetime.now().astimezone(),
            output_path=output_path,
            upscaled_output_path=(
                output_path.with_name(f"{output_path.stem}-upscale.png")
                if settings.upscale.enabled
                else None
            ),
            upscale=settings.upscale,
            mode=settings.mode,
            prompt=settings.prompt,
            negative_prompt=settings.negative_prompt,
            model_path=settings.model.path,
            vae_path=settings.vae.path if settings.vae else None,
            text_encoder_path=settings.text_encoder,
            sampler=settings.sampler,
            scheduler=settings.scheduler,
            width=settings.width,
            height=settings.height,
            steps=settings.steps,
            seed=settings.seed,
            cfg=settings.cfg,
            model_loader=settings.model_loader,
        )

    def submit(self, settings, count):
        settings.validate()
        self.submitted.append((settings, count))
        batch_id = str(uuid.uuid4()) if count > 1 else None
        jobs = []
        for _ in range(count):
            job = self.make_job(settings, batch_id=batch_id)
            self.jobs[job.id] = job
            jobs.append(job)
        return jobs

    def get_job(self, job_id):
        return self.jobs[job_id]

    def list_jobs(self, *, status=None, mode=None, limit=50, cursor=None):
        self.list_jobs_calls.append(
            {"status": status, "mode": mode, "limit": limit, "cursor": cursor}
        )
        if cursor == "bad-cursor":
            raise ValueError("无效的分页 cursor")
        jobs = sorted(
            self.jobs.values(), key=lambda job: (job.submitted_at, job.id), reverse=True
        )
        if status:
            jobs = [job for job in jobs if job.status == status]
        if mode:
            jobs = [job for job in jobs if job.mode == mode]
        return jobs[:limit], None

    def runtime_status(self):
        return {
            "queue": 0,
            "running": None,
            "gpu": "1.0/16.0 GiB",
            "worker": "ready",
            "pid": 1234,
            "loaded_model": str(self.root / "secret-model.safetensors"),
            "loaded_resources": self.loaded_resources,
        }

    def set_event_sink(self, sink):
        self.sink = sink

    def set_preview_enabled(self, enabled):
        self.preview_enabled = enabled

    def stop(self):
        self.stopped = True

    def skip_current(self):
        self.skip_calls += 1
        if self.skip_error is not None:
            raise RuntimeError(self.skip_error)
        return self.skipped_id

    def shutdown(self):
        self.closed = True

    def list_video_models(self, video_model):
        if video_model not in self.config.video_resources:
            return []
        return self.video_model_items[video_model]

    def list_video_vaes(self, video_model):
        if video_model not in self.config.video_resources:
            return []
        return self.video_vae_items[video_model]

    def make_video_job(self, settings, batch_id=None, status="queued") -> VideoJobRecord:
        output_path = (
            self.config.output_dir
            / "2026-07-27"
            / f"{settings.video_model.value}-{len(self.video_jobs) + 1:05d}.mp4"
        )
        return VideoJobRecord(
            id=str(uuid.uuid4()),
            batch_id=batch_id,
            status=status,
            submitted_at=datetime.now().astimezone(),
            output_path=output_path,
            video_model=settings.video_model,
            prompt=settings.prompt,
            negative_prompt=settings.negative_prompt,
            model_path=settings.model.path,
            vae_path=settings.vae.path,
            text_encoder_path=settings.text_encoder,
            audio_vae_path=settings.audio_vae,
            sampler=settings.sampler,
            scheduler=settings.scheduler,
            width=settings.width,
            height=settings.height,
            duration_seconds=settings.duration_seconds,
            fps=settings.fps,
            length=settings.length,
            steps=settings.steps,
            seed=settings.seed,
            cfg=settings.cfg,
            shift=settings.shift,
            latent_multiplier=settings.latent_multiplier,
            input_image_path=settings.input_image,
            reference_image_path=settings.reference_image,
        )

    def submit_video(self, settings, count):
        settings.validate()
        self.video_submitted.append((settings, count))
        batch_id = str(uuid.uuid4()) if count > 1 else None
        jobs = []
        for _ in range(count):
            job = self.make_video_job(settings, batch_id=batch_id)
            self.video_jobs[job.id] = job
            jobs.append(job)
        return jobs

    def get_video_job(self, job_id):
        return self.video_jobs[job_id]

    def find_video_job_by_output(self, date_dir, name):
        for job in self.video_jobs.values():
            if job.output_path.name == name and job.output_path.parent.name == date_dir:
                return job
        raise KeyError(f"{date_dir}/{name}")
