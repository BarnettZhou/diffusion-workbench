from dataclasses import dataclass, replace
from pathlib import Path

from diffusion_workbench_core.domain import (
    GenerationSettings,
    Mode,
    ModelLoader,
    ResourceItem,
    ResourceKind,
    SAMPLERS,
    SCHEDULERS,
    IMAGE_UPSCALE_INTERPOLATIONS,
    LATENT_UPSCALE_INTERPOLATIONS,
    UpscaleMethod,
    UpscaleSettings,
    VideoGenerationSettings,
    VideoModel,
    validate_cfg,
    validate_steps,
)


@dataclass(frozen=True)
class CommandResponse:
    lines: tuple[str, ...] = ()
    exit_requested: bool = False


class CommandSession:
    def __init__(self, core):
        self.core = core
        self.mode = (
            Mode.ZIT
            if Mode.ZIT in core.config.resources
            else next(iter(core.config.resources), Mode.ZIT)
        )
        self.prompt = ""
        self.negative_prompt = ""
        self.width = 576
        self.height = 576
        self.steps = 8
        self.seed = -1
        self.cfg = 1.0
        self.sampler = "euler"
        self.scheduler = "simple"
        self.upscale = UpscaleSettings()
        self._models: dict[Mode, ResourceItem | None] = {mode: None for mode in Mode}
        self._vaes: dict[Mode, ResourceItem | None] = {mode: None for mode in Mode}
        self.video_model = next(
            iter(core.config.video_resources), VideoModel.WAN22_TI2V_5B
        )
        self.video_prompt = ""
        self.video_negative_prompt = ""
        self.video_width = 704
        self.video_height = 960
        self.video_duration = 5
        self.video_fps = 24
        self.video_steps = 20
        self.video_seed = -1
        self.video_cfg = 5.0
        self.video_sampler = "uni_pc"
        self.video_scheduler = "simple"
        self.video_image: Path | None = None
        self._video_models: dict[VideoModel, ResourceItem | None] = {
            model: None for model in VideoModel
        }
        self._video_vaes: dict[VideoModel, ResourceItem | None] = {
            model: None for model in VideoModel
        }

    @property
    def selected_model(self) -> ResourceItem | None:
        return self._models[self.mode]

    @property
    def selected_vae(self) -> ResourceItem | None:
        return self._vaes[self.mode]

    def handle(self, line: str) -> CommandResponse:
        line = line.strip()
        if not line.startswith("/"):
            return CommandResponse(("错误: 命令必须以 / 开头",))
        command_line = line[1:].strip()
        if not command_line:
            return CommandResponse(("错误: 空命令",))
        parts = command_line.split()
        command = parts[0].lower()
        args = parts[1:]
        try:
            if command == "mode":
                return self._mode(args)
            if command == "video":
                return self._video(args, command_line)
            if command == "resources":
                return self._resources(args)
            if command == "model":
                return self._resource(ResourceKind.DIFFUSION, args)
            if command == "vae":
                return self._resource(ResourceKind.VAE, args)
            if command == "prompt":
                self.prompt = command_line[len(parts[0]) :].strip()
                return CommandResponse(("prompt 已更新",))
            if command in {"negative", "negative-prompt"}:
                self.negative_prompt = command_line[len(parts[0]) :].strip()
                return CommandResponse(("negative prompt 已更新",))
            if command == "size":
                return self._size(args)
            if command == "steps":
                return self._steps(args)
            if command == "seed":
                return self._seed(args)
            if command == "cfg":
                return self._cfg(args)
            if command == "sampler":
                return self._choice("sampler", SAMPLERS, args)
            if command == "scheduler":
                return self._choice("scheduler", SCHEDULERS, args)
            if command == "upscale":
                return self._upscale(args)
            if command == "start":
                return self._start(args)
            if command == "status":
                return self._status()
            if command == "stop":
                self.core.stop()
                return CommandResponse(("已停止当前任务并清空队列",))
            if command == "skip":
                job_id = self.core.skip_current()
                if job_id is None:
                    return CommandResponse(("当前没有可跳过的任务",))
                return CommandResponse((f"已跳过当前任务: {job_id}",))
            if command == "exit":
                return CommandResponse(("正在卸载资源……",), exit_requested=True)
            if command == "help":
                return CommandResponse((
                    "/mode  /model list|set|set-alias  /vae list|set|set-alias",
                    "/prompt  /negative  /size  /steps  /seed  /cfg  /sampler  /scheduler",
                    "/video ...  /resources status|release",
                    "/upscale ...  /start  /status  /skip  /stop  /exit",
                ))
            return CommandResponse((f"错误: 未知命令 /{command}",))
        except (OSError, ValueError, IndexError, RuntimeError) as exc:
            return CommandResponse((f"错误: {exc}",))

    def _mode(self, args: list[str]) -> CommandResponse:
        if len(args) > 1:
            raise ValueError("用法: /mode [mode]")
        if args:
            mode = Mode(args[0].lower())
            if mode not in self.core.config.resources:
                raise ValueError(f"未配置 mode: {mode.value}")
            self.mode = mode
        else:
            modes = tuple(self.core.config.resources)
            self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        return CommandResponse((f"mode 已切换为 {self.mode.value}",))

    def _video(self, args: list[str], command_line: str) -> CommandResponse:
        if not args:
            raise ValueError(
                "用法: /video type|model|vae|prompt|negative|size|duration|fps|steps|"
                "seed|cfg|sampler|scheduler|image|status|start"
            )
        action = args[0].lower()
        rest = args[1:]
        if action == "type":
            if rest == ["list"]:
                lines = tuple(
                    f"[{model.value}]" for model in self.core.config.video_resources
                )
                return CommandResponse(lines or ("没有配置视频模型",))
            if not rest:
                return CommandResponse((f"video type: {self.video_model.value}",))
            if len(rest) != 2 or rest[0].lower() != "set":
                raise ValueError("用法: /video type list | set <wan2.2-ti2v-5b>")
            model = VideoModel(rest[1].lower())
            if model not in self.core.config.video_resources:
                raise ValueError(f"未配置视频模型: {model.value}")
            self.video_model = model
            self._video_models[model] = None
            self._video_vaes[model] = None
            return CommandResponse((f"video type 已设置为 {model.value}",))
        if action == "model":
            items = self.core.list_video_models(self.video_model)
            if rest == ["list"]:
                lines = tuple(
                    f"[{item.index}] {item.display_name}" for item in items
                )
                return CommandResponse(lines or ("没有可用视频模型",))
            if len(rest) == 2 and rest[0].lower() == "set":
                item = self._item_at(items, rest[1])
                self._video_models[self.video_model] = item
                return CommandResponse((f"video model 已设置为 {item.display_name}",))
            raise ValueError("用法: /video model list | set <index>")
        if action == "vae":
            items = self.core.list_video_vaes(self.video_model)
            if rest == ["list"]:
                lines = tuple(
                    f"[{item.index}] {item.display_name}" for item in items
                )
                return CommandResponse(lines or ("没有可用视频 VAE",))
            if len(rest) == 2 and rest[0].lower() == "set":
                item = self._item_at(items, rest[1])
                self._video_vaes[self.video_model] = item
                return CommandResponse((f"video VAE 已设置为 {item.display_name}",))
            raise ValueError("用法: /video vae list | set <index>")
        if action in {"prompt", "negative"}:
            value = command_line.split(None, 2)[2] if len(command_line.split(None, 2)) > 2 else ""
            if action == "prompt":
                self.video_prompt = value
            else:
                self.video_negative_prompt = value
            return CommandResponse((f"video {action} 已更新",))
        if action == "size":
            self._set_video_size(rest)
            return CommandResponse((f"video size 已设置为 {self.video_width}*{self.video_height}",))
        if action in {"duration", "fps", "steps", "seed", "cfg"}:
            if len(rest) != 1:
                raise ValueError(f"用法: /video {action} <value>")
            value = float(rest[0]) if action == "cfg" else int(rest[0])
            if action == "duration" and value <= 0:
                raise ValueError("视频时长必须大于 0")
            if action == "fps" and not 1 <= value <= 120:
                raise ValueError("帧率必须在 1 到 120 之间")
            if action == "steps":
                validate_steps(value)
            if action == "seed" and value < -1:
                raise ValueError("seed 必须为 -1 或非负整数")
            if action == "cfg":
                validate_cfg(value)
            setattr(self, f"video_{action}", value)
            displayed = f"{value:g}" if isinstance(value, float) else str(value)
            return CommandResponse((f"video {action} 已设置为 {displayed}",))
        if action in {"sampler", "scheduler"}:
            choices = SAMPLERS if action == "sampler" else SCHEDULERS
            if rest == ["list"]:
                return CommandResponse(tuple(f"[{index}] {name}" for index, name in enumerate(choices, 1)))
            if len(rest) != 2 or rest[0].lower() != "set":
                raise ValueError(f"用法: /video {action} list | set <name|index>")
            selected = self._choice_at(action, choices, rest[1])
            setattr(self, f"video_{action}", selected)
            return CommandResponse((f"video {action} 已设置为 {selected}",))
        if action == "image":
            if rest == ["clear"]:
                self.video_image = None
                return CommandResponse(("video image 已清除（T2V）",))
            if len(rest) >= 2 and rest[0].lower() == "set":
                raw = command_line.split(None, 2)[2].strip()
                if raw.lower().startswith("set "):
                    raw = raw[4:].strip()
                if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
                    raw = raw[1:-1]
                path = Path(raw).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"找不到输入图片: {path}")
                self.video_image = path
                return CommandResponse((f"video image 已设置为 {path}",))
            raise ValueError("用法: /video image set <path> | clear")
        if action == "status":
            model = self._video_models[self.video_model]
            vae = self._video_vaes[self.video_model]
            image = str(self.video_image) if self.video_image else "无（T2V）"
            return CommandResponse((
                f"video type: {self.video_model.value} ({'I2V' if self.video_image else 'T2V'})",
                f"model: {model.display_name if model else '未选择'}",
                f"vae: {vae.display_name if vae else '未选择'}",
                f"prompt: {self.video_prompt or '未设置'}",
                f"size: {self.video_width}*{self.video_height}  "
                f"duration: {self.video_duration}s  fps: {self.video_fps}  "
                f"length: {self.video_duration * self.video_fps + 1}",
                f"steps: {self.video_steps}  seed: {self.video_seed}  cfg: {self.video_cfg:g}",
                f"sampler: {self.video_sampler}  scheduler: {self.video_scheduler}  denoise: 1",
                f"image: {image}",
            ))
        if action == "start":
            return self._start_video(rest)
        raise ValueError(f"未知视频命令: {action}")

    def _set_video_size(self, args: list[str]) -> None:
        if len(args) != 1 or "*" not in args[0]:
            raise ValueError("用法: /video size <width>*<height>")
        width, height = (int(value) for value in args[0].split("*", 1))
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("视频宽高必须为正数且是 16 的倍数")
        self.video_width, self.video_height = width, height

    def _start_video(self, args: list[str]) -> CommandResponse:
        if len(args) > 1:
            raise ValueError("用法: /video start [num]")
        model = self._video_models[self.video_model]
        if model is None:
            raise ValueError("请先使用 /video model set <index> 选择模型")
        vae = self._video_vaes[self.video_model]
        if vae is None:
            raise ValueError("请先使用 /video vae set <index> 选择 VAE")
        resources = self.core.config.video_resources[self.video_model]
        settings = VideoGenerationSettings(
            video_model=self.video_model,
            model=model,
            vae=vae,
            text_encoder=resources.text_encoder,
            prompt=self.video_prompt,
            negative_prompt=self.video_negative_prompt,
            input_image=self.video_image,
            width=self.video_width,
            height=self.video_height,
            duration_seconds=self.video_duration,
            fps=self.video_fps,
            steps=self.video_steps,
            seed=self.video_seed,
            cfg=self.video_cfg,
            sampler=self.video_sampler,
            scheduler=self.video_scheduler,
        )
        jobs = self.core.submit_video(settings, int(args[0]) if args else 1)
        return CommandResponse((f"已加入视频队列: {len(jobs)} 个任务",))

    def _resources(self, args: list[str]) -> CommandResponse:
        if args == ["status"]:
            status = self.core.runtime_status()
            loaded = status.get("loaded_resources") or {}
            return CommandResponse((
                f"worker: {status.get('worker', 'stopped')}  workload: {loaded.get('workload') or '无'}",
                f"model: {loaded.get('model') or '无'}",
                f"vae: {loaded.get('vae') or '无'}",
                f"text encoder: {loaded.get('text_encoder') or '无'}  "
                f"clip_type: {loaded.get('clip_type') or '无'}",
                f"gpu: {status.get('gpu', '不可用')}",
            ))
        if args == ["release"]:
            self.core.release_resources()
            return CommandResponse(("已释放 Worker 模型和显存资源",))
        raise ValueError("用法: /resources status | release")

    def _resource(self, kind: ResourceKind, args: list[str]) -> CommandResponse:
        label = "model" if kind == ResourceKind.DIFFUSION else "vae"
        if not args:
            raise ValueError(f"用法: /{label} list|set|set-alias")
        action = args[0].lower()
        items = self.core.list_resources(self.mode, kind)
        if action == "list" and len(args) == 1:
            if not items:
                return CommandResponse((f"当前 {self.mode.value} 没有可用 {label}",))
            return CommandResponse(
                tuple(f"[{item.index}] {item.display_name}" for item in items)
            )
        if action == "set" and len(args) == 2:
            item = self._item_at(items, args[1])
            target = self._models if kind == ResourceKind.DIFFUSION else self._vaes
            target[self.mode] = item
            return CommandResponse((f"{label} 已设置为 {item.display_name}",))
        if action == "set-alias" and len(args) >= 3:
            item = self._item_at(items, args[1])
            alias = " ".join(args[2:]).strip()
            self.core.set_alias(self.mode, kind, item.path, alias)
            updated = ResourceItem(item.index, item.path, alias)
            target = self._models if kind == ResourceKind.DIFFUSION else self._vaes
            if target[self.mode] and target[self.mode].path == item.path:
                target[self.mode] = updated
            return CommandResponse((f"{label} [{item.index}] alias 已设置为 {alias}",))
        raise ValueError(f"用法: /{label} list | set <index> | set-alias <index> <alias-name>")

    @staticmethod
    def _item_at(items: list[ResourceItem], raw_index: str) -> ResourceItem:
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError("index 必须是整数") from exc
        for item in items:
            if item.index == index:
                return item
        raise IndexError(f"找不到 index {index}")

    def _size(self, args: list[str]) -> CommandResponse:
        if len(args) != 1 or "*" not in args[0]:
            raise ValueError("用法: /size <width>*<height>")
        width_text, height_text = args[0].split("*", 1)
        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("size 必须为正数且是 16 的倍数")
        self.width, self.height = width, height
        return CommandResponse((f"size 已设置为 {width}*{height}",))

    def _steps(self, args: list[str]) -> CommandResponse:
        if len(args) != 1:
            raise ValueError("用法: /steps <1-100>")
        steps = int(args[0])
        validate_steps(steps)
        self.steps = steps
        return CommandResponse((f"steps 已设置为 {steps}",))

    def _seed(self, args: list[str]) -> CommandResponse:
        if len(args) != 1:
            raise ValueError("用法: /seed <-1|非负整数>")
        seed = int(args[0])
        if seed < -1:
            raise ValueError("seed 必须为 -1 或非负整数")
        self.seed = seed
        return CommandResponse((f"seed 已设置为 {seed}",))

    def _cfg(self, args: list[str]) -> CommandResponse:
        if len(args) != 1:
            raise ValueError("用法: /cfg <正数>")
        cfg = float(args[0])
        validate_cfg(cfg)
        self.cfg = cfg
        return CommandResponse((f"cfg 已设置为 {cfg:g}",))

    def _choice(
        self,
        label: str,
        choices: tuple[str, ...],
        args: list[str],
    ) -> CommandResponse:
        if len(args) == 1 and args[0].lower() == "list":
            return CommandResponse(
                tuple(f"[{index}] {name}" for index, name in enumerate(choices, 1))
            )
        if not args or args[0].lower() != "set":
            raise ValueError(f"用法: /{label} list | set <name|index>")
        value_args = args[1:]
        if len(value_args) != 1:
            raise ValueError(f"用法: /{label} list | set <name|index>")
        selected = self._choice_at(label, choices, value_args[0])
        setattr(self, label, selected)
        return CommandResponse((f"{label} 已设置为 {selected}",))

    @staticmethod
    def _choice_at(label: str, choices: tuple[str, ...], value: str) -> str:
        try:
            index = int(value)
        except ValueError:
            normalized = value.lower()
            if normalized in choices:
                return normalized
        else:
            if 1 <= index <= len(choices):
                return choices[index - 1]
        raise ValueError(f"找不到 {label}: {value}")

    def _start(self, args: list[str]) -> CommandResponse:
        if len(args) > 1:
            raise ValueError("用法: /start [num]")
        count = int(args[0]) if args else 1
        if count <= 0:
            raise ValueError("任务数量必须大于 0")
        if self.selected_model is None:
            raise ValueError("请先使用 /model set <index> 选择模型")
        resources = self.core.config.resources[self.mode]
        if (
            resources.model_loader == ModelLoader.COMPONENTS
            and self.selected_vae is None
        ):
            raise ValueError("请先使用 /vae set <index> 选择 VAE")
        settings = GenerationSettings(
            mode=self.mode,
            model=self.selected_model,
            vae=self.selected_vae,
            text_encoder=resources.text_encoder,
            clip_type=resources.clip_type,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            steps=self.steps,
            seed=self.seed,
            cfg=self.cfg,
            sampler=self.sampler,
            scheduler=self.scheduler,
            upscale=self.upscale,
            model_loader=resources.model_loader,
        )
        settings.validate()
        jobs = self.core.submit(settings, count)
        return CommandResponse((f"已加入队列: {len(jobs)} 个任务",))

    def _upscale(self, args: list[str]) -> CommandResponse:
        if not args:
            raise ValueError("用法: /upscale on|off|status|reset|<参数>")
        action = args[0].lower()
        rest = args[1:]
        if action in {"on", "off"} and not rest:
            self._set_upscale(enabled=action == "on")
            return CommandResponse((f"图片放大已{('开启' if action == 'on' else '关闭')}",))
        if action == "status" and not rest:
            return self._upscale_status()
        if action == "reset" and not rest:
            self.upscale = UpscaleSettings()
            return CommandResponse(("图片放大设置已重置",))
        if action == "method":
            if rest == ["list"]:
                return self._list_choices(tuple(method.value for method in UpscaleMethod))
            selected = self._set_choice(
                "upscale method",
                tuple(method.value for method in UpscaleMethod),
                rest,
            )
            method = UpscaleMethod(selected)
            interpolation = self.upscale.interpolation
            if method == UpscaleMethod.LATENT_HIRES:
                if interpolation not in LATENT_UPSCALE_INTERPOLATIONS:
                    interpolation = "bislerp"
            elif interpolation not in IMAGE_UPSCALE_INTERPOLATIONS:
                interpolation = "lanczos"
            self.upscale = replace(
                self.upscale, method=method, interpolation=interpolation
            )
            return CommandResponse((f"放大方法已设置为 {method.value}",))
        if action == "scale" and len(rest) == 1:
            self._set_upscale(scale=float(rest[0]))
            return CommandResponse((f"放大倍数已设置为 {self.upscale.scale:g}",))
        if action == "interpolation":
            choices = (
                LATENT_UPSCALE_INTERPOLATIONS
                if self.upscale.method == UpscaleMethod.LATENT_HIRES
                else IMAGE_UPSCALE_INTERPOLATIONS
            )
            if rest == ["list"]:
                return self._list_choices(choices)
            selected = self._set_choice("interpolation", choices, rest)
            self.upscale = replace(self.upscale, interpolation=selected)
            return CommandResponse((f"放大插值已设置为 {selected}",))
        if action == "model":
            models = self.core.list_upscale_models()
            if rest == ["list"]:
                if not models:
                    return CommandResponse(("没有可用的放大模型",))
                return CommandResponse(
                    tuple(f"[{item.index}] {item.display_name}" for item in models)
                )
            if len(rest) == 2 and rest[0].lower() == "set":
                item = self._resource_at(models, rest[1])
                self.upscale = replace(self.upscale, model=item)
                return CommandResponse((f"放大模型已设置为 {item.display_name}",))
            raise ValueError("用法: /upscale model list | set <name|index>")
        if action == "tile" and len(rest) == 1:
            self._set_upscale(tile=int(rest[0]))
            return CommandResponse((f"tile 已设置为 {self.upscale.tile}",))
        if action == "overlap" and len(rest) == 1:
            self._set_upscale(overlap=int(rest[0]))
            return CommandResponse((f"overlap 已设置为 {self.upscale.overlap}",))
        if action == "steps" and len(rest) == 1:
            self._set_upscale(steps=int(rest[0]))
            return CommandResponse((f"二次采样总步数已设置为 {self.upscale.steps}",))
        if action == "start-step" and len(rest) == 1:
            self._set_upscale(start_step=int(rest[0]))
            return CommandResponse((f"二次采样开始步已设置为 {self.upscale.start_step}",))
        if action == "cfg" and len(rest) == 1:
            value = None if rest[0].lower() == "inherit" else float(rest[0])
            self._set_upscale(cfg=value)
            return CommandResponse((f"二次采样 CFG 已设置为 {self._inherit(value)}",))
        if action in {"sampler", "scheduler"}:
            choices = SAMPLERS if action == "sampler" else SCHEDULERS
            if rest == ["list"]:
                return CommandResponse(("[0] inherit",) + self._list_choices(choices).lines)
            if len(rest) != 2 or rest[0].lower() != "set":
                raise ValueError(
                    f"用法: /upscale {action} list | set <name|index|inherit>"
                )
            value = (
                None
                if rest[1].lower() == "inherit" or rest[1] == "0"
                else self._choice_at(action, choices, rest[1])
            )
            self.upscale = replace(self.upscale, **{action: value})
            return CommandResponse((f"二次采样 {action} 已设置为 {self._inherit(value)}",))
        if action == "seed" and len(rest) == 1:
            value = None if rest[0].lower() == "inherit" else int(rest[0])
            self._set_upscale(seed=value)
            return CommandResponse((f"放大 seed 已设置为 {self._inherit(value)}",))
        raise ValueError("无效的 /upscale 命令；使用 /upscale status 查看当前设置")

    def _upscale_status(self) -> CommandResponse:
        value = self.upscale
        target_width = round(self.width * value.scale)
        target_height = round(self.height * value.scale)
        lines = [
            f"放大: {'开启' if value.enabled else '关闭'}  方法: {value.method.value}",
            f"倍数: {value.scale:g}  预计尺寸: {self.width}*{self.height} -> {target_width}*{target_height}",
            f"插值: {value.interpolation}",
        ]
        if value.method == UpscaleMethod.UPSCALE_MODEL:
            lines.extend(
                (
                    f"模型: {value.model.display_name if value.model else '未选择'}",
                    f"tile: {value.tile}  overlap: {value.overlap}",
                )
            )
        elif value.method == UpscaleMethod.LATENT_HIRES:
            lines.extend(
                (
                    f"二次采样: steps={value.steps}  start-step={value.start_step}  实际执行: {value.executed_steps} 步",
                    f"CFG: {self._inherit(value.cfg)}  sampler: {self._inherit(value.sampler)}  scheduler: {self._inherit(value.scheduler)}",
                    f"seed: {self._inherit(value.seed)}",
                )
            )
        return CommandResponse(tuple(lines))

    @staticmethod
    def _list_choices(choices: tuple[str, ...]) -> CommandResponse:
        return CommandResponse(
            tuple(f"[{index}] {name}" for index, name in enumerate(choices, 1))
        )

    def _set_choice(
        self, label: str, choices: tuple[str, ...], args: list[str]
    ) -> str:
        if len(args) != 2 or args[0].lower() != "set":
            raise ValueError(f"用法: set <name|index>")
        return self._choice_at(label, choices, args[1])

    @staticmethod
    def _resource_at(items: list[ResourceItem], value: str) -> ResourceItem:
        try:
            index = int(value)
        except ValueError:
            normalized = value.casefold()
            for item in items:
                if item.path.name.casefold() == normalized:
                    return item
        else:
            for item in items:
                if item.index == index:
                    return item
        raise ValueError(f"找不到放大模型: {value}")

    @staticmethod
    def _inherit(value) -> str:
        if value is None:
            return "inherit"
        return f"{value:g}" if isinstance(value, float) else str(value)

    def _set_upscale(self, **changes) -> None:
        candidate = replace(self.upscale, **changes)
        candidate.validate()
        self.upscale = candidate

    def _status(self) -> CommandResponse:
        runtime = self.core.runtime_status()
        resources = self.core.config.resources.get(self.mode)
        if resources is None:
            loaded = runtime.get("loaded_resources") or {}
            return CommandResponse((
                "图片模式未配置",
                f"视频资源 workload: {loaded.get('workload') or '无'}",
                f"gpu: {runtime.get('gpu', '不可用')}",
            ))
        prompt = self.prompt if len(self.prompt) <= 12 else self.prompt[:12] + "..."
        negative = (
            self.negative_prompt
            if len(self.negative_prompt) <= 12
            else self.negative_prompt[:12] + "..."
        )
        model = self.selected_model.display_name if self.selected_model else "未选择"
        vae = (
            "checkpoint 内嵌"
            if resources.model_loader == ModelLoader.CHECKPOINT
            else self.selected_vae.display_name if self.selected_vae else "未选择"
        )
        text_encoder = (
            resources.text_encoder.name
            if resources.text_encoder is not None
            else "checkpoint 内嵌"
        )
        lines = (
            f"mode: {self.mode.value}",
            f"model: {model}",
            f"vae: {vae}",
            f"text encoder: {text_encoder}",
            f"prompt: {prompt or '未设置'}",
            f"negative: {negative or '未设置'}",
            f"size: {self.width}*{self.height}",
            f"steps: {self.steps}  seed: {self.seed}  cfg: {self.cfg:g}",
            f"sampler: {self.sampler}  scheduler: {self.scheduler}",
            f"queue: {runtime.get('queue', 0)}  running: {runtime.get('running') or '无'}",
            f"worker: {runtime.get('worker', 'stopped')}  pid: {runtime.get('pid') or '无'}",
            f"gpu: {runtime.get('gpu', '不可用')}",
        )
        return CommandResponse(lines)
