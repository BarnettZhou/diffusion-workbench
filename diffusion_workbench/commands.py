from dataclasses import dataclass, replace

from diffusion_workbench_core.domain import (
    GenerationSettings,
    Mode,
    ResourceItem,
    ResourceKind,
    SAMPLERS,
    SCHEDULERS,
    IMAGE_UPSCALE_INTERPOLATIONS,
    LATENT_UPSCALE_INTERPOLATIONS,
    UpscaleMethod,
    UpscaleSettings,
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
        self.mode = Mode.ZIT
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
                    "/upscale ...  /start  /status  /skip  /stop  /exit",
                ))
            return CommandResponse((f"错误: 未知命令 /{command}",))
        except (ValueError, IndexError, RuntimeError) as exc:
            return CommandResponse((f"错误: {exc}",))

    def _mode(self, args: list[str]) -> CommandResponse:
        if len(args) > 1:
            raise ValueError("用法: /mode [zit|krea2|zib]")
        if args:
            self.mode = Mode(args[0].lower())
        else:
            modes = tuple(Mode)
            self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        return CommandResponse((f"mode 已切换为 {self.mode.value}",))

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
        if self.selected_vae is None:
            raise ValueError("请先使用 /vae set <index> 选择 VAE")
        resources = self.core.config.resources[self.mode]
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
        resources = self.core.config.resources[self.mode]
        prompt = self.prompt if len(self.prompt) <= 12 else self.prompt[:12] + "..."
        negative = (
            self.negative_prompt
            if len(self.negative_prompt) <= 12
            else self.negative_prompt[:12] + "..."
        )
        model = self.selected_model.display_name if self.selected_model else "未选择"
        vae = self.selected_vae.display_name if self.selected_vae else "未选择"
        lines = (
            f"mode: {self.mode.value}",
            f"model: {model}",
            f"vae: {vae}",
            f"text encoder: {resources.text_encoder.name}",
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
