# AGENTS.md

面向 AI 编码代理的项目说明。阅读本文假定你对本项目一无所知。

## 项目概览

diffusion-workbench 是一个基于 ComfyUI Core 的**本地单机文生图工作台**，支持
ZIT、Krea2、ZIB 三种模型模式（`Mode` 枚举）。项目不是 ComfyUI 的插件，而是把
ComfyUI 的 Python 环境当作推理后端：Core 以子进程方式启动一个长驻的 headless
ComfyUI Worker（`diffusion_workbench_core/comfy_worker.py`，必须用 ComfyUI 自带的
Python 运行），通过 stdin/stdout 上的 `DWB_EVENT=` JSON 行通信。

主要能力：单 GPU 串行任务队列、批量提交、模型跨任务复用与切模重载、跳过/停止、
SQLite 任务持久化与资源别名、PNG 内嵌完整生成参数（iTXt 块）、采样进度/速度/ETA
与 latent 预览事件、UTF-8 轮转日志。

运行目标环境为 Windows + RTX 5070 Ti（Blackwell, sm_120），因此 PyTorch 必须
使用 cu128 及以上构建（见 `pyproject.toml` 的 `[tool.uv.index]`）。

## 仓库布局

- `diffusion_workbench_core/` — 可复用核心层，不含任何 HTTP/TUI 代码。
  - `core.py` `WorkbenchCore`：唯一入口门面，组装配置、锁、日志、存储、目录、运行时、控制器。
  - `config.py`：加载 `workbench.yaml`（ComfyUI 路径、各模式模型/VAE/text encoder 目录、输出与数据库路径）。
  - `domain.py`：`Mode` / `ResourceKind` / `GenerationSettings` / `JobRecord` 及参数校验（采样器、步数、CFG 等）。
  - `persistent_runtime.py` + `comfy_worker.py`：Worker 子进程生命周期与事件协议。
  - `controller.py`：串行队列与任务状态机；`storage.py`：SQLite（任务、别名）；`catalog.py`：资源目录。
  - `instance_lock.py`：同一 SQLite 数据库同时只允许一个 Core 实例。
  - `png_metadata.py`：生成参数写入/读取 PNG iTXt。
- `diffusion_workbench/` — Textual TUI 与 CLI（入口 `diffusion-workbench = diffusion_workbench.cli:main`）。
- `diffusion_workbench_api/` — FastAPI 服务，把唯一的 `WorkbenchCore` 包装为 HTTP/WebSocket API。
  路由模块：`jobs`、`files`、`album`、`settings`、`models`、`llm`（提示词辅助）、`events`（WebSocket 事件流）、`video`（视频生成）。
  `console_status.py`：TTY 终端底部的任务进度状态栏（rich Live，非 TTY 为 no-op，`DWB_CONSOLE_STATUS=0` 关闭）。
  `app.py` 在 lifespan 中创建/关闭 Core，并静态托管 `frontend/dist`（存在时挂载到 `/`）。
- `frontend/` — React 18 + Vite 7 前端（`src/` 下 `App.jsx`、`api/client.js`、`components/`）。
- `demo/` — 演示/研究脚本（如 `demo_txt2img`、`demo_png_metadata`、krea2/zib runner），以 `uv run python -m demo.xxx` 运行。
- `configs/` — `workbench.example.yaml` 配置模板；`workbench.yaml` 主配置由它复制而来，
  含本机绝对路径，已被 `.gitignore` 忽略；`configs/krea2`、`configs/zit` 为 diffusers 管线配置。
- `tests/` — pytest 测试；`fake_api_core.py` 提供无 GPU 的 Core 替身。
- `AGENTS/` — 契约文档（中文），改动了对应行为时应同步更新（见下文"文档约定"）。
- `docs/` — 前期设计与研究文档，已被 `.gitignore` 忽略，仅供本地查阅。
- `output/`、`outputs/`、`.cache/` — 运行产物（git 已忽略）。

## 构建与运行

包管理用 **uv**（`uv.lock`，`tool.uv.package = true`），Python ≥ 3.12。

```powershell
uv sync                                                  # 安装依赖
uv run diffusion-workbench --config .\configs\workbench.yaml   # TUI
uv run uvicorn diffusion_workbench_api.app:app --host 127.0.0.1 --port 8188 --workers 1  # HTTP API
python start.py [--port 8188]                            # 一键：按需构建前端 + 启动 API
```

前端：

```powershell
cd frontend
npm install
npm run dev      # vite dev server :5173，已代理 /api(含 ws) 到 127.0.0.1:8188
npm run build    # 产物到 frontend/dist，由 FastAPI 同源托管
```

注意：`start.py` 用 `dist/.build-hash` 判断前端是否需要重新构建；vite preview 的
proxy 当前版本不转发 `/api`，不要用 `npm run preview` 对外提供页面。

## 测试

```powershell
uv run pytest            # 全部测试
uv run pytest tests/test_api_jobs.py   # 单个文件
```

- 测试不需要 GPU：API 测试通过 `create_app(core_factory=...)` 注入
  `tests/fake_api_core.py` 的 `FakeApiCore`（内存 JobRecord，记录调用）。
- 新增 API 行为应优先扩展 FakeCore 而不是引入真实 Worker。
- 没有 lint/format 配置（无 ruff/black/mypy），保持与周围代码一致即可。

## 架构约束（必须遵守）

- **单进程单实例**：一个 OS 进程、一个 `WorkbenchCore`、一个事件 sink、一个 GPU
  Worker。Uvicorn 必须 `--workers 1`，禁止 gunicorn 多 worker，TUI 与 API 服务
  互斥（由 SQLite 旁边的实例锁强制）。
- 所有生成任务只能经 `core.submit()` 串行执行；HTTP/TUI 层不得绕过 Core 启动
  第二套 GPU Worker 或重新实现队列/状态机。
- Core 只在 FastAPI lifespan 中创建/关闭，不要在模块 import 时创建。
- 客户端不能提交任何服务器路径；模型/VAE/text encoder 由服务端 `workbench.yaml`
  固定（`WorkbenchCore.submit` 会校验）。资源用 index 引用，目录变化后 index 会
  重排，不是永久 ID。
- 无鉴权、无限流、无 CORS allowlist：只绑定 `127.0.0.1` 单机使用，**不要暴露到网络**。
- `POST /api/v1/control/stop` 是全局停止（清队列并终止当前任务），没有单任务取消；
  `skip` 只终止当前任务、保留队列。
- WebSocket 事件不持久化、不重放，断线后用 `GET /api/v1/jobs/{id}` 恢复状态。

## 代码与文档约定

- 代码注释、docstring、错误信息和文档一律使用**中文**；标识符用英文。
- 输出图片写入 `output/YYYY-MM-DD/<mode>-NNNNN.png`，视频写入 `output/YYYY-MM-DD/<video_model>-NNNNN.mp4`；任务与别名存
  `.cache/diffusion_workbench.sqlite3`；日志为同目录 `.log`。
- 默认采样参数：Euler + simple、CFG 1、576×576（宽高须为 16 的倍数）、8 步、
  seed -1 表示随机。合法值以 `diffusion_workbench_core/domain.py` 为唯一事实来源。
- 文档即契约：`AGENTS/api/README.md`（REST/WebSocket 参考）、
  `AGENTS/core-technical-reference.md`（Core 实现契约）、
  `AGENTS/backend-integration.md`（部署约束）、
  `AGENTS/generation-preview-speed-and-metadata.md`（预览/速度/PNG 元数据契约）、
  `AGENTS/workbench-cli.md`（TUI 命令）。修改了对应行为时必须同步更新这些文档。

## 安全注意事项

- 服务无鉴权，仅监听回环地址；修改绑定地址前先加鉴权。
- `comfy_worker.py` 会以 ComfyUI 的 Python 启动子进程，配置中的路径均为本机绝对
  路径（见 `configs/workbench.yaml`），不要把用户输入拼进 Worker 命令行或文件路径。
- `.cache/`、`output/`、`*.log`、`.env` 均在 `.gitignore` 中，不要提交。
