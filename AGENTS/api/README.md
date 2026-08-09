# diffusion-workbench HTTP API

`diffusion_workbench_api` 包把唯一的 `WorkbenchCore` 包装为单机 FastAPI 服务。它是
Core 的调用端，不重新实现 GPU 队列、SQLite 状态机或 ComfyUI Worker 生命周期。

- [REST API 参考](rest.md)
- [WebSocket 事件流](events.md)
- 设计约束与接入细节：[../backend-integration.md](../backend-integration.md)
- Core 内部契约：[../core-technical-reference.md](../core-technical-reference.md)
- 预览/性能/PNG 元数据契约：[../generation-preview-speed-and-metadata.md](../generation-preview-speed-and-metadata.md)
- Core 日志与任务审计记录：[../core-logging-and-job-audit.md](../core-logging-and-job-audit.md)

## 运行

```powershell
uv run uvicorn diffusion_workbench_api.app:app --host 127.0.0.1 --port 8188 --workers 1
```

配置文件默认 `configs/workbench.yaml`，可用环境变量 `DWB_CONFIG` 覆盖。

服务同时托管前端构建产物（`frontend/dist`，存在时挂载到 `/`），单端口同源访问，
无需反向代理；前端开发调试可用 `frontend/` 下的 vite dev server（已配置 `/api` 代理）。
vite preview 的 `preview.proxy` 当前版本不会转发 `/api`，不要用它对外提供页面。

## 部署约束（必须满足）

- 一个 OS 进程、一个 `WorkbenchCore`、一个事件 sink、一个 GPU Worker。
- Uvicorn 必须 `--workers 1`。禁止 gunicorn 多 worker、禁止与 TUI 同时运行。
- Core 在 FastAPI lifespan 中创建和关闭；不要在模块 import 时创建。
- 所有生成任务只经 `core.submit()` 串行执行。

## 当前边界（第一版）

- 无鉴权、无限流、无 CORS allowlist：只绑定 `127.0.0.1` 单机使用，不要暴露到网络。
- 无单任务取消：`POST /api/v1/control/stop` 是全局停止（清空整个队列并终止当前任务）。
- 资源用 index 引用；目录内容变化后 index 会重新排序，不是永久 ID。
- WebSocket 事件不持久化、不重放；断线后通过 `GET /api/v1/jobs/{id}` 恢复状态。
- 客户端不能提交任何服务器路径；模型/VAE/text encoder 由服务端配置固定。

## API 概览

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/api/v1/health` | 进程存活 |
| `GET` | `/api/v1/status` | 队列、running job、Worker、GPU |
| `GET` | `/api/v1/sampling-options` | 全部可用采样器/调度器及默认值 |
| `GET` | `/api/v1/upscale-options` | 图片放大方法/插值/采样选项及默认值 |
| `GET` | `/api/v1/upscale-models` | 放大模型列表（不含路径） |
| `GET` | `/api/v1/resources/{mode}/{kind}` | 模型或 VAE 列表 |
| `PUT` | `/api/v1/resources/{mode}/{kind}/{index}/alias` | 设置资源别名 |
| `POST` | `/api/v1/jobs` | 提交一张或一批任务（202） |
| `GET` | `/api/v1/jobs` | 历史任务分页 |
| `GET` | `/api/v1/jobs/{job_id}` | 读取单个任务持久化状态 |
| `POST` | `/api/v1/control/stop` | 全局停止并清空队列 |
| `POST` | `/api/v1/control/skip` | 跳过当前任务，保留队列 |
| `GET` | `/api/v1/settings` | 读取全部设置项 |
| `PUT` | `/api/v1/settings` | 更新设置项（部分字段） |
| `POST` | `/api/v1/prompt-assist` | 大模型提示词生成（FLUX/SD 风格，一次性非流式） |
| `POST` | `/api/v1/prompt-assist/chat` | 对话式提示词生成（FLUX/SD 风格，SSE 流式） |
| `GET` | `/api/v1/llm/requests` | 大模型请求记录（按时间倒序分页，保留最近 200 条） |
| `GET` | `/api/v1/models/{mode}` | 模型列表（别名/大小/量化/备注/封面） |
| `GET` | `/api/v1/models/{mode}/{name}/cover` | 模型封面图片 |
| `PUT` | `/api/v1/models/{mode}/{name}/cover` | 上传模型封面（png/jpeg/webp） |
| `PUT` | `/api/v1/models/{mode}/{name}/info` | 更新模型别名/备注 |
| `GET` | `/api/v1/images/{job_id}` | 下载任务原始 PNG（文件存在即可，不限 completed） |
| `GET` | `/api/v1/images/{job_id}/upscaled` | 下载任务的放大图 PNG |
| `GET` | `/api/v1/images/{job_id}/metadata` | 读取 PNG 内嵌生成参数（脱敏） |
| `GET` | `/api/v1/images/{job_id}/upscaled/metadata` | 读取放大图内嵌生成参数（脱敏） |
| `GET` | `/api/v1/album` | output 目录图片索引（扫盘，按时间倒序分页） |
| `GET` | `/api/v1/album/dirs` | 相册目录列表（内置 output + 用户目录） |
| `POST` | `/api/v1/album/dirs` | 添加相册目录 |
| `PUT` | `/api/v1/album/dirs/{dir_id}` | 相册目录改名 |
| `DELETE` | `/api/v1/album/dirs/{dir_id}` | 移除相册目录（不动磁盘文件） |
| `GET` | `/api/v1/album/image/{path}` | 按相对路径读取图片 |
| `DELETE` | `/api/v1/album/image/{path}` | 从本机删除该图片 |
| `GET` | `/api/v1/album/image/{path}/metadata` | 按相对路径读取 PNG 元数据 |
| `WS` | `/api/v1/events` | 实时公共事件流 |

FastAPI 自动生成的 OpenAPI 文档位于 `/docs`(Swagger UI)和 `/openapi.json`。
