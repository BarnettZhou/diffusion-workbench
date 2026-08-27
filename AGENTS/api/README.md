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
- 客户端不能提交任何服务器路径；模型/VAE/text encoder 按 index 引用服务端配置的
  资源列表（图片模式的 text encoder 为目录/文件列表，可挑选），clip type 由配置固定。

## API 概览

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/api/v1/health` | 进程存活 |
| `GET` | `/api/v1/status` | 队列、running job、Worker、系统内存、显存与 GPU 利用率 |
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
| `POST` | `/api/v1/control/release` | 释放 Worker 已加载的模型资源（空闲时才允许，409 冲突） |
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
| `GET` | `/api/v1/album` | output 目录图片/视频索引（扫盘，按时间倒序分页，含 `kind` 字段） |
| `GET` | `/api/v1/album/dirs` | 相册目录列表（内置 output + 用户目录） |
| `POST` | `/api/v1/album/dirs` | 添加相册目录 |
| `PUT` | `/api/v1/album/dirs/{dir_id}` | 相册目录改名 |
| `DELETE` | `/api/v1/album/dirs/{dir_id}` | 移除相册目录（不动磁盘文件） |
| `GET` | `/api/v1/album/image/{path}` | 按相对路径读取图片 |
| `DELETE` | `/api/v1/album/image/{path}` | 从本机删除该图片 |
| `POST` | `/api/v1/album/batch-delete` | 批量删除图片/视频（单个失败不影响其余） |
| `GET` | `/api/v1/album/image/{path}/metadata` | 按相对路径读取 PNG 元数据（mp4 返回 404） |
| `GET` | `/api/v1/video/models` | 已配置的视频模型及 T2V/I2V 能力 |
| `GET` | `/api/v1/video/models/{video_model}/resources` | 视频模型的 diffusion/VAE/text encoder 资源 |
| `POST` | `/api/v1/video/input-images` | 受控上传 I2V 输入图片（201） |
| `POST` | `/api/v1/video/input-images/from-album` | 从相册导入 I2V 输入图片（201） |
| `POST` | `/api/v1/video/input-images/from-job` | 从生成/编辑任务输出图导入 I2V 输入图片（201） |
| `GET` | `/api/v1/video/input-images/{image_id}` | 按 id 读取上传的输入图片 |
| `POST` | `/api/v1/video/input-videos` | 受控上传 Ref2VA 参考视频（201） |
| `GET` | `/api/v1/video/input-videos/{video_id}` | 按 id 读取上传的参考视频 |
| `POST` | `/api/v1/video/input-audios` | 受控上传 Ref2VA 参考音频（201） |
| `GET` | `/api/v1/video/input-audios/{audio_id}` | 按 id 读取上传的参考音频 |
| `POST` | `/api/v1/video/jobs` | 提交视频生成任务（202,T2V/I2V/R2V） |
| `GET` | `/api/v1/video/jobs/{job_id}` | 读取单个视频任务状态 |
| `GET` | `/api/v1/videos/{job_id}` | 下载视频任务的 mp4 |
| `GET` | `/api/v1/edit/info` | Krea2 图像编辑可用性与默认参数 |
| `POST` | `/api/v1/edit/input-images` | 受控上传编辑输入图片（201） |
| `POST` | `/api/v1/edit/input-images/from-album` | 从相册导入编辑输入图片（201） |
| `POST` | `/api/v1/edit/input-images/from-job` | 从生成/编辑任务输出图导入编辑输入图片（201；反推共用本通道） |
| `GET` | `/api/v1/edit/input-images/{image_id}` | 按 id 读取上传的编辑输入图片 |
| `POST` | `/api/v1/edit/jobs` | 提交 Krea2 图像编辑任务（202） |
| `POST` | `/api/v1/caption` | 图片反推（同步；输入图复用 `/edit/input-images` 的受控 id） |
| `POST` | `/api/v1/caption/remote` | API 反推（同步；走 `caption_api` 设置的外部视觉模型接口，不经 GPU Worker） |
| `POST` | `/api/v1/caption/remote/test` | 反推 API 连通性测试 |
| `GET` | `/api/v1/caption/remote/requests` | API 反推请求记录（分页，最多保留 200 条） |
| `GET` | `/api/v1/video-models/{video_model}` | 视频模型卡片列表（量化/备注/封面） |
| `GET`/`PUT` | `/api/v1/video-models/{video_model}/{name}/cover` | 视频模型封面读取/上传 |
| `POST` | `/api/v1/video-models/{video_model}/{name}/quant` | 扫描视频模型量化方式 |
| `PUT` | `/api/v1/video-models/{video_model}/{name}/info` | 更新视频模型备注（不支持 alias） |
| `WS` | `/api/v1/events` | 实时公共事件流 |

FastAPI 自动生成的 OpenAPI 文档位于 `/docs`(Swagger UI)和 `/openapi.json`。
