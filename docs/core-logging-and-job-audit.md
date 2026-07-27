# Core 日志与任务审计记录

本文定义 `diffusion_workbench_core` 的两类持久化诊断数据：SQLite 任务记录和 Python
运行日志。二者职责不同，调用端不应根据图片文件是否仍存在来修改任务历史。

## SQLite 是任务审计账本

`jobs` 表保存任务提交时的完整参数和最终执行结果，包括 prompt、seed、mode、模型、VAE、
text encoder、sampler、scheduler、尺寸、步数、CFG、提交/开始/完成时间、耗时、状态、错误
和生成时的 `output_path`。

`output_path` 是图片成功生成时的路径快照，也是图片下载接口的 best-effort 定位信息，
不是任务记录的外键或有效性条件。图片随后可能被移动、改名或删除，此时：

- SQLite 任务读取接口仍返回完整任务记录；
- completed 状态不会因文件缺失自动改为 failed 或被删除；
- FastAPI `GET /api/v1/images/{job_id}` 返回 `410 Gone`；
- 调用端仍可通过任务接口查看生成参数、时间、状态和错误。

该语义不需要 SQLite schema 迁移，也不改变 Core 或 FastAPI 的 DTO、事件和状态字段。
日志不能替代 SQLite；日志用于排查运行过程，SQLite 才是持久任务事实来源。

图片自己的 PNG iTXt 元数据可独立用于复现。即使 SQLite 不可用，仍可调用
`read_generation_metadata(path)` 读取；即使图片丢失，SQLite 中仍保留原任务参数。

## Core Python 日志

每个 `WorkbenchCore` 实例创建一个 UTF-8 轮转日志。路径由数据库路径派生：

```text
.cache/diffusion-workbench.sqlite3
.cache/diffusion-workbench.log
```

单个日志文件最多 10 MiB，保留 5 个轮转备份：

```text
diffusion-workbench.log
diffusion-workbench.log.1
...
diffusion-workbench.log.5
```

日志格式固定为：

```text
[INFO] 2026-07-27 12:00:00 job started job_id=... mode=zit seed=42 ...
[WARN] 2026-07-27 12:00:05 job cancelled job_id=... reason=...
[ERROR] 2026-07-27 12:00:08 job failed job_id=...
```

Python `WARNING` 显示为 `[WARN]`。Core 使用独立文件 handler 且不向 root logger 传播，
因此 TUI 不会新增控制台噪声；FastAPI 也无需修改现有日志配置。`core.log_path` 可用于定位
当前日志文件，但 `runtime_status()` 和 HTTP 响应不会新增路径字段。

## 记录范围

Core 日志覆盖：

- Core 启动、就绪和关闭；
- 任务入队、开始、阶段、每步采样速度与 ETA、完成、取消和失败；
- Worker 启动、PID、就绪、关闭、强制终止和异常退出；
- Worker 的普通 stdout/stderr；以 `WARN`/`WARNING` 开头的行记为 `[WARN]`，以
  `ERROR`/`CRITICAL`/`FATAL`/`Traceback` 开头的行记为 `[ERROR]`，其他行记为 `[INFO]`；
- Worker 输出的无效 IPC 事件和 Core event sink 异常。

Worker 最近 100 行输出仍保留在内存中，用于拼接异常退出信息；持久日志是额外记录，
不会改变原有异常、事件或 API 行为。

Core 不把 prompt 写入 Python 日志，避免常规诊断日志复制用户内容。完整 prompt 保存在
SQLite 和 PNG 元数据中。日志会包含 job id、seed、模型文件名、输出路径、错误和 traceback，
仍应按本机敏感诊断数据管理，不应通过未鉴权接口公开。

## 运维判断

- 查询“任务提交了什么、最终状态是什么”：读 SQLite。
- 查询“某次运行在哪个阶段出错、Worker 输出了什么”：读 Core 日志。
- 查询“这张现存图片如何复现”：读 PNG 元数据。
- 图片接口返回 `410`：只代表路径快照对应的文件已不存在，不代表任务记录损坏。
