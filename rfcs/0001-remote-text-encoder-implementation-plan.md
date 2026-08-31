# RFC 0001 远端 Text Encoder 落地计划

> 本文是 `rfcs/0001-remote-text-encoder.md` 的实施计划和前置决策记录。
> 目标是先验证可行性，再逐步接入生产路径；不直接按草案一次性实现全部功能。

## 1. 当前环境结论

### Windows 本端 CUDA

- 项目 `pyproject.toml` 使用 PyTorch cu128 index。
- `uv.lock` 当前锁定 `torch 2.11.0+cu128`。
- 当前 ComfyUI 整合环境的既有实测记录为 `torch 2.7.0+cu128`。
- cu130 尚未作为当前运行环境；它只应作为可回滚的独立实验，用于验证
  `comfy-kitchen` 的优化 CUDA backend。cu128 下仍可使用 eager fallback，不能把
  “可运行”与“启用优化 kernel”混为一谈。

### Mac ComfyUI 环境

- 官方 ComfyUI DMG 是桌面应用封装，不等同于本项目所需的源码目录 + 可调用的
  Python 解释器。
- 本项目的 Worker 需要直接执行 `python comfy_worker.py --comfy-root ...`，并导入
  `comfy`、`comfy_extras` 以及自定义节点源码。
- 因此远端 Encoder 的基线部署应使用 Mac 上单独的源码版 ComfyUI、独立 Python
  环境和独立配置。DMG 可以继续用于交互式使用，但不作为 RFC 0001 的服务运行时。
- 是否能在 Apple Silicon/MPS 上运行目标 Qwen3-VL 和两个 Krea2 自定义节点，必须
  通过 P0 冒烟测试确认，不能只根据统一内存容量推断。

## 2. 必须先修订的设计

### 2.1 资源标识不能传本地绝对路径

Windows 路径对 Mac 无效，也不应让远端接受任意路径。`LoadClip` 应改为逻辑
`encoder_id` + 模型 fingerprint + `clip_type`，Mac 服务通过本地配置把
`encoder_id` 映射到自己的文件路径。

### 2.2 Conditioning 不能建模为单一 Tensor

ComfyUI conditioning 是嵌套的 list/tuple/dict，除 embedding 外还可能包含
`pooled_output`、mask 等张量和元数据。协议需要使用带版本号的递归 payload，安全
编解码 list/tuple/dict/tensor；不能只定义 `Conditioning.cond` 一个字段，也不能对
网络输入直接执行不受控 pickle 反序列化。

### 2.3 远端化范围必须按模式隔离

当前 `self.clip` 同时服务图片生成、Wan/MiniMax H3 视频和 `describe_image`。
第一版只远端化：

- ZIT/Krea2/ZIB 纯文本 EncodeText；
- `edit-krea2` 的 grounded encode；
- `krea2-rebalance` 的参考图 encode。

视频和 `describe_image` 暂留本地，不能通过全局改造 `_ensure_clip` 意外接管。

### 2.4 fallback 需要支持严格模式

Krea2 三种流程的本地 fallback 可能重新触发 OOM。远端不可用时应允许按 mode
直接失败，而不是无条件回退本地；默认建议 Krea2/edit/rebalance 使用 strict remote。

### 2.5 Mac 不能直接安装 Windows CUDA 项目的同一环境

当前 `pyproject.toml` 的 torch source 面向 CUDA/cu128。远端服务应使用独立 Mac
venv/uv 项目或独立依赖组，只共享协议和编码器代码，不能要求 Mac 安装 CUDA wheel。

## 3. 分阶段实施

### P0：Mac 本地可运行性验证（阻塞后续开发）

1. 在 Mac 安装源码版 ComfyUI、与本端匹配的 ComfyUI commit、
   `comfyui-krea2edit` 和 `ComfyUI-Conditioning-Rebalance`。
2. 配置 Qwen3-VL/Krea2 text encoder 的本地路径。
3. 不引入 gRPC，直接运行脚本完成一次纯文本和一次 grounded encode。
4. 对比本端结果的结构、shape、dtype、数值误差和耗时，确认 MPS/CPU 路径可用。

验收失败时，先解决 ComfyUI/节点/MPS 兼容性，不继续协议开发。

### P1：纯文本 EncodeText 远端化

Windows：

- 新增 remote encoder client、proto、递归 conditioning codec；
- 增加资源 ID/fingerprint 和远端配置；
- 接入图片模式纯文本路径、本地 LRU、重试、超时和 strict/fallback；
- 增加 fake server 和协议/错误处理测试。

Mac：

- 新增独立 remote encoder server；
- 实现 encoder ID 到本地路径映射、LoadClip、EncodeText、Release；
- 增加 Ping 和一个命令行 smoke test。

### P2：Grounded/Rebalance

- 支持 CPU 图片 tensor 上传和压缩；
- 复用同一递归 conditioning codec；
- 验证 edit-krea2 与 krea2-rebalance 端到端结果；
- 视频和 Caption 仍保持本地路径。

### P3：运维与文档

- Mac 使用 `launchd` 守护远端服务；
- 增加版本/ComfyUI commit 检查、健康状态和统计事件；
- 更新 `AGENTS/core-technical-reference.md`、`AGENTS/backend-integration.md`、
  `configs/workbench.example.yaml`；
- 增加 Mac 更新脚本和回滚说明。

## 4. Git/Gitea 同步方案

可以由 Windows 本端完成代码、协议、测试和脚本开发，提交到局域网 Gitea；Mac
执行 `git pull --ff-only` 后更新远端服务。

Git 只同步代码和配置模板，不同步模型权重、虚拟环境、SQLite、输出文件和本机
绝对路径。Mac 应保留未提交的 `configs/remote-encoder.yaml`，并通过脚本完成：

1. 拉取指定提交；
2. 检查 ComfyUI/custom node commit；
3. 检查模型 fingerprint；
4. 更新 Mac 专用依赖；
5. 重启 `launchd` 服务；
6. 执行 Ping/smoke test。

Gitea 和 gRPC 都只监听局域网；跨网段时使用 SSH 隧道或其他明确的访问控制。

## 5. 实施前不做的事情

- 不先升级本端唯一 ComfyUI 环境到 cu130；如需验证，复制环境后隔离试验。
- 不把官方 DMG 内部路径当作稳定服务 API。
- 不把模型文件放入 Git/Gitea。
- 不在第一版远端化 Wan、MiniMax H3 或 `describe_image`。
- 不把“远端故障自动本地回退”作为 Krea2 默认行为。

