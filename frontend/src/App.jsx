import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, eventsUrl } from "./api/client";
import AlbumPage from "./components/AlbumPage";
import ParameterForm from "./components/ParameterForm";
import ProgressPanel from "./components/ProgressPanel";
import ResultGallery from "./components/ResultGallery";
import SettingsPage from "./components/SettingsPage";
import { useMessage } from "./components/Message";
import VideoParameterForm from "./components/VideoParameterForm";
import VideoResultGallery from "./components/VideoResultGallery";
import Modal from "./components/Modal";

const MODE_LABELS = { zit: "ZIT", zib: "ZIB", krea2: "Krea2", sdxl: "SDXL" };
// 视频模型分类显示名(设置页「视频模型」tab 同款映射)
const VIDEO_MODEL_LABELS = {
  "wan2.2-ti2v-5b": "Wan 2.2 TI2V-5B",
  "wan2.2-i2v-14b": "Wan 2.2 I2V-14B",
  "minimax-h3": "MiniMax H3",
};

export default function App() {
  const [tab, setTab] = useState("workbench");
  const [resources, setResources] = useState(null);
  const [settings, setSettings] = useState(null);
  const [jobs, setJobs] = useState({});
  // 视频任务与图片任务分开存放;事件按 job_id 命中哪张表就更新哪张
  const [videoJobs, setVideoJobs] = useState({});
  // 视频模型分类列表与资源({video_model: {models, vaes}})
  const [videoModels, setVideoModels] = useState([]);
  const [videoModelInfo, setVideoModelInfo] = useState({});
  const [videoResources, setVideoResources] = useState({});
  const [videoModel, setVideoModel] = useState(null);
  const [videoSubmitting, setVideoSubmitting] = useState(false);
  const [queueState, setQueueState] = useState({ queued: 0, running: null });
  const [previews, setPreviews] = useState({});
  const [wsConnected, setWsConnected] = useState(false);
  // tabs 右侧资源占用面板:轮询 /status 刷新
  const [resourceStatus, setResourceStatus] = useState(null);
  const [releasing, setReleasing] = useState(false);
  const [resourceModalOpen, setResourceModalOpen] = useState(false);
  const [fatalError, setFatalError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  // 模式切换上移为工作台级全局状态,由 App 统一持有并下发给左侧表单
  const [mode, setMode] = useState("zit");
  // 相册"发送到工作台"带过来的表单预填参数
  const [prefill, setPrefill] = useState(null);
  // 相册"发送到视频生成"带过来的 I2V 输入图片预填
  const [videoPrefill, setVideoPrefill] = useState(null);
  // 相册视频抽屉"发送到工作台"带过来的视频参数预填
  const [videoParamsPrefill, setVideoParamsPrefill] = useState(null);
  const message = useMessage();
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  const videoJobsRef = useRef(videoJobs);
  videoJobsRef.current = videoJobs;

  const updateJob = useCallback((id, patch) => {
    if (!id) return;
    setJobs((prev) => (prev[id] ? { ...prev, [id]: { ...prev[id], ...patch } } : prev));
    setVideoJobs((prev) => (prev[id] ? { ...prev, [id]: { ...prev[id], ...patch } } : prev));
  }, []);

  const fetchJob = useCallback(async (id) => {
    try {
      const job = await api.job(id);
      setJobs((prev) => ({ ...prev, [id]: job }));
    } catch {
      /* 任务可能已被清理 */
    }
  }, []);

  const fetchVideoJob = useCallback(async (id) => {
    try {
      const job = await api.videoJob(id);
      setVideoJobs((prev) => ({ ...prev, [id]: job }));
    } catch {
      /* 任务可能已被清理 */
    }
  }, []);

  // 队列里出现未知 running id 时,先按图片任务拉取,失败再按视频任务拉取
  const fetchAnyJob = useCallback(async (id) => {
    try {
      const job = await api.job(id);
      setJobs((prev) => ({ ...prev, [id]: job }));
    } catch {
      try {
        const job = await api.videoJob(id);
        setVideoJobs((prev) => ({ ...prev, [id]: job }));
      } catch {
        /* 任务可能已被清理 */
      }
    }
  }, []);

  const refreshResources = useCallback(async () => {
    const modeResponse = await api.modes();
    const modes = modeResponse.modes;
    const entries = await Promise.all(
      modes.flatMap((item) => [
        api.resources(item.mode, "diffusion"),
        api.resources(item.mode, "vae"),
      ]),
    );
    // 按 mode 动态组装,避免 entries 位置与 mode 对应关系随请求数量变化而错位
    setResources(
      Object.fromEntries(
        modes.map((item, i) => [
          item.mode,
          {
            models: entries[i * 2].resources,
            vaes: entries[i * 2 + 1].resources,
            modelLoader: item.model_loader,
          },
        ]),
      ),
    );
  }, []);

  // 初始快照:资源列表 + 队列状态。不加载历史任务——右侧只展示本次
  // 会话的任务队列与结果,历史相册后续单独做页面。
  useEffect(() => {
    (async () => {
      try {
        const [status, loadedSettings] = await Promise.all([
          api.status(),
          api.settings(),
          refreshResources(),
        ]);
        setSettings(loadedSettings);
        setQueueState({ queued: status.queue, running: status.running });
      } catch (err) {
        setFatalError(`后端连接失败:${err.message}。请确认 API 服务已启动(端口 8188)。`);
      }
    })();
  }, [refreshResources]);

  // 视频模型分类与资源独立加载:视频端点不可用时不影响图片工作台
  useEffect(() => {
    (async () => {
      try {
        const body = await api.videoModels();
        const items = body.video_models ?? [];
        const list = items.map((item) => item.video_model);
        setVideoModels(list);
        setVideoModelInfo(Object.fromEntries(items.map((item) => [item.video_model, item])));
        setVideoModel((prev) => (prev && list.includes(prev) ? prev : (list[0] ?? null)));
        const entries = await Promise.all(list.map((key) => api.videoResources(key)));
        setVideoResources(
          Object.fromEntries(
            list.map((key, i) => [key, { models: entries[i].models, vaes: entries[i].vaes }]),
          ),
        );
      } catch {
        /* 视频功能不可用时保持空列表 */
      }
    })();
  }, []);

  // 资源加载后若当前模式不可用(配置变化),回退到第一个可用模式
  useEffect(() => {
    const modes = Object.keys(resources ?? {});
    if (modes.length && !modes.includes(mode)) setMode(modes[0]);
  }, [resources, mode]);

  // WebSocket 实时事件(断线自动重连;事实以 REST 快照为准)
  useEffect(() => {
    let ws = null;
    let closed = false;
    let retryTimer = null;

    function handleEvent(event) {
      switch (event.type) {
        case "queue_progress":
          setQueueState({ queued: event.queued, running: event.running });
          if (
            event.running &&
            !jobsRef.current[event.running] &&
            !videoJobsRef.current[event.running]
          ) {
            fetchAnyJob(event.running);
          }
          break;
        case "job_started":
          updateJob(event.job_id, { status: "running", seed: event.seed });
          break;
        case "stage_progress":
          updateJob(event.job_id, { stage: event.stage });
          break;
        case "step_progress":
          updateJob(event.job_id, {
            step: event.step,
            total: event.total,
            secondsPerStep: event.seconds_per_step,
            stepsPerSecond: event.steps_per_second,
            etaSeconds: event.eta_seconds,
          });
          break;
        case "preview_image":
          setPreviews((prev) => ({ ...prev, [event.job_id]: event }));
          break;
        case "job_error":
          if (event.job_id) updateJob(event.job_id, { error: event.error });
          break;
        case "job_finished":
          if (event.artifact_type === "video") {
            updateJob(event.job_id, {
              status: event.status,
              video_url: event.video_url ?? null,
            });
            fetchVideoJob(event.job_id);
          } else {
            updateJob(event.job_id, {
              status: event.status,
              image_url: event.image_url ?? null,
              upscaled_image_url: event.upscaled_image_url ?? null,
            });
            // 拉取最终持久化状态(seed、耗时等以 SQLite 为准)
            fetchJob(event.job_id);
          }
          break;
      }
    }

    function connect() {
      ws = new WebSocket(eventsUrl());
      ws.onopen = () => setWsConnected(true);
      ws.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
      ws.onclose = () => {
        setWsConnected(false);
        if (!closed) retryTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => {
      closed = true;
      clearTimeout(retryTimer);
      ws?.close();
    };
  }, [updateJob, fetchJob, fetchVideoJob, fetchAnyJob]);

  const runningJob = queueState.running ? (jobs[queueState.running] ?? null) : null;
  const videoRunningJob = queueState.running
    ? (videoJobs[queueState.running] ?? null)
    : null;
  const busy = queueState.running !== null || queueState.queued > 0;

  // 资源占用轮询:5 秒一次,后端不可达时保留旧值
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const status = await api.status();
        if (!cancelled) setResourceStatus(status);
      } catch {
        /* 后端不可达时保留旧值 */
      }
    }
    poll();
    const timer = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // 清除资源占用:后端在任务/队列非空时返回 409
  async function handleReleaseResources() {
    if (releasing) return;
    setReleasing(true);
    try {
      await api.releaseResources();
      message.success("已清除资源占用");
      setResourceStatus(await api.status());
    } catch (err) {
      message.error(`清除资源失败:${err.message}`);
    } finally {
      setReleasing(false);
    }
  }

  const updateSettings = useCallback(async (patch) => {
    const updated = await api.updateSettings(patch);
    setSettings(updated);
  }, []);

  async function handleSubmit(payload) {
    setSubmitting(true);
    try {
      const result = await api.submit(payload);
      setJobs((prev) => ({
        ...prev,
        ...Object.fromEntries(result.jobs.map((job) => [job.id, job])),
      }));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleVideoSubmit(payload) {
    setVideoSubmitting(true);
    try {
      const result = await api.submitVideo(payload);
      setVideoJobs((prev) => ({
        ...prev,
        ...Object.fromEntries(result.jobs.map((job) => [job.id, job])),
      }));
    } finally {
      setVideoSubmitting(false);
    }
  }

  const orderedJobs = useMemo(
    () =>
      Object.values(jobs).sort(
        (a, b) => new Date(a.submitted_at) - new Date(b.submitted_at),
      ),
    [jobs],
  );

  const orderedVideoJobs = useMemo(
    () =>
      Object.values(videoJobs).sort(
        (a, b) => new Date(a.submitted_at) - new Date(b.submitted_at),
      ),
    [videoJobs],
  );

  // 相册抽屉"发送到工作台":切到工作台 tab,把图片的生成参数预填进表单
  const handleSendToWorkbench = useCallback((meta) => {
    setPrefill({
      mode: meta.mode,
      prompt: meta.prompt ?? "",
      negativePrompt: meta.negative_prompt ?? "",
      width: meta.width,
      height: meta.height,
      steps: meta.steps,
      cfg: meta.cfg,
      seed: meta.seed,
      sampler: meta.sampler,
      scheduler: meta.scheduler,
      modelName: meta.model_name,
      vaeName: meta.vae_name,
    });
    setTab("workbench");
  }, []);

  // 相册"发送到视频生成":切到视频生成 tab,把导入好的输入图片预填进表单
  const handleSendToVideo = useCallback((inputImage) => {
    setVideoPrefill(inputImage);
    setTab("video");
  }, []);

  // 相册视频抽屉"发送到工作台":切到视频生成 tab,把视频的生成参数预填进表单
  const handleSendVideoMeta = useCallback((meta) => {
    setVideoParamsPrefill(meta);
    setTab("video");
  }, []);

  // 资源占用展示:RAM(系统内存)、GPU(核心利用率)、VRAM(显存容量),探测未完成时显示 -
  const memInfo = resourceStatus?.memory ?? null;
  const ramValue = memInfo
    ? `${memInfo.used_gib.toFixed(1)}/${memInfo.total_gib.toFixed(1)}G`
    : "-";
  const gpuValue =
    resourceStatus?.gpu_utilization_percent != null
      ? `${Math.round(resourceStatus.gpu_utilization_percent)}%`
      : "-";
  const vramValue =
    resourceStatus?.gpu_memory_used_gib != null
      ? `${resourceStatus.gpu_memory_used_gib.toFixed(1)}/${(resourceStatus.gpu_memory_total_gib ?? 0).toFixed(1)}G`
      : "-";
  const ramText = `RAM ${ramValue}`;
  const gpuText = `GPU ${gpuValue}`;
  const vramText = `VRAM ${vramValue}`;

  return (
    <div id="app-shell">
      <header id="app-header">
        <nav id="app-tabs">
          <button
            id="tab-workbench"
            type="button"
            className={tab === "workbench" ? "active" : ""}
            onClick={() => setTab("workbench")}
          >
            图片生成
          </button>
          <button
            id="tab-video"
            type="button"
            className={tab === "video" ? "active" : ""}
            onClick={() => setTab("video")}
          >
            视频生成
          </button>
          <button
            id="tab-album"
            type="button"
            className={tab === "album" ? "active" : ""}
            onClick={() => setTab("album")}
          >
            相册
          </button>
          <button
            id="tab-settings"
            type="button"
            className={tab === "settings" ? "active" : ""}
            onClick={() => setTab("settings")}
          >
            设置
          </button>
        </nav>
        <span id="ws-status" className={`ws-badge ${wsConnected ? "online" : "offline"}`}>
          {wsConnected ? "实时已连接" : "实时断开"}
        </span>
        <div id="resource-panel">
          <span
            id="resource-usage"
            title={
              resourceStatus?.loaded_resources
                ? `模型 ${resourceStatus.loaded_resources.model ?? "-"} · VAE ${resourceStatus.loaded_resources.vae ?? "-"}`
                : "Worker 未加载模型资源"
            }
          >
            {resourceStatus?.loaded_resources
              ? `${resourceStatus.loaded_resources.workload ?? "?"} · ${resourceStatus.loaded_resources.model ?? "已加载"}`
              : "资源空闲"}
            {` · ${ramText} · ${gpuText} · ${vramText}`}
          </span>
          <button
            id="release-resources-btn"
            type="button"
            disabled={busy || releasing}
            title={busy ? "任务运行期间不能清除资源" : "释放 Worker 已加载的模型/VAE/text encoder"}
            onClick={handleReleaseResources}
          >
            {releasing ? "清除中……" : "清除资源"}
          </button>
        </div>
        <button
          id="resource-panel-toggle"
          type="button"
          title="查看资源占用"
          onClick={() => setResourceModalOpen(true)}
        >
          资源
        </button>
      </header>
      {/* 相册与工作台一样始终挂载,切 tab 仅隐藏,保留目录/子目录与滚动位置 */}
      <div
        id="album-tab-wrapper"
        className={tab === "album" ? "tab-contents" : "tab-hidden"}
      >
        <AlbumPage
          onSendToWorkbench={handleSendToWorkbench}
          onSendToVideo={handleSendToVideo}
          onSendVideoMeta={handleSendVideoMeta}
        />
      </div>
      {tab === "settings" && (
          <SettingsPage
            settings={settings}
            modes={Object.keys(resources ?? {})}
            onUpdate={updateSettings}
          onResourcesChanged={refreshResources}
        />
      )}
      {/* 工作台始终挂载,切 tab 仅隐藏,保住表单与结果现场 */}
      <main id="app-main" className={tab === "workbench" ? undefined : "tab-hidden"}>
        {/* 模式切换上移为工作台级全局开关,作用于左侧表单与提交参数 */}
        <div id="mode-bar">
          <div id="mode-switch" className="mode-switch" role="tablist" aria-label="模式">
            {Object.keys(resources ?? {}).map((key) => (
              <button
                key={key}
                id={`mode-${key}`}
                type="button"
                role="tab"
                aria-selected={mode === key}
                className={mode === key ? "active" : ""}
                onClick={() => setMode(key)}
              >
                {MODE_LABELS[key] ?? key}
              </button>
            ))}
          </div>
        </div>
        <div id="workbench-body">
        <aside id="controls-panel" className="controls">
          {fatalError && <div id="fatal-error" className="panel form-error">{fatalError}</div>}
          <ParameterForm
            mode={mode}
            onModeChange={setMode}
            resources={resources}
            sizePresets={settings?.size_presets ?? []}
            samplingDefaults={settings?.sampling_defaults}
            promptPresets={settings?.prompt_presets ?? []}
            prefill={prefill}
            onSubmit={handleSubmit}
          />
        </aside>
        <section id="results-panel" className="results">
          {/* 生成/跳过/停止放在结果区顶部;提交按钮通过 form 属性关联左侧表单 */}
          <div id="action-bar">
            <button
              id="submit-btn"
              type="submit"
              form="parameter-form"
              className="primary"
              disabled={submitting}
            >
              {submitting ? "提交中……" : "生成"}
            </button>
            <button
              id="skip-btn"
              type="button"
              className="warn"
              onClick={() => api.skip()}
              disabled={!busy}
            >
              跳过
            </button>
            <button
              id="stop-btn"
              type="button"
              className="danger"
              onClick={() => api.stop()}
              disabled={!busy}
            >
              停止
            </button>
          </div>
          <ProgressPanel
            job={runningJob}
            preview={runningJob ? previews[runningJob.id] : null}
            queuedCount={queueState.queued}
          />
          <ResultGallery jobs={orderedJobs} />
        </section>
        </div>
      </main>
      {/* 视频生成页同样始终挂载,切 tab 仅隐藏,保住表单与结果现场 */}
      <main id="video-main" className={tab === "video" ? undefined : "tab-hidden"}>
        <div id="video-model-bar">
          <div id="video-model-switch" className="mode-switch" role="tablist" aria-label="视频模型">
            {videoModels.map((key) => (
              <button
                key={key}
                id={`video-model-${key}`}
                type="button"
                role="tab"
                aria-selected={videoModel === key}
                className={videoModel === key ? "active" : ""}
                onClick={() => setVideoModel(key)}
              >
                {videoModelInfo[key]?.label ?? VIDEO_MODEL_LABELS[key] ?? key}
              </button>
            ))}
          </div>
        </div>
        <div id="video-body">
        <aside id="video-controls-panel" className="controls">
          {videoModel && (
            <VideoParameterForm
              videoModel={videoModel}
              modelInfo={videoModelInfo[videoModel]}
              resources={videoResources[videoModel]}
              inputImagePrefill={videoPrefill}
              paramsPrefill={videoParamsPrefill}
              promptPresets={settings?.prompt_presets ?? []}
              sizePresets={videoModel?.startsWith("minimax")
                ? (settings?.minimax_video_size_presets ?? [])
                : (settings?.wan_video_size_presets ?? [])}
              onSubmit={handleVideoSubmit}
            />
          )}
        </aside>
        <section id="video-results-panel" className="results">
          {/* 生成/跳过/停止放在结果区顶部;提交按钮通过 form 属性关联左侧表单 */}
          <div id="video-action-bar">
            <button
              id="video-submit-btn"
              type="submit"
              form="video-parameter-form"
              className="primary"
              disabled={videoSubmitting || !videoModel}
            >
              {videoSubmitting ? "提交中……" : "生成"}
            </button>
            <button
              id="video-skip-btn"
              type="button"
              className="warn"
              onClick={() => api.skip()}
              disabled={!busy}
            >
              跳过
            </button>
            <button
              id="video-stop-btn"
              type="button"
              className="danger"
              onClick={() => api.stop()}
              disabled={!busy}
            >
              停止
            </button>
          </div>
          {/* 视频任务没有 latent 预览,preview 固定传 null */}
          <ProgressPanel
            job={videoRunningJob}
            preview={null}
            queuedCount={queueState.queued}
          />
          <VideoResultGallery jobs={orderedVideoJobs} />
        </section>
        </div>
      </main>
      {resourceModalOpen && (
        <Modal
          id="resource-modal"
          title="资源占用"
          onClose={() => setResourceModalOpen(false)}
          footer={
            <button
              id="resource-modal-release-btn"
              type="button"
              disabled={busy || releasing}
              title={busy ? "任务运行期间不能清除资源" : "释放 Worker 已加载的模型/VAE/text encoder"}
              onClick={handleReleaseResources}
            >
              {releasing ? "清除中……" : "清除资源"}
            </button>
          }
        >
          <div className="resource-modal-rows">
            <div className="resource-modal-row">
              <span>工作负载</span>
              <span>
                {resourceStatus?.loaded_resources
                  ? `${resourceStatus.loaded_resources.workload ?? "?"} · ${resourceStatus.loaded_resources.model ?? "已加载"}`
                  : "资源空闲"}
              </span>
            </div>
            {resourceStatus?.loaded_resources?.vae && (
              <div className="resource-modal-row">
                <span>VAE</span>
                <span>{resourceStatus.loaded_resources.vae}</span>
              </div>
            )}
            <div className="resource-modal-row">
              <span>RAM</span>
              <span>{ramValue}</span>
            </div>
            <div className="resource-modal-row">
              <span>GPU</span>
              <span>{gpuValue}</span>
            </div>
            <div className="resource-modal-row">
              <span>VRAM</span>
              <span>{vramValue}</span>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
