import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, eventsUrl } from "./api/client";
import AlbumPage from "./components/AlbumPage";
import ParameterForm from "./components/ParameterForm";
import ProgressPanel from "./components/ProgressPanel";
import ResultGallery from "./components/ResultGallery";
import SettingsPage from "./components/SettingsPage";

const MODE_LABELS = { zit: "ZIT", zib: "ZIB", krea2: "Krea2", sdxl: "SDXL" };

export default function App() {
  const [tab, setTab] = useState("workbench");
  const [resources, setResources] = useState(null);
  const [settings, setSettings] = useState(null);
  const [jobs, setJobs] = useState({});
  const [queueState, setQueueState] = useState({ queued: 0, running: null });
  const [previews, setPreviews] = useState({});
  const [wsConnected, setWsConnected] = useState(false);
  const [fatalError, setFatalError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  // 模式切换上移为工作台级全局状态,由 App 统一持有并下发给左侧表单
  const [mode, setMode] = useState("zit");
  // 相册"发送到工作台"带过来的表单预填参数
  const [prefill, setPrefill] = useState(null);
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  const updateJob = useCallback((id, patch) => {
    if (!id) return;
    setJobs((prev) => (prev[id] ? { ...prev, [id]: { ...prev[id], ...patch } } : prev));
  }, []);

  const fetchJob = useCallback(async (id) => {
    try {
      const job = await api.job(id);
      setJobs((prev) => ({ ...prev, [id]: job }));
    } catch {
      /* 任务可能已被清理 */
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
          if (event.running && !jobsRef.current[event.running]) fetchJob(event.running);
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
          updateJob(event.job_id, {
            status: event.status,
            image_url: event.image_url ?? null,
            upscaled_image_url: event.upscaled_image_url ?? null,
          });
          // 拉取最终持久化状态(seed、耗时等以 SQLite 为准)
          fetchJob(event.job_id);
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
  }, [updateJob, fetchJob]);

  const runningJob = queueState.running ? (jobs[queueState.running] ?? null) : null;
  const busy = queueState.running !== null || queueState.queued > 0;

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

  const orderedJobs = useMemo(
    () =>
      Object.values(jobs).sort(
        (a, b) => new Date(a.submitted_at) - new Date(b.submitted_at),
      ),
    [jobs],
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
      sampler: meta.sampler,
      scheduler: meta.scheduler,
      modelName: meta.model_name,
      vaeName: meta.vae_name,
    });
    setTab("workbench");
  }, []);

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
            工作台
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
      </header>
      {tab === "album" && <AlbumPage onSendToWorkbench={handleSendToWorkbench} />}
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
    </div>
  );
}
