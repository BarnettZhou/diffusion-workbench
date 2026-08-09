const STAGE_LABELS = {
  starting_worker: "启动推理 Worker",
  loading_model: "加载模型资源",
  prompt: "编码提示词",
  latent: "准备 latent",
  sampling: "采样中",
  vae: "VAE 解码",
  saving: "保存图片",
  saved: "图片已保存",
};

function formatEta(seconds) {
  if (seconds === null || seconds === undefined) return null;
  if (seconds < 60) return `约 ${Math.ceil(seconds)} 秒`;
  return `约 ${Math.floor(seconds / 60)} 分 ${Math.ceil(seconds % 60)} 秒`;
}

export default function ProgressPanel({ job, preview, queuedCount }) {
  if (!job) {
    return (
      <div id="progress-panel" className="panel progress-panel">
        <div id="progress-idle" className="progress-idle">
          空闲{queuedCount > 0 ? ` · 队列等待 ${queuedCount}` : " · 等待提交任务"}
        </div>
      </div>
    );
  }

  const stageLabel = STAGE_LABELS[job.stage] ?? job.status;
  const sampling = job.stage === "sampling" && job.total;
  const percent = sampling
    ? Math.round((job.step / job.total) * 100)
    : job.stage === "saved"
      ? 100
      : 5;
  const eta = formatEta(job.etaSeconds);

  return (
    <div id="progress-panel" className="panel progress-panel">
      <div id="progress-header" className="progress-header">
        <span id="progress-stage" className="stage-label">
          {stageLabel}
          {sampling && ` ${job.step}/${job.total}`}
        </span>
        <span id="progress-queue-count" className="queue-label">队列等待 {queuedCount}</span>
      </div>
      <div id="progress-body" className="progress-body">
        <div id="progress-main" className="progress-main">
          <div id="progress-bar" className="progress-bar">
            <div id="progress-bar-fill" style={{ width: `${percent}%` }} />
          </div>
          <div id="progress-sub" className="progress-sub">
            {job.prompt} · {job.width}×{job.height} · seed {job.seed}
          </div>
          {sampling && (
            <div id="progress-metrics" className="progress-metrics">
              {job.secondsPerStep != null && `${job.secondsPerStep.toFixed(2)} 秒/步`}
              {job.stepsPerSecond != null && ` · ${job.stepsPerSecond.toFixed(2)} 步/秒`}
              {eta && ` · 剩余 ${eta}`}
            </div>
          )}
        </div>
        {preview && (
          <img
            id="progress-preview"
            className="progress-preview"
            src={`data:${preview.mime_type};base64,${preview.data}`}
            alt={`采样预览 ${preview.step}/${preview.total}`}
          />
        )}
      </div>
    </div>
  );
}
