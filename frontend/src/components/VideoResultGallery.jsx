import { useMemo } from "react";
import { api } from "../api/client";
import StatusChip from "./StatusChip";

// 视频结果:只展示最新一个 batch,每个任务一张卡片。
// 视频文件大、生成耗时久,<video> 不自动播放,由用户手动点播。
export default function VideoResultGallery({ jobs }) {
  const latestGroup = useMemo(() => {
    const finished = jobs
      .filter((job) => ["completed", "failed", "cancelled"].includes(job.status))
      .slice()
      .reverse();
    const byBatch = new Map();
    for (const job of finished) {
      const key = job.batch_id ?? job.id;
      if (!byBatch.has(key)) byBatch.set(key, []);
      byBatch.get(key).push(job);
    }
    const first = byBatch.entries().next().value;
    if (!first) return null;
    const [key, group] = first;
    return { key, isBatch: group[0].batch_id !== null, jobs: group.slice().reverse() };
  }, [jobs]);

  if (!latestGroup) {
    return <div id="video-empty-hint" className="empty-hint">本次生成的视频会显示在这里</div>;
  }

  return (
    <div id="video-result-gallery">
      <section className="batch-group" id={`video-batch-${latestGroup.key}`}>
        {latestGroup.isBatch && (
          <h3 className="batch-title">批次 {latestGroup.jobs.length} 个</h3>
        )}
        <div className="video-card-grid" id={`video-card-grid-${latestGroup.key}`}>
          {latestGroup.jobs.map((job) => (
            <VideoCard key={job.id} job={job} />
          ))}
        </div>
      </section>
    </div>
  );
}

function VideoCard({ job }) {
  // video_url 由后端在任务完成时给出(/api/v1/videos/{job_id}),缺省时按约定路径兜底
  const src = job.video_url ?? api.videoUrl(job.id);
  return (
    <div className="video-card" id={`video-card-${job.id}`}>
      {job.status === "completed" ? (
        <video
          src={src}
          controls
          preload="metadata"
          playsInline
          title={job.prompt}
        />
      ) : (
        <div className="video-card-placeholder">
          <StatusChip status={job.status} />
          {job.error && <div className="video-card-error">{job.error}</div>}
        </div>
      )}
      <div className="video-card-info">
        <div className="video-card-prompt" title={job.prompt}>
          <span className="video-card-type-badge">
            {job.generation_type === "r2v"
              ? "参考生视频"
              : job.generation_type === "i2v" ? "图生视频" : "文生视频"}
          </span>
          {job.prompt}
        </div>
        {job.input_image_url && (
          <div className="video-card-input-image">
            <img src={job.input_image_url} alt="输入图片" loading="lazy" />
          </div>
        )}
        {job.last_frame_image_url && (
          <div className="video-card-input-image">
            <img src={job.last_frame_image_url} alt="尾帧图片" loading="lazy" />
          </div>
        )}
        {(job.reference_image_urls ?? []).length > 0 && (
          <div className="video-card-input-image video-card-ref-images">
            {job.reference_image_urls.map((url) => (
              <img key={url} src={url} alt="参考图片" loading="lazy" />
            ))}
          </div>
        )}
        {(job.reference_video_urls ?? []).length > 0 && (
          <div className="video-card-params">参考视频 ×{job.reference_video_urls.length}</div>
        )}
        {(job.reference_audio_urls ?? []).length > 0 && (
          <div className="video-card-params">参考音频 ×{job.reference_audio_urls.length}</div>
        )}
        <div className="video-card-params">
          {job.width}×{job.height} · {job.duration_seconds}s · {job.fps}fps ·{" "}
          {job.steps} 步 · CFG {job.cfg}
          {job.shift != null ? ` · Shift ${job.shift}` : ""}
          {job.latent_multiplier != null ? ` · Latent ×${job.latent_multiplier}` : ""}
          {` · ${job.sampler}/${job.scheduler}`}
        </div>
        <div className="video-card-params">
          seed {job.seed}
          {job.model_name ? ` · ${job.model_name}` : ""}
          {job.elapsed_seconds != null ? ` · 耗时 ${Math.round(job.elapsed_seconds)}s` : ""}
        </div>
      </div>
    </div>
  );
}
