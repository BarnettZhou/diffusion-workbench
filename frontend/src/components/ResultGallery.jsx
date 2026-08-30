import { useMemo, useState } from "react";
import { api } from "../api/client";
import Lightbox from "./Lightbox";
import StatusChip from "./StatusChip";

// 只展示最新一个 batch 的结果:新 batch 生成出图片后,整体替换掉上一批。
// 每组:主图(默认最新一张)+ 按时间正序的缩略图;主图可点开全屏预览。
// onSendToVideo/onSendToEdit/onSendToCaption 由 App 注入(接收 job,负责导入+切 tab)。
// 历史相册见 AlbumPage。
export default function ResultGallery({ jobs, onSendToVideo = null, onSendToEdit = null, onSendToCaption = null }) {
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
    return <div id="empty-hint" className="empty-hint">本次生成的图片会显示在这里</div>;
  }

  return (
    <div id="result-gallery">
      <BatchGroup
        key={latestGroup.key}
        group={latestGroup}
        onSendToVideo={onSendToVideo}
        onSendToEdit={onSendToEdit}
        onSendToCaption={onSendToCaption}
      />
    </div>
  );
}

function BatchGroup({ group, onSendToVideo = null, onSendToEdit = null, onSendToCaption = null }) {
  const [selectedId, setSelectedId] = useState(null);
  const [lightbox, setLightbox] = useState(false);
  // 原图 / 放大后 查看切换;默认展示放大后(有放大图时)
  const [viewMode, setViewMode] = useState("upscaled");

  const completed = group.jobs.filter((job) => job.status === "completed");
  const selected =
    completed.find((job) => job.id === selectedId) ??
    completed[completed.length - 1] ??
    null;
  // 放大图可用时按切换项展示;不可用(未启用放大或放大失败)回退原图
  const hasUpscaled = Boolean(selected?.upscaled_image_url);
  const displayImageUrl = selected
    ? viewMode === "upscaled" && selected.upscaled_image_url
      ? selected.upscaled_image_url
      : selected.image_url
    : null;
  const displayMetadataUrl = selected
    ? viewMode === "upscaled" && selected.upscaled_image_url
      ? `/api/v1/images/${selected.id}/upscaled/metadata`
      : `/api/v1/images/${selected.id}/metadata`
    : null;


  return (
    <section className="batch-group" id={`batch-${group.key}`}>
      {group.isBatch && <h3 className="batch-title">批次 {group.jobs.length} 张</h3>}

      <div className="batch-main" id={`batch-main-${group.key}`}>
        {selected && (
          <>
            {hasUpscaled && (
              <div className="view-switch" role="group" aria-label="图片查看">
                <button
                  type="button"
                  className={viewMode === "original" ? "active" : ""}
                  aria-pressed={viewMode === "original"}
                  onClick={() => setViewMode("original")}
                >
                  原图
                </button>
                <button
                  type="button"
                  className={viewMode === "upscaled" ? "active" : ""}
                  aria-pressed={viewMode === "upscaled"}
                  onClick={() => setViewMode("upscaled")}
                >
                  放大后
                </button>
              </div>
            )}
            <img
              id={`main-image-${group.key}`}
              src={displayImageUrl}
              alt={selected.prompt}
              onClick={() => setLightbox(true)}
            />
          </>
        )}
        {!selected && (
          <div className="batch-main-placeholder">
            <StatusChip status={group.jobs[group.jobs.length - 1].status} />
          </div>
        )}
      </div>

      {group.jobs.length > 1 && (
        <div className="thumb-row" id={`thumb-row-${group.key}`}>
          {group.jobs.map((job) => (
            <button
              key={job.id}
              id={`thumb-${job.id}`}
              type="button"
              className={`thumb ${selected?.id === job.id ? "active" : ""}`}
              disabled={job.status !== "completed"}
              onClick={() => setSelectedId(job.id)}
              title={job.prompt}
            >
              {job.status === "completed" ? (
                <img src={api.imageUrl(job.id)} alt="" loading="lazy" />
              ) : (
                <StatusChip status={job.status} />
              )}
            </button>
          ))}
        </div>
      )}

      {lightbox && selected && (
        <Lightbox
          imageUrl={displayImageUrl}
          metadataUrl={displayMetadataUrl}
          fallback={selected}
          onClose={() => setLightbox(false)}
          onSendToVideo={
            // 成功(父级已切 tab)后关闭弹窗;失败保持打开,由 Lightbox 提示错误
            onSendToVideo
              ? async () => {
                  await onSendToVideo(selected);
                  setLightbox(false);
                }
              : null
          }
          onSendToEdit={
            onSendToEdit
              ? async (slot) => {
                  await onSendToEdit(selected, slot);
                  setLightbox(false);
                }
              : null
          }
          onSendToCaption={
            onSendToCaption
              ? async () => {
                  await onSendToCaption(selected);
                  setLightbox(false);
                }
              : null
          }
        />
      )}
    </section>
  );
}
