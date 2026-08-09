const STATUS_LABELS = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

export default function StatusChip({ status }) {
  return (
    <span className={`chip status-${status}`}>{STATUS_LABELS[status] ?? status}</span>
  );
}
