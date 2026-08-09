import { useEffect } from "react";

// 公共右键菜单:fixed 定位在 (x, y),自动避让窗口右/下边缘;
// 点击任意处 / Esc / 滚动时关闭。items: [{id, label, danger?, onClick}]
export default function ContextMenu({ x, y, items, onClose }) {
  useEffect(() => {
    const onKey = (event) => event.key === "Escape" && onClose();
    window.addEventListener("click", onClose);
    window.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onClose, true);
    return () => {
      window.removeEventListener("click", onClose);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onClose, true);
    };
  }, [onClose]);

  return (
    <div
      className="context-menu"
      role="menu"
      style={{
        left: Math.min(x, window.innerWidth - 180),
        top: Math.min(y, window.innerHeight - 40 - items.length * 38),
      }}
    >
      {items.map((item) => (
        <button
          key={item.id}
          id={item.id}
          type="button"
          role="menuitem"
          className={item.danger ? "danger-item" : undefined}
          onClick={() => {
            onClose();
            item.onClick();
          }}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
