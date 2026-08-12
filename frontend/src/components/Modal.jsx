import { useEffect } from "react";

// 公共模态框:遮罩 + 居中对话框。Esc / 点击遮罩关闭(maskClosable=false 时
// 遮罩点击不关闭;closeDisabled 时全部关闭方式禁用,用于提交中等不可打断状态)。
// id 同时作为对话框 id 和 CSS 尺寸定制钩子。
// titleExtra 渲染在标题行右侧(如新建会话按钮)。
export default function Modal({
  id,
  title,
  titleExtra = null,
  onClose,
  closeDisabled = false,
  maskClosable = true,
  children,
  footer = null,
}) {
  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape" && !closeDisabled) onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, closeDisabled]);

  return (
    <div
      className="modal-overlay"
      id={id ? `${id}-overlay` : undefined}
      onClick={() => !closeDisabled && maskClosable && onClose()}
    >
      <div
        className="modal"
        id={id}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
      >
        {title && (
          <div className="modal-head">
            <h3 className="modal-title">{title}</h3>
            {titleExtra}
          </div>
        )}
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-footer">{footer}</footer>}
      </div>
    </div>
  );
}
