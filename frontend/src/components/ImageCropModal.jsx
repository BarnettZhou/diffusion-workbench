import { useEffect, useRef, useState } from "react";
import Modal from "./Modal";

// 显示坐标系下裁剪框的最小边长(px),避免缩得太小无法操作
const MIN_RECT = 32;

// 固定比例图片裁剪弹窗:裁剪框宽高比锁定为 ratio(宽 / 高,取自表单的宽度/高度输入值),
// 拖动裁剪框移动、拖右下角手柄等比缩放;确认后用 canvas 输出 PNG 文件交给调用方上传。
export default function ImageCropModal({ src, ratio, onConfirm, onClose }) {
  const imgRef = useRef(null);
  // 图片在弹窗中的显示尺寸(等比缩放到容器内)
  const [stage, setStage] = useState(null);
  // 显示坐标系下的裁剪框 {x, y, w, h}
  const [rect, setRect] = useState(null);
  const [error, setError] = useState(null);
  // 进行中的拖动:{mode: "move" | "resize", startX, startY, rect}
  const dragRef = useRef(null);

  // 图片加载后计算显示尺寸,并初始化一个居中的最大比例裁剪框
  function handleLoad() {
    const img = imgRef.current;
    const scale = Math.min(
      560 / img.naturalWidth,
      (window.innerHeight * 0.55) / img.naturalHeight,
      1,
    );
    const w = Math.round(img.naturalWidth * scale);
    const h = Math.round(img.naturalHeight * scale);
    setStage({ w, h });
    let cw = w;
    let ch = w / ratio;
    if (ch > h) {
      ch = h;
      cw = h * ratio;
    }
    setRect({ x: (w - cw) / 2, y: (h - ch) / 2, w: cw, h: ch });
  }

  function startDrag(event, mode) {
    if (!rect) return;
    event.preventDefault();
    event.stopPropagation();
    dragRef.current = { mode, startX: event.clientX, startY: event.clientY, rect };
  }

  useEffect(() => {
    function onPointerMove(event) {
      const drag = dragRef.current;
      if (!drag || !stage) return;
      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      if (drag.mode === "move") {
        setRect({
          ...drag.rect,
          x: Math.min(Math.max(drag.rect.x + dx, 0), stage.w - drag.rect.w),
          y: Math.min(Math.max(drag.rect.y + dy, 0), stage.h - drag.rect.h),
        });
      } else {
        // 等比缩放:以横向位移为主,裁剪框不能越出图片边界
        let w = Math.min(Math.max(drag.rect.w + dx, MIN_RECT), stage.w - drag.rect.x);
        let h = w / ratio;
        if (drag.rect.y + h > stage.h) {
          h = stage.h - drag.rect.y;
          w = h * ratio;
        }
        setRect({ ...drag.rect, w, h });
      }
    }
    function onPointerUp() {
      dragRef.current = null;
    }
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
  }, [stage, ratio]);

  // 把显示坐标换算回原图像素,canvas 裁剪后输出 PNG File
  function confirmCrop() {
    const img = imgRef.current;
    if (!img || !stage || !rect) return;
    const scale = img.naturalWidth / stage.w;
    const sx = Math.round(rect.x * scale);
    const sy = Math.round(rect.y * scale);
    const sw = Math.max(1, Math.round(rect.w * scale));
    const sh = Math.max(1, Math.round(rect.h * scale));
    const canvas = document.createElement("canvas");
    canvas.width = sw;
    canvas.height = sh;
    canvas.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
    canvas.toBlob((blob) => {
      if (!blob) {
        setError("裁剪失败,请重试");
        return;
      }
      onConfirm(new File([blob], "cropped-input.png", { type: "image/png" }));
    }, "image/png");
  }

  return (
    <Modal
      id="image-crop-modal"
      title="裁剪输入图片"
      onClose={onClose}
      footer={
        <>
          <button
            id="image-crop-confirm-btn"
            type="button"
            className="primary"
            disabled={!rect}
            onClick={confirmCrop}
          >
            裁剪并上传
          </button>
          <button id="image-crop-cancel-btn" type="button" className="chip" onClick={onClose}>
            取消
          </button>
        </>
      }
    >
      <div
        id="image-crop-stage"
        className="crop-stage"
        style={stage ? { width: stage.w, height: stage.h } : undefined}
      >
        <img ref={imgRef} src={src} alt="待裁剪图片" draggable={false} onLoad={handleLoad} />
        {rect && (
          <div
            id="image-crop-rect"
            className="crop-rect"
            style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
            onPointerDown={(event) => startDrag(event, "move")}
          >
            <div
              id="image-crop-handle"
              className="crop-handle"
              onPointerDown={(event) => startDrag(event, "resize")}
            />
          </div>
        )}
      </div>
      <p className="form-hint">
        拖动裁剪框移动,拖右下角手柄缩放;比例已锁定为当前表单的宽度 × 高度。
      </p>
      {error && <div className="form-error">{error}</div>}
    </Modal>
  );
}
