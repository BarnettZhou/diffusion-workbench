import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import Modal from "./Modal";

// 显示坐标系下裁剪框的最小边长(px),避免缩得太小无法操作
const MIN_RECT = 32;

// 图片在弹窗中的显示尺寸:等比缩放到容器内,上限 maxWidth 与视口高度的 maxHeightRatio。
// 供 CropTool / PaintMaskTool 等量宽场景共用。
export function computeStageSize(naturalWidth, naturalHeight, availableWidth, maxWidth = 560, maxHeightRatio = 0.55) {
  const scale = Math.min(
    Math.min(maxWidth, availableWidth) / naturalWidth,
    (window.innerHeight * maxHeightRatio) / naturalHeight,
    1,
  );
  return {
    w: Math.round(naturalWidth * scale),
    h: Math.round(naturalHeight * scale),
    scale,
  };
}

// 固定/自由比例裁剪工具:ratio 为宽 / 高(取自表单的宽度/高度输入值),传 null 表示自由裁剪。
// 拖动裁剪框移动、拖右下角手柄缩放(锁定比例时等比,自由模式宽高各自跟随指针)。
// 通过 ref 暴露 confirm():确认后用 canvas 输出 PNG 文件交给 onConfirm;
// onReadyChange 在裁剪框就绪状态变化时回调(供弹窗禁用/启用确认按钮)。
// idPrefix 区分多个使用方的 DOM id(ImageCropModal / ImageEditModal)。
// initialFullRect:自由模式下初始裁剪框铺满整图(用于裁剪烘焙后的重新挂载,
// 此时整图就是上次选区,铺满才不会出现"框缩小居中"的跳变)。
export const CropTool = forwardRef(function CropTool(
  { src, ratio = null, idPrefix = "image-crop", initialFullRect = false, onConfirm, onReadyChange },
  ref,
) {
  const imgRef = useRef(null);
  // stage 外层容器:量出弹窗实际可用宽度,避免窄屏下按 560px 计算导致图片变形/裁剪框错位
  const containerRef = useRef(null);
  // 图片在弹窗中的显示尺寸(等比缩放到容器内)
  const [stage, setStage] = useState(null);
  // 显示坐标系下的裁剪框 {x, y, w, h}
  const [rect, setRect] = useState(null);
  const [error, setError] = useState(null);
  // 进行中的拖动:{mode: "move" | "resize", startX, startY, rect}
  const dragRef = useRef(null);

  // 图片加载后计算显示尺寸,并初始化裁剪框:锁定比例取最大适配矩形;
  // 自由模式默认 80% 居中,initialFullRect(烘焙后重挂载)时铺满整图
  function handleLoad() {
    const img = imgRef.current;
    const availableWidth = containerRef.current?.clientWidth;
    if (!availableWidth) {
      // 弹窗尚未完成布局(移动端刚打开/图片命中缓存)时量到 0:下一帧重试,
      // 避免按 560 兜底算出超出容器的 stage,导致裁剪框与实际渲染错位
      requestAnimationFrame(() => handleLoad());
      return;
    }
    const { w, h } = computeStageSize(img.naturalWidth, img.naturalHeight, availableWidth);
    setStage({ w, h });
    let cw;
    let ch;
    if (ratio) {
      cw = w;
      ch = w / ratio;
      if (ch > h) {
        ch = h;
        cw = h * ratio;
      }
    } else if (initialFullRect) {
      // 烘焙后的工作图:整图即上次选区,裁剪框铺满,视觉连续
      cw = w;
      ch = h;
    } else {
      cw = w * 0.8;
      ch = h * 0.8;
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
      } else if (ratio) {
        // 锁定比例:以横向位移为主等比缩放,裁剪框不能越出图片边界
        let w = Math.min(Math.max(drag.rect.w + dx, MIN_RECT), stage.w - drag.rect.x);
        let h = w / ratio;
        if (drag.rect.y + h > stage.h) {
          h = stage.h - drag.rect.y;
          w = h * ratio;
        }
        setRect({ ...drag.rect, w, h });
      } else {
        // 自由裁剪:宽高各自跟随指针,夹紧到最小边长与图片边界
        const w = Math.min(Math.max(drag.rect.w + dx, MIN_RECT), stage.w - drag.rect.x);
        const h = Math.min(Math.max(drag.rect.h + dy, MIN_RECT), stage.h - drag.rect.y);
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

  // 把显示坐标换算回原图像素,canvas 裁剪后输出 PNG File。
  // 比例以确认瞬间 img 的实际渲染尺寸为准(而不是 stage 状态):stage 状态可能与
  // 真实渲染不一致(如量宽时布局未就绪、max-width 钳制),用 getBoundingClientRect 自校正。
  function confirmCrop() {
    const img = imgRef.current;
    if (!img || !stage || !rect) return;
    const bounds = img.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const scale = img.naturalWidth / bounds.width;
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

  useImperativeHandle(ref, () => ({ confirm: confirmCrop }));

  const ready = Boolean(stage && rect);
  useEffect(() => {
    onReadyChange?.(ready);
  }, [ready, onReadyChange]);

  return (
    <>
      <div ref={containerRef} className="crop-stage-wrap">
        <div
          id={`${idPrefix}-stage`}
          className="crop-stage"
          style={stage ? { width: stage.w, height: stage.h } : undefined}
        >
          <img ref={imgRef} src={src} alt="待裁剪图片" draggable={false} onLoad={handleLoad} />
          {rect && (
            <div
              id={`${idPrefix}-rect`}
              className="crop-rect"
              style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
              onPointerDown={(event) => startDrag(event, "move")}
            >
              <div
                id={`${idPrefix}-handle`}
                className="crop-handle"
                onPointerDown={(event) => startDrag(event, "resize")}
              />
            </div>
          )}
        </div>
      </div>
      <p className="form-hint">
        {ratio
          ? "拖动裁剪框移动,拖右下角手柄缩放;比例已锁定为当前表单的宽度 × 高度。"
          : "自由裁剪:拖动裁剪框移动,拖右下角手柄分别调整宽和高。"}
      </p>
      {error && <div className="form-error">{error}</div>}
    </>
  );
});

// 固定比例图片裁剪弹窗:确认后用 canvas 输出 PNG 文件交给调用方上传。
export default function ImageCropModal({ src, ratio, onConfirm, onClose }) {
  const cropRef = useRef(null);
  const [ready, setReady] = useState(false);

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
            disabled={!ready}
            onClick={() => cropRef.current?.confirm()}
          >
            裁剪并上传
          </button>
          <button id="image-crop-cancel-btn" type="button" className="chip" onClick={onClose}>
            取消
          </button>
        </>
      }
    >
      <CropTool
        ref={cropRef}
        src={src}
        ratio={ratio}
        onConfirm={onConfirm}
        onReadyChange={setReady}
      />
    </Modal>
  );
}
