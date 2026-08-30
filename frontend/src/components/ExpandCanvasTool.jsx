import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { MASK_COLOR } from "./PaintMaskTool";

// 图片与画布边缘至少保留的重叠量(px, 显示坐标),防止图片被拖丢
const MIN_VISIBLE = 32;
const SCALE_MIN = 0.05;
const SCALE_MAX = 8;

// 图像拓展工具:按表单目标宽高新建纯蓝画布(蓝色 = 待拓展区域),
// 工作图作为画布上的一层,拖动移动、拖右下角手柄或滚轮等比缩放。
// 通过 ref 暴露 apply():按目标像素输出"蓝底 + 工作图"的合成 PNG 交给 onApply。
const ExpandCanvasTool = forwardRef(function ExpandCanvasTool(
  { src, targetWidth, targetHeight, idPrefix = "image-edit-expand", onApply, onReadyChange },
  ref,
) {
  const imgRef = useRef(null);
  const containerRef = useRef(null);
  const stageRef = useRef(null);
  // 画布显示尺寸(比例 = 目标宽高比)
  const [stage, setStage] = useState(null);
  // 图片层在显示坐标系下的摆放 {x, y, scale}
  const [placement, setPlacement] = useState(null);
  const [error, setError] = useState(null);
  // 进行中的拖动:{mode: "move" | "resize", startX, startY, placement}
  const dragRef = useRef(null);

  // 计算画布显示尺寸:量宽后按目标比例推导高度,过高时按视口高度反推
  function measureStage() {
    const availableWidth = containerRef.current?.clientWidth || 560;
    let w = Math.min(560, availableWidth);
    let h = (w * targetHeight) / targetWidth;
    const maxH = window.innerHeight * 0.5;
    if (h > maxH) {
      h = maxH;
      w = (h * targetWidth) / targetHeight;
    }
    return { w: Math.round(w), h: Math.round(h) };
  }

  // 图片加载后定画布尺寸,图片初始"适配画布(contain)"居中
  function handleLoad() {
    const img = imgRef.current;
    const nextStage = measureStage();
    setStage(nextStage);
    const scale = Math.min(
      nextStage.w / img.naturalWidth,
      nextStage.h / img.naturalHeight,
    );
    setPlacement({
      x: (nextStage.w - img.naturalWidth * scale) / 2,
      y: (nextStage.h - img.naturalHeight * scale) / 2,
      scale,
    });
  }

  const imgDispW = imgRef.current ? imgRef.current.naturalWidth * (placement?.scale ?? 1) : 0;
  const imgDispH = imgRef.current ? imgRef.current.naturalHeight * (placement?.scale ?? 1) : 0;

  function clampPlacement(next) {
    if (!stage) return next;
    const img = imgRef.current;
    const w = img.naturalWidth * next.scale;
    const h = img.naturalHeight * next.scale;
    return {
      ...next,
      x: Math.min(Math.max(next.x, MIN_VISIBLE - w), stage.w - MIN_VISIBLE),
      y: Math.min(Math.max(next.y, MIN_VISIBLE - h), stage.h - MIN_VISIBLE),
    };
  }

  function startDrag(event, mode) {
    if (!placement) return;
    event.preventDefault();
    event.stopPropagation();
    dragRef.current = { mode, startX: event.clientX, startY: event.clientY, placement };
  }

  useEffect(() => {
    function onPointerMove(event) {
      const drag = dragRef.current;
      if (!drag || !stage) return;
      const img = imgRef.current;
      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      if (drag.mode === "move") {
        setPlacement(
          clampPlacement({ ...drag.placement, x: drag.placement.x + dx, y: drag.placement.y + dy }),
        );
      } else {
        // 等比缩放:以横向位移推导新 scale,图片左上角不动
        const nextScale = Math.min(
          Math.max(drag.placement.scale + dx / img.naturalWidth, SCALE_MIN),
          SCALE_MAX,
        );
        setPlacement(clampPlacement({ ...drag.placement, scale: nextScale }));
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage]);

  // 滚轮缩放(以画布中心为锚);wheel 需要 passive:false 才能 preventDefault
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return undefined;
    function onWheel(event) {
      event.preventDefault();
      setPlacement((prev) => {
        if (!prev) return prev;
        const factor = event.deltaY < 0 ? 1.08 : 1 / 1.08;
        const next = Math.min(Math.max(prev.scale * factor, SCALE_MIN), SCALE_MAX);
        return clampPlacement({ ...prev, scale: next });
      });
    }
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage]);

  function zoomBy(factor) {
    setPlacement((prev) => {
      if (!prev) return prev;
      const next = Math.min(Math.max(prev.scale * factor, SCALE_MIN), SCALE_MAX);
      return clampPlacement({ ...prev, scale: next });
    });
  }

  // 把显示坐标换算到目标像素,合成"蓝底 + 工作图"PNG
  function apply() {
    const img = imgRef.current;
    if (!img || !stage || !placement) return;
    const canvas = document.createElement("canvas");
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = `rgb(${MASK_COLOR})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const k = targetWidth / stage.w;
    ctx.drawImage(
      img,
      placement.x * k,
      placement.y * k,
      img.naturalWidth * placement.scale * k,
      img.naturalHeight * placement.scale * k,
    );
    canvas.toBlob((blob) => {
      if (!blob) {
        setError("拓展画布合成失败,请重试");
        return;
      }
      onApply(new File([blob], "expanded-input.png", { type: "image/png" }));
    }, "image/png");
  }

  useImperativeHandle(ref, () => ({ apply }));

  const ready = Boolean(stage && placement);
  useEffect(() => {
    onReadyChange?.(ready);
  }, [ready, onReadyChange]);

  return (
    <>
      <div className="expand-toolbar" id={`${idPrefix}-toolbar`}>
        <button
          id={`${idPrefix}-zoom-out`}
          type="button"
          className="chip"
          disabled={!ready}
          onClick={() => zoomBy(1 / 1.2)}
        >
          缩小 −
        </button>
        <button
          id={`${idPrefix}-zoom-in`}
          type="button"
          className="chip"
          disabled={!ready}
          onClick={() => zoomBy(1.2)}
        >
          放大 +
        </button>
        <button
          id={`${idPrefix}-fit`}
          type="button"
          className="chip"
          disabled={!ready}
          onClick={handleLoad}
        >
          适配画布
        </button>
      </div>
      <div ref={containerRef} className="crop-stage-wrap">
        <div
          ref={stageRef}
          id={`${idPrefix}-stage`}
          className="expand-stage"
          style={stage ? { width: stage.w, height: stage.h } : undefined}
        >
          {/* 隐藏 img 仅用于加载/取原始尺寸;显示层由下面的定位 img 承担 */}
          <img
            ref={imgRef}
            src={src}
            alt=""
            draggable={false}
            onLoad={handleLoad}
            style={{ display: "none" }}
          />
          {stage && placement && (
            <>
              <img
                id={`${idPrefix}-image`}
                className="expand-stage-image"
                src={src}
                alt="待拓展图片"
                draggable={false}
                style={{
                  left: placement.x,
                  top: placement.y,
                  width: imgDispW,
                  height: imgDispH,
                }}
                onPointerDown={(event) => startDrag(event, "move")}
              />
              <div
                id={`${idPrefix}-handle`}
                className="crop-handle expand-handle"
                style={{ left: placement.x + imgDispW - 10, top: placement.y + imgDispH - 10 }}
                onPointerDown={(event) => startDrag(event, "resize")}
              />
            </>
          )}
        </div>
      </div>
      <p className="form-hint">
        蓝色区域为待拓展部分;拖动图片移动,拖图片右下角手柄或滚轮缩放。
        应用后按当前表单的宽度 × 高度({targetWidth} × {targetHeight})生成画布。
      </p>
      {error && <div className="form-error">{error}</div>}
    </>
  );
});

export default ExpandCanvasTool;
