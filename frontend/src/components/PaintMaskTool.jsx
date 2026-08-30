import { useEffect, useRef, useState } from "react";
import { computeStageSize } from "./ImageCropModal";

// 遮罩颜色:与原工作流 DrawMaskOnImage 的 "0,0,255" 一致。
// 绘制时半透明便于看到下方内容;保存合成时由 ImageEditModal 转为不透明纯蓝。
export const MASK_COLOR = "0, 0, 255";
const MASK_FILL = `rgba(${MASK_COLOR}, 0.55)`;
const BRUSH_MIN = 4;
const BRUSH_MAX = 80;

// 涂抹遮罩工具:在工作图上用蓝色画笔涂抹待重绘区域。
// 遮罩数据保存在 maskCanvasRef(offscreen canvas,工作图原始分辨率),
// 组件重挂载(切换工具)时从 offscreen 恢复、绘制结束时同步回去,遮罩不丢失。
export default function PaintMaskTool({ src, maskCanvasRef, idPrefix = "image-edit-paint" }) {
  const imgRef = useRef(null);
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  // 图片显示尺寸(等比缩放到容器内)
  const [stage, setStage] = useState(null);
  const [brushSize, setBrushSize] = useState(24);
  // brush = 画笔;eraser = 橡皮
  const [mode, setMode] = useState("brush");
  // 光标处的画笔大小预览圈
  const [cursor, setCursor] = useState(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef(null);

  // 图片加载后量宽定显示尺寸;初始化/恢复遮罩画布(像素尺寸 = 工作图原始分辨率)
  function handleLoad() {
    const img = imgRef.current;
    const availableWidth = containerRef.current?.clientWidth || 560;
    setStage(computeStageSize(img.naturalWidth, img.naturalHeight, availableWidth));
    const canvas = canvasRef.current;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const saved = maskCanvasRef.current;
    if (saved && saved.width === img.naturalWidth && saved.height === img.naturalHeight) {
      canvas.getContext("2d").drawImage(saved, 0, 0);
    } else {
      // 工作图已更换(裁剪/拓展后),旧遮罩坐标系失效,清空
      maskCanvasRef.current = null;
    }
  }

  // 显示坐标 → 遮罩画布(原图分辨率)坐标
  function toMaskPoint(event) {
    const canvas = canvasRef.current;
    const bounds = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - bounds.left) / bounds.width) * canvas.width,
      y: ((event.clientY - bounds.top) / bounds.height) * canvas.height,
    };
  }

  function strokeTo(point) {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const ctxScale = canvas.width / (stage?.w || canvas.width);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize * ctxScale;
    ctx.globalCompositeOperation = mode === "eraser" ? "destination-out" : "source-over";
    ctx.strokeStyle = MASK_FILL;
    ctx.beginPath();
    const from = lastPointRef.current ?? point;
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPointRef.current = point;
  }

  // 一笔结束后把显示画布同步到 offscreen(切工具不丢遮罩)
  function syncToOffscreen() {
    const canvas = canvasRef.current;
    if (!canvas?.width) return;
    let offscreen = maskCanvasRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      maskCanvasRef.current = offscreen;
    }
    offscreen.width = canvas.width;
    offscreen.height = canvas.height;
    offscreen.getContext("2d").drawImage(canvas, 0, 0);
  }

  function onPointerDown(event) {
    if (!stage) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drawingRef.current = true;
    lastPointRef.current = null;
    strokeTo(toMaskPoint(event));
  }

  function onPointerMove(event) {
    const canvas = canvasRef.current;
    const bounds = canvas.getBoundingClientRect();
    setCursor({
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    });
    if (drawingRef.current) strokeTo(toMaskPoint(event));
  }

  function onPointerUp() {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastPointRef.current = null;
    syncToOffscreen();
  }

  function clearMask() {
    const canvas = canvasRef.current;
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    maskCanvasRef.current = null;
  }

  // 组件卸载前同步一次,避免最后一笔丢失
  useEffect(() => {
    const saved = maskCanvasRef;
    return () => {
      const canvas = canvasRef.current;
      if (!canvas?.width) return;
      let offscreen = saved.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        saved.current = offscreen;
      }
      offscreen.width = canvas.width;
      offscreen.height = canvas.height;
      offscreen.getContext("2d").drawImage(canvas, 0, 0);
    };
  }, [maskCanvasRef]);

  return (
    <>
      <div className="paint-toolbar" id={`${idPrefix}-toolbar`}>
        <label className="paint-brush-size" htmlFor={`${idPrefix}-brush-size`}>
          画笔大小
          <input
            id={`${idPrefix}-brush-size`}
            type="range"
            min={BRUSH_MIN}
            max={BRUSH_MAX}
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
          />
          <span>{brushSize}px</span>
        </label>
        <button
          id={`${idPrefix}-mode-brush`}
          type="button"
          className={`chip${mode === "brush" ? " active" : ""}`}
          aria-pressed={mode === "brush"}
          onClick={() => setMode("brush")}
        >
          画笔
        </button>
        <button
          id={`${idPrefix}-mode-eraser`}
          type="button"
          className={`chip${mode === "eraser" ? " active" : ""}`}
          aria-pressed={mode === "eraser"}
          onClick={() => setMode("eraser")}
        >
          橡皮
        </button>
        <button
          id={`${idPrefix}-clear`}
          type="button"
          className="chip"
          onClick={clearMask}
        >
          清除
        </button>
      </div>
      <div ref={containerRef} className="crop-stage-wrap">
        <div
          id={`${idPrefix}-stage`}
          className="crop-stage paint-stage"
          style={stage ? { width: stage.w, height: stage.h } : undefined}
        >
          <img ref={imgRef} src={src} alt="待涂抹图片" draggable={false} onLoad={handleLoad} />
          <canvas
            ref={canvasRef}
            id={`${idPrefix}-canvas`}
            className="paint-mask-canvas"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => {
              setCursor(null);
              onPointerUp();
            }}
          />
          {cursor && stage && (
            <div
              className="paint-brush-cursor"
              style={{
                left: cursor.x,
                top: cursor.y,
                width: brushSize,
                height: brushSize,
              }}
            />
          )}
        </div>
      </div>
      <p className="form-hint">
        用蓝色画笔涂抹要重绘/移除的区域,保存时遮罩会以纯蓝合成到图片上;
        在 prompt 中描述对蓝色区域的期望(如「移除蓝色区域」)。
      </p>
    </>
  );
}
