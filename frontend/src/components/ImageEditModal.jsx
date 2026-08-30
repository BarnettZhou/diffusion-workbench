import { useEffect, useRef, useState } from "react";
import Modal from "./Modal";
import { CropTool } from "./ImageCropModal";
import PaintMaskTool, { MASK_COLOR } from "./PaintMaskTool";
import ExpandCanvasTool from "./ExpandCanvasTool";

// 图片编辑弹窗的工具列表;后续新工具按 key 追加
const TOOLS = [
  { key: "crop", label: "裁剪" },
  { key: "paint", label: "涂抹遮罩" },
  { key: "expand", label: "拓展" },
];

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("图片加载失败,请重试"));
    img.src = src;
  });
}

// 图片编辑弹窗:顶部横向排列工具(裁剪/涂抹遮罩/拓展)。
// 内部维护"工作图":进入时为原图;裁剪、拓展的结果会烘焙为新的工作图(不关闭弹窗);
// 遮罩是工作图上的独立绘制层(maskCanvasRef,原图分辨率)。
// 「保存」时把 工作图 + 不透明纯蓝遮罩 合成为 PNG 交给 onConfirm 上传;「取消」丢弃全部改动。
// ratio 为表单目标宽/高(锁定比例裁剪用);defaultFreeCrop 时裁剪默认自由比例(第二参考图用);
// targetWidth/targetHeight 是拓展工具合成画布的输出像素。
export default function ImageEditModal({
  src,
  ratio,
  targetWidth,
  targetHeight,
  defaultFreeCrop = false,
  onConfirm,
  onClose,
}) {
  const [tool, setTool] = useState(TOOLS[0].key);
  // 工作图 URL;烘焙产生的 blob 在弹窗卸载时统一回收
  const [workSrc, setWorkSrc] = useState(src);
  // 烘焙计数:作为各工具的 key 后缀,工作图更换后强制重新挂载(重新量宽/初始化)
  const [workVersion, setWorkVersion] = useState(0);
  // 裁剪比例开关:true = 自由裁剪,false = 锁定目标比例
  const [freeCrop, setFreeCrop] = useState(defaultFreeCrop);
  // 遮罩层(offscreen canvas,工作图原始分辨率);由 PaintMaskTool 读写
  const maskCanvasRef = useRef(null);
  const cropRef = useRef(null);
  const expandRef = useRef(null);
  const [cropReady, setCropReady] = useState(false);
  const [expandReady, setExpandReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const initialSrcRef = useRef(src);

  // 弹窗卸载时回收烘焙产生的 blob URL(原始 src 由调用方管理,不动)
  useEffect(() => {
    return () => {
      if (workSrc !== initialSrcRef.current && workSrc.startsWith("blob:")) {
        URL.revokeObjectURL(workSrc);
      }
    };
  }, [workSrc]);

  // 裁剪/拓展的确认结果:成为新工作图;坐标系已变,遮罩层随之清空
  function bakeToolResult(file) {
    const url = URL.createObjectURL(file);
    setWorkSrc((prev) => {
      if (prev !== initialSrcRef.current && prev.startsWith("blob:")) {
        URL.revokeObjectURL(prev);
      }
      return url;
    });
    maskCanvasRef.current = null;
    setWorkVersion((v) => v + 1);
    setError(null);
  }

  // 保存:工作图 + 遮罩(转为不透明纯蓝)合成 PNG,交给调用方上传
  async function save() {
    setSaving(true);
    setError(null);
    try {
      const img = await loadImage(workSrc);
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const mask = maskCanvasRef.current;
      if (mask && mask.width === canvas.width && mask.height === canvas.height) {
        // 绘制时是半透明蓝,合成时转为不透明纯蓝(与原工作流 DrawMaskOnImage 一致)
        const maskLayer = document.createElement("canvas");
        maskLayer.width = canvas.width;
        maskLayer.height = canvas.height;
        const maskCtx = maskLayer.getContext("2d");
        maskCtx.fillStyle = `rgb(${MASK_COLOR})`;
        maskCtx.fillRect(0, 0, maskLayer.width, maskLayer.height);
        maskCtx.globalCompositeOperation = "destination-in";
        maskCtx.drawImage(mask, 0, 0);
        ctx.drawImage(maskLayer, 0, 0);
      }
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("图片合成失败,请重试");
      onConfirm(new File([blob], "edited-input.png", { type: "image/png" }));
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal
      id="image-edit-modal"
      title="编辑图片"
      onClose={onClose}
      footer={
        <>
          {tool === "crop" && (
            <button
              id="image-edit-apply-btn"
              type="button"
              className="chip"
              disabled={!cropReady}
              title="应用裁剪并继续编辑"
              onClick={() => cropRef.current?.confirm()}
            >
              应用裁剪
            </button>
          )}
          {tool === "expand" && (
            <button
              id="image-edit-apply-btn"
              type="button"
              className="chip"
              disabled={!expandReady}
              title="应用拓展并继续编辑"
              onClick={() => expandRef.current?.apply()}
            >
              应用拓展
            </button>
          )}
          <button
            id="image-edit-save-btn"
            type="button"
            className="primary"
            disabled={saving}
            title="合成当前工作图与遮罩,作为输入图片"
            onClick={save}
          >
            {saving ? "保存中……" : "保存"}
          </button>
          <button id="image-edit-cancel-btn" type="button" className="chip" onClick={onClose}>
            取消
          </button>
        </>
      }
    >
      <div
        id="image-edit-toolbar"
        className="image-edit-toolbar"
        role="toolbar"
        aria-label="图片编辑工具"
      >
        {TOOLS.map((item) => (
          <button
            key={item.key}
            id={`image-edit-tool-${item.key}`}
            type="button"
            className={`chip${tool === item.key ? " active" : ""}`}
            aria-pressed={tool === item.key}
            onClick={() => setTool(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {tool === "crop" && (
        <>
          <div className="image-edit-toolbar" id="image-edit-crop-mode">
            <button
              id="image-edit-crop-locked"
              type="button"
              className={`chip${freeCrop ? "" : " active"}`}
              aria-pressed={!freeCrop}
              title={`按当前表单的宽度 × 高度(${targetWidth} × ${targetHeight})裁剪`}
              onClick={() => setFreeCrop(false)}
            >
              锁定比例
            </button>
            <button
              id="image-edit-crop-free"
              type="button"
              className={`chip${freeCrop ? " active" : ""}`}
              aria-pressed={freeCrop}
              title="不限定宽高比,自由框选"
              onClick={() => setFreeCrop(true)}
            >
              自由裁剪
            </button>
          </div>
          <CropTool
            key={`crop-${workVersion}-${freeCrop}`}
            ref={cropRef}
            src={workSrc}
            ratio={freeCrop ? null : ratio}
            initialFullRect={workVersion > 0}
            idPrefix="image-edit-crop"
            onConfirm={bakeToolResult}
            onReadyChange={setCropReady}
          />
        </>
      )}

      {tool === "paint" && (
        <PaintMaskTool
          key={`paint-${workVersion}`}
          src={workSrc}
          maskCanvasRef={maskCanvasRef}
        />
      )}

      {tool === "expand" && (
        <ExpandCanvasTool
          key={`expand-${workVersion}`}
          ref={expandRef}
          src={workSrc}
          targetWidth={targetWidth}
          targetHeight={targetHeight}
          onApply={bakeToolResult}
          onReadyChange={setExpandReady}
        />
      )}

      {error && <div id="image-edit-error" className="form-error">{error}</div>}
    </Modal>
  );
}
