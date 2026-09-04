import { useRef, useState } from "react";
import { api } from "../api/client";
import ImageEditModal from "./ImageEditModal";
import { useMessage } from "./Message";

// 服务端接受的输入图片类型(与后端白名单一致)
const UPLOAD_TYPES = ["image/png", "image/jpeg", "image/webp"];

// 手机相册选出的图片可能是 HEIC 或没有 Content-Type:浏览器能解码的(如 iOS 的 HEIC)
// 统一转成 PNG 再上传;浏览器也解码不了时给出明确错误。
// EditParameterForm 再导出给 RebalanceParameterForm / CaptionPanel 复用。
export async function normalizeUploadFile(file) {
  if (UPLOAD_TYPES.includes(file.type)) return file;
  let bitmap = null;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error("该图片格式无法识别(如 HEIC),请先转换为 PNG/JPEG/WebP");
  }
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  bitmap.close();
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("图片转换失败,请换一张图片");
  const base = file.name.replace(/\.[^.]+$/, "") || "input";
  return new File([blob], `${base}.png`, { type: "image/png" });
}

// 水平翻转(镜像):不依赖后端 pipeline,在浏览器用 Canvas 生成新 File。
// 输出格式沿用原文件类型(jpeg/png/webp),文件名追加 -flipped 后缀便于区分。
// 失败时把原始错误抛给调用方,UI 已经在调用点 toast 出来。
export async function flipImageHorizontal(file) {
  const bitmap = await createImageBitmap(file);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d");
    // 镜像:先平移到右上角,再水平反转,再画图
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(bitmap, 0, 0);
    const outType = UPLOAD_TYPES.includes(file.type) ? file.type : "image/png";
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, outType));
    if (!blob) throw new Error("图片翻转失败");
    const base = (file.name || "input").replace(/\.[^.]+$/, "") || "input";
    const ext = outType.split("/")[1] || "png";
    return new File([blob], `${base}-flipped.${ext}`, { type: outType });
  } finally {
    bitmap.close();
  }
}

// 受控的图片输入组件:上传原图 + ImageEditModal 编辑(裁剪/遮罩/拓展) + 双栏展示。
// value 形如 { original: {id, previewUrl, name, file?}, edited: {id, previewUrl, name} | null },
// null 表示未上传。edited 存在时作为提交输入,否则用 original;编辑始终基于原图。
// ratio 为裁剪锁定比例(宽/高);defaultFreeCrop 为 true 时弹窗默认自由裁剪,不校验宽高。
export default function EditImageField({
  label = "输入图片(必选)",
  idPrefix,
  value,
  onChange,
  ratio,
  targetWidth,
  targetHeight,
  defaultFreeCrop = false,
  requiredHint = "点击上传要编辑的图片",
  onError,
  onUploadingChange,
}) {
  const message = useMessage();
  const [originalUploading, setOriginalUploading] = useState(false);
  const [editedUploading, setEditedUploading] = useState(false);
  // 水平翻转进度:与上传互斥,共享 onUploadingChange 通知外层
  const [flipping, setFlipping] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const original = value?.original ?? null;
  const edited = value?.edited ?? null;

  function reportError(err) {
    onError?.(err.message);
    // 移动端表单较长,底部内联错误可能不在视野内,同步弹全局 toast
    message?.error(err.message);
  }

  function revokePreview(item) {
    if (item?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(item.previewUrl);
  }

  // 拖拽上传:dragover 必须 preventDefault 才会触发 drop;拖入的文件走与选择文件相同的上传路径
  function handleDragOver(event) {
    event.preventDefault();
    if (!originalUploading) setDragOver(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    setDragOver(false);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragOver(false);
    if (originalUploading) return;
    handleOriginalSelect(event.dataTransfer?.files?.[0]);
  }

  // 选择文件后立即上传,本地用 object URL 预览;保存 File 以便后续再次编辑。
  // 上传新原图会清空已编辑图片(右栏),编辑始终基于原图。
  async function handleOriginalSelect(file) {
    if (!file) return;
    onError?.(null);
    setOriginalUploading(true);
    onUploadingChange?.(true);
    try {
      const uploadFile = await normalizeUploadFile(file);
      const previewUrl = URL.createObjectURL(uploadFile);
      try {
        const saved = await api.uploadEditInputImage(uploadFile);
        revokePreview(original);
        revokePreview(edited);
        onChange({
          original: { id: saved.id, previewUrl, name: uploadFile.name, file: uploadFile },
          edited: null,
        });
      } catch (err) {
        URL.revokeObjectURL(previewUrl);
        throw err;
      }
    } catch (err) {
      reportError(err);
    } finally {
      setOriginalUploading(false);
      onUploadingChange?.(false);
    }
  }

  // 编辑图片弹窗确认后的结果图:上传受控通道,展示在右栏并作为提交输入
  async function handleEditedSelect(file) {
    onError?.(null);
    setEditedUploading(true);
    onUploadingChange?.(true);
    try {
      const previewUrl = URL.createObjectURL(file);
      try {
        const saved = await api.uploadEditInputImage(file);
        revokePreview(edited);
        onChange({
          original,
          edited: { id: saved.id, previewUrl, name: file.name },
        });
      } catch (err) {
        URL.revokeObjectURL(previewUrl);
        throw err;
      }
    } catch (err) {
      reportError(err);
    } finally {
      setEditedUploading(false);
      onUploadingChange?.(false);
    }
  }

  function clearImage() {
    revokePreview(original);
    revokePreview(edited);
    onChange(null);
  }

  // 水平翻转当前原图:浏览器侧 Canvas 镜像后,走受控上传通道写一个新 id。
  // 翻转后清除已编辑图(原编辑基于旧原图,继续显示会产生误导),
  // 新原图作为后续编辑 / 提交的新基线。
  async function handleFlipHorizontal() {
    if (!original?.file || flipping || originalUploading || editedUploading) return;
    setFlipping(true);
    onUploadingChange?.(true);
    onError?.(null);
    try {
      const flippedFile = await flipImageHorizontal(original.file);
      const previewUrl = URL.createObjectURL(flippedFile);
      try {
        const saved = await api.uploadEditInputImage(flippedFile);
        revokePreview(original);
        revokePreview(edited);
        onChange({
          original: {
            id: saved.id,
            previewUrl,
            name: flippedFile.name,
            file: flippedFile,
          },
          edited: null,
        });
      } catch (err) {
        URL.revokeObjectURL(previewUrl);
        throw err;
      }
    } catch (err) {
      reportError(err);
    } finally {
      setFlipping(false);
      onUploadingChange?.(false);
    }
  }

  // 打开编辑图片弹窗:锁定比例取自当前宽度/高度,自由裁剪时不校验;编辑源始终是原图
  function openEdit() {
    if (!defaultFreeCrop) {
      if (!Number.isFinite(ratio) || ratio <= 0) {
        onError?.("裁剪比例取自宽度和高度,请先把这两个值填为有效数值");
        return;
      }
    }
    onError?.(null);
    setEditOpen(true);
  }

  return (
    <div className="field" id={`field-${idPrefix}`}>
      <div className="label-row">
        <label htmlFor={`${idPrefix}-input`}>{label}</label>
        <div className="label-actions">
          {original && (
            <button
              id={`${idPrefix}-edit-btn`}
              type="button"
              className="prompt-assist-btn"
              title="编辑输入图片(裁剪等),编辑源为原图"
              aria-label="编辑输入图片"
              onClick={openEdit}
            >
              编辑图片
            </button>
          )}
          {original && (
            <button
              id={`${idPrefix}-flip-btn`}
              type="button"
              className="prompt-assist-btn"
              title="水平翻转当前原图(浏览器侧处理,会替换原图并清空已编辑结果)"
              aria-label="水平翻转输入图片"
              disabled={flipping || originalUploading || editedUploading}
              onClick={handleFlipHorizontal}
            >
              {flipping ? "翻转中……" : "水平翻转"}
            </button>
          )}
          {original && (
            <button
              id={`${idPrefix}-clear-btn`}
              type="button"
              className="prompt-assist-btn"
              title="移除输入图片"
              aria-label="移除输入图片"
              onClick={clearImage}
            >
              移除
            </button>
          )}
        </div>
      </div>
      {edited && original ? (
        <div className="input-image-lanes" id={`${idPrefix}-lanes`}>
          <div className="input-image-lane" id={`${idPrefix}-lane-original`}>
            <div
              id={`${idPrefix}-drop`}
              className={`video-input-image has-image${dragOver ? " drag-over" : ""}`}
              role="button"
              tabIndex={0}
              title="点击或拖入图片重新上传原图(png/jpeg/webp,≤32MB)"
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <img src={original.previewUrl} alt={original.name} />
              {originalUploading && (
                <div className="video-input-image-loading">上传中……</div>
              )}
            </div>
            <span className="input-image-lane-label">原图(点击重新上传)</span>
          </div>
          <div className="input-image-lane" id={`${idPrefix}-lane-edited`}>
            <div
              id={`${idPrefix}-edited-preview`}
              className="video-input-image has-image is-static"
              title="编辑后的图片将作为输入"
            >
              <img src={edited.previewUrl} alt={edited.name} />
              {editedUploading && (
                <div className="video-input-image-loading">上传中……</div>
              )}
            </div>
            <span className="input-image-lane-label">已编辑(将作为输入)</span>
          </div>
        </div>
      ) : (
        <div
          id={`${idPrefix}-drop`}
          className={`video-input-image${original ? " has-image" : ""}${dragOver ? " drag-over" : ""}`}
          role="button"
          tabIndex={0}
          title="点击选择或拖入图片(png/jpeg/webp,≤32MB)"
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {original ? (
            <img src={original.previewUrl} alt={original.name} />
          ) : (
            <div className="video-input-image-empty">{requiredHint}</div>
          )}
          {originalUploading && (
            <div className="video-input-image-loading">上传中……</div>
          )}
        </div>
      )}
      <input
        ref={inputRef}
        id={`${idPrefix}-input`}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        aria-required="true"
        hidden
        disabled={originalUploading}
        onChange={(e) => {
          handleOriginalSelect(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {original && (
        <p className="form-hint" id={`${idPrefix}-hint`}>
          {edited
            ? `已上传:${original.name};编辑后的图片将作为输入,点击左侧原图可重新上传。`
            : defaultFreeCrop
              ? `已上传:${original.name},可在「编辑图片」中自由裁剪。`
              : `已上传:${original.name},可在「编辑图片」中按当前宽度×高度裁剪。`}
        </p>
      )}

      {editOpen && original && (
        <ImageEditModal
          src={original.previewUrl}
          ratio={defaultFreeCrop ? null : ratio}
          targetWidth={targetWidth}
          targetHeight={targetHeight}
          defaultFreeCrop={defaultFreeCrop}
          onClose={() => setEditOpen(false)}
          onConfirm={(file) => {
            setEditOpen(false);
            handleEditedSelect(file);
          }}
        />
      )}
    </div>
  );
}
