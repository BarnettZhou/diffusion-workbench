import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { normalizeUploadFile } from "./EditParameterForm";
import { useMessage } from "./Message";
import RemoteModelSelect from "./RemoteModelSelect";

// 图片反推面板:上传图片 → 后端复用 krea2 的 Qwen3-VL 生成英文描述提示词。
// 「本地反推」是同步请求,与生成任务共用同一个 GPU Worker 串行执行;正在跑生成任务时会排队等待。
// 「API 反推」走设置页配置的外部视觉模型接口(Ollama / OpenAI 兼容),不占用本地 GPU。
// 结果只保留在本组件内存中(新结果置顶),不落库、不进任务队列、不发 WebSocket 事件。

// 反推方式 tabs(与视频/编辑模式切换同一个 mode-switch 样式)
const CAPTION_TYPES = [
  { key: "local", label: "本地反推" },
  { key: "remote", label: "API 反推" },
];

export default function CaptionPanel({ inputImagePrefill = null, captionApi = null, onSelectModel }) {
  const message = useMessage();
  const [captionType, setCaptionType] = useState("local");
  // 输入图片:{id, previewUrl, name};id 为服务端受控上传返回的 id
  const [image, setImage] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [hint, setHint] = useState("");
  // 数字输入框存原始字符串,提交时才校验/转换(与参数表单约定一致)
  const [maxLength, setMaxLength] = useState("2048");
  const [submitting, setSubmitting] = useState(false);
  // 反推结果列表:[{key, name, previewUrl, caption, loadSeconds, inferSeconds}]
  const [results, setResults] = useState([]);
  const inputRef = useRef(null);

  // 相册"发送到图片反推"带过来的输入图片:已在服务端导入,直接引用受控 URL,不走本地上传
  useEffect(() => {
    if (!inputImagePrefill) return;
    setImage({
      id: inputImagePrefill.id,
      previewUrl: inputImagePrefill.url,
      name: inputImagePrefill.name,
    });
  }, [inputImagePrefill]);

  // 选择文件后立即上传(HEIC 等先转 PNG),本地用 object URL 预览。
  // blob URL 不主动 revoke:结果卡片会继续引用同一张缩略图。
  async function handleImageSelect(file) {
    if (!file) return;
    setUploading(true);
    try {
      const uploadFile = await normalizeUploadFile(file);
      const previewUrl = URL.createObjectURL(uploadFile);
      try {
        const saved = await api.uploadEditInputImage(uploadFile);
        setImage({ id: saved.id, previewUrl, name: uploadFile.name });
      } catch (err) {
        URL.revokeObjectURL(previewUrl);
        throw err;
      }
    } catch (err) {
      message?.error(err.message);
    } finally {
      setUploading(false);
    }
  }

  function clearImage() {
    setImage(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!image) {
      message?.warning("请先上传要反推的图片");
      return;
    }
    const maxLengthNum = Number(maxLength);
    if (!Number.isInteger(maxLengthNum) || maxLengthNum < 64 || maxLengthNum > 2048) {
      message?.warning("最大长度需为 64-2048 的整数");
      return;
    }
    setSubmitting(true);
    try {
      const call = captionType === "remote" ? api.captionRemote : api.caption;
      const result = await call({
        image_id: image.id,
        hint: hint.trim(),
        max_length: maxLengthNum,
      });
      setResults((prev) => [
        {
          key: `${Date.now()}-${prev.length}`,
          name: image.name,
          previewUrl: image.previewUrl,
          source: captionType,
          caption: result.caption,
          loadSeconds: result.load_seconds,
          inferSeconds: result.infer_seconds,
        },
        ...prev,
      ]);
    } catch (err) {
      message?.error(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div id="caption-body">
      <aside id="caption-controls" className="controls">
        <div
          id="caption-type-switch"
          className="mode-switch"
          role="tablist"
          aria-label="反推方式"
        >
          {CAPTION_TYPES.map((item) => (
            <button
              key={item.key}
              id={`caption-type-${item.key}`}
              type="button"
              role="tab"
              aria-selected={captionType === item.key}
              className={captionType === item.key ? "active" : ""}
              onClick={() => setCaptionType(item.key)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <form id="caption-form" className="form-stack" onSubmit={handleSubmit}>
          <div className="field" id="field-caption-image">
            <div className="label-row">
              <label htmlFor="caption-image-input">输入图片(必选)</label>
              <div className="label-actions">
                {image && (
                  <button
                    id="caption-image-clear-btn"
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
            <div
              id="caption-image-drop"
              className={`video-input-image${image ? " has-image" : ""}`}
              role="button"
              tabIndex={0}
              title="点击选择图片(png/jpeg/webp,≤32MB)"
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
            >
              {image ? (
                <img src={image.previewUrl} alt={image.name} />
              ) : (
                <div className="video-input-image-empty">点击上传要反推的图片</div>
              )}
              {uploading && (
                <div className="video-input-image-loading">上传中……</div>
              )}
              <input
                ref={inputRef}
                id="caption-image-input"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                aria-required="true"
                hidden
                disabled={uploading}
                onChange={(e) => {
                  handleImageSelect(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </div>
          </div>

          {captionType === "remote" && (
            <div className="field" id="field-caption-model">
              <label htmlFor="caption-remote-model">反推模型</label>
              <RemoteModelSelect
                idPrefix="caption-remote-model"
                endpoints={captionApi?.endpoints ?? []}
                selected={captionApi?.selected ?? { endpoint_id: "", model: "" }}
                disabled={submitting}
                onChange={(next) => onSelectModel?.(next)}
              />
            </div>
          )}

          <div className="field" id="field-caption-hint">
            <label htmlFor="caption-hint-input">补充要求(可选)</label>
            <textarea
              id="caption-hint-input"
              rows={3}
              value={hint}
              placeholder="对反推结果的额外要求,如侧重光线/构图描述……"
              onChange={(e) => setHint(e.target.value)}
            />
          </div>

          <div className="field" id="field-caption-max-length">
            <label htmlFor="caption-max-length-input">最大长度(64-2048)</label>
            <input
              id="caption-max-length-input"
              type="number"
              min={64}
              max={2048}
              step={64}
              value={maxLength}
              onChange={(e) => setMaxLength(e.target.value)}
            />
            <p className="form-hint">
              {captionType === "remote"
                ? "走设置页「图片反推」中配置的视觉模型 API;不占用本地 GPU,无需排队。"
                : "反推复用 krea2 的 Qwen3-VL text encoder;正在跑生成任务时本请求会排队等待。"}
            </p>
          </div>

          <button
            id="caption-submit-btn"
            type="submit"
            className="primary"
            disabled={submitting || uploading || !image}
          >
            {submitting
              ? captionType === "remote"
                ? "反推中……"
                : "反推中……(如有生成任务将排队)"
              : "开始反推"}
          </button>
        </form>
      </aside>
      <section id="caption-results" className="results">
        {results.length === 0 ? (
          <div className="panel form-hint" id="caption-results-empty">
            反推结果会显示在这里。
          </div>
        ) : (
          <div className="caption-result-list">
            {results.map((item) => (
              <CaptionResultCard key={item.key} item={item} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

// 单条反推结果:图片缩略 + 描述文本 + 复制按钮(复制逻辑同 Lightbox 的 PromptSection)
function CaptionResultCard({ item }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(item.caption);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = item.caption;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <article className="caption-result-card">
      <div className="caption-result-head">
        {item.previewUrl && (
          <img className="caption-result-thumb" src={item.previewUrl} alt={item.name} />
        )}
        <div className="caption-result-meta">
          <span className="caption-result-name" title={item.name}>{item.name}</span>
          <span className="caption-result-timing">
            {item.source === "remote"
              ? `API · 推理 ${item.inferSeconds?.toFixed?.(1) ?? "-"}s`
              : `本地 · 加载 ${item.loadSeconds?.toFixed?.(1) ?? "-"}s · 推理 ${item.inferSeconds?.toFixed?.(1) ?? "-"}s`}
          </span>
        </div>
        <button
          type="button"
          className="drawer-copy-btn"
          onClick={copy}
        >
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      <div className="drawer-prompt">
        <div className="drawer-prompt-text">{item.caption}</div>
      </div>
    </article>
  );
}
