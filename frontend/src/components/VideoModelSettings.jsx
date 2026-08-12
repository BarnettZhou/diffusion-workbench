import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import Modal from "./Modal";

// 视频模型分类显示名(与 App 顶部视频生成 tab 同款映射)
const VIDEO_MODEL_LABELS = {
  "wan2.2-ti2v-5b": "Wan 2.2 TI2V-5B",
  "wan2.2-i2v-14b": "Wan 2.2 I2V-14B",
  "minimax-h3": "MiniMax H3",
};

function formatSize(bytes) {
  if (bytes == null) return "未知";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

// 视频模型设置:结构与 ModelSettings 一致,分类 tab 从 /api/v1/video/models 动态渲染;
// 后端不支持视频模型别名,编辑弹窗只有封面/备注/量化。
export default function VideoModelSettings() {
  const [videoModels, setVideoModels] = useState([]);
  const [videoModelInfo, setVideoModelInfo] = useState({});
  const [videoModel, setVideoModel] = useState(null);
  const [models, setModels] = useState([]);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState(null);
  const [editing, setEditing] = useState(null);

  // 分类列表加载(失败时保持空 tab 并提示)
  useEffect(() => {
    api
      .videoModels()
      .then((body) => {
        const items = body.video_models ?? [];
        const list = items.map((item) => item.video_model);
        setVideoModels(list);
        setVideoModelInfo(Object.fromEntries(items.map((item) => [item.video_model, item])));
        setVideoModel((prev) => (prev && list.includes(prev) ? prev : (list[0] ?? null)));
      })
      .catch((err) => setError(err.message));
  }, []);

  async function load(target) {
    try {
      const body = await api.videoModelCards(target);
      setModels(body.models);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    if (videoModel) load(videoModel);
  }, [videoModel]);

  const keyword = filter.trim().toLowerCase();
  const filteredModels = keyword
    ? models.filter(
        (m) =>
          m.name.toLowerCase().includes(keyword) ||
          (m.alias ?? "").toLowerCase().includes(keyword),
      )
    : models;

  async function handleSaved() {
    if (videoModel) await load(videoModel);
  }

  return (
    <div id="settings-video-models" className="panel">
      <div id="video-models-toolbar">
        <div id="video-models-tabs" className="mode-switch">
          {videoModels.map((key) => (
            <button
              key={key}
              id={`video-models-tab-${key}`}
              type="button"
              className={videoModel === key ? "active" : ""}
              onClick={() => setVideoModel(key)}
            >
              {videoModelInfo[key]?.label ?? VIDEO_MODEL_LABELS[key] ?? key}
            </button>
          ))}
        </div>
        <input
          id="video-models-filter-input"
          type="text"
          value={filter}
          placeholder="筛选模型"
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      {error && <div className="form-error">加载失败:{error}</div>}
      {filteredModels.length === 0 && !error && (
        <div className="empty-hint">
          {models.length === 0 ? "该分类下没有模型" : "没有匹配的模型"}
        </div>
      )}

      <div id="video-models-grid">
        {filteredModels.map((model) => (
          <div
            className="model-card"
            id={`video-model-card-${model.name}`}
            key={model.name}
            role="button"
            tabIndex={0}
            onClick={() => setEditing(model)}
            onKeyDown={(e) => e.key === "Enter" && setEditing(model)}
          >
            <div className="model-cover">
              {model.has_cover ? (
                <img src={model.cover_url} alt={model.name} loading="lazy" />
              ) : (
                <div className="model-cover-empty">暂无封面</div>
              )}
            </div>
            <div className="model-info">
              <div className="model-name" title={model.name}>{model.name}</div>
              <div className="model-sub">
                {model.alias ? <span className="model-alias">{model.alias}</span> : null}
                <span className="model-mode">{model.mode}</span>
              </div>
              <div className="model-sub">
                <span>{formatSize(model.size_bytes)}</span>
                <span className="model-quant">{model.quant ?? "未知"}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {editing && (
        <VideoModelEditor
          videoModel={videoModel}
          model={editing}
          onClose={() => setEditing(null)}
          onSaved={handleSaved}
        />
      )}
    </div>
  );
}

function VideoModelEditor({ videoModel, model, onClose, onSaved }) {
  const [note, setNote] = useState(model.note ?? "");
  const [coverUrl, setCoverUrl] = useState(model.has_cover ? model.cover_url : null);
  const [quant, setQuant] = useState(model.quant);
  const [fetchingQuant, setFetchingQuant] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  async function fetchQuant() {
    setFetchingQuant(true);
    setError(null);
    try {
      const body = await api.fetchVideoModelQuant(videoModel, model.name);
      setQuant(body.quant);
      onSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setFetchingQuant(false);
    }
  }

  async function saveInfo() {
    setSaving(true);
    setError(null);
    try {
      await api.updateVideoModelInfo(videoModel, model.name, { note });
      await onSaved();
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function uploadCover(file) {
    if (!file) return;
    setError(null);
    try {
      await api.uploadVideoModelCover(videoModel, model.name, file);
      // 加时间戳绕过浏览器缓存
      setCoverUrl(
        `/api/v1/video-models/${encodeURIComponent(videoModel)}/${encodeURIComponent(model.name)}/cover?t=${Date.now()}`,
      );
      onSaved();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <Modal id="video-model-editor" title={`编辑模型 ${model.name}`} onClose={onClose}>
      <div className="model-editor-main">
        <div
          id="video-editor-cover"
          className="model-cover editable"
          role="button"
          tabIndex={0}
          title="点击替换封面"
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
        >
          {coverUrl ? (
            <img src={coverUrl} alt={model.name} />
          ) : (
            <div className="model-cover-empty">暂无封面<br />点击上传</div>
          )}
          <input
            ref={fileInputRef}
            id="video-editor-cover-input"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            hidden
            onChange={(e) => {
              uploadCover(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>

        <div className="model-editor-form">
          <div className="field">
            <label htmlFor="video-editor-note-input">备注</label>
            <textarea
              id="video-editor-note-input"
              value={note}
              placeholder="备注信息"
              onChange={(e) => setNote(e.target.value)}
            />
          </div>
          {error && <div id="video-editor-error" className="form-error">{error}</div>}
          <div className="row">
            <button
              id="video-editor-save-btn"
              type="button"
              className="primary"
              disabled={saving}
              onClick={saveInfo}
            >
              {saving ? "保存中……" : "保存"}
            </button>
            <button id="video-editor-cancel-btn" type="button" className="chip" onClick={onClose}>
              取消
            </button>
          </div>
        </div>
      </div>

      <div id="video-editor-info-card" className="model-readonly-card">
        <div><span>文件名</span><em title={model.name}>{model.name}</em></div>
        <div><span>类型</span><em>{model.mode}</em></div>
        <div><span>大小</span><em>{formatSize(model.size_bytes)}</em></div>
        <div>
          <span>量化</span>
          {quant ? (
            <em>{quant}</em>
          ) : (
            <button
              id="video-editor-quant-fetch-btn"
              type="button"
              className="chip"
              disabled={fetchingQuant}
              onClick={fetchQuant}
            >
              {fetchingQuant ? "获取中……" : "获取"}
            </button>
          )}
        </div>
      </div>
    </Modal>
  );
}
