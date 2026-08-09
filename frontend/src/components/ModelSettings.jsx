import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import Modal from "./Modal";

const MODE_LABELS = {
  zit: "Z-Image-Turbo",
  krea2: "Krea 2",
  zib: "Z-Image Base",
  sdxl: "SDXL",
};

function formatSize(bytes) {
  if (bytes == null) return "未知";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

export default function ModelSettings({ modes, onResourcesChanged }) {
  const [mode, setMode] = useState("zit");
  const [models, setModels] = useState([]);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState(null);
  const [editing, setEditing] = useState(null);
  const modeTabs = (modes ?? []).map((key) => ({ key, label: MODE_LABELS[key] ?? key }));

  useEffect(() => {
    if (modes?.length && !modes.includes(mode)) setMode(modes[0]);
  }, [modes, mode]);

  async function load(targetMode) {
    try {
      const body = await api.models(targetMode);
      setModels(body.models);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    load(mode);
  }, [mode]);

  const keyword = filter.trim().toLowerCase();
  const filteredModels = keyword
    ? models.filter(
        (m) =>
          m.name.toLowerCase().includes(keyword) ||
          (m.alias ?? "").toLowerCase().includes(keyword),
      )
    : models;

  async function handleSaved() {
    await load(mode);
    onResourcesChanged?.();
  }

  return (
    <div id="settings-models" className="panel">
      <div id="models-toolbar">
        <div id="models-mode-tabs" className="mode-switch">
          {modeTabs.map((tab) => (
            <button
              key={tab.key}
              id={`models-mode-${tab.key}`}
              type="button"
              className={mode === tab.key ? "active" : ""}
              onClick={() => setMode(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <input
          id="models-filter-input"
          type="text"
          value={filter}
          placeholder="筛选模型"
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      {error && <div className="form-error">加载失败:{error}</div>}
      {filteredModels.length === 0 && !error && (
        <div className="empty-hint">
          {models.length === 0 ? "该模式下没有模型" : "没有匹配的模型"}
        </div>
      )}

      <div id="models-grid">
        {filteredModels.map((model) => (
          <div
            className="model-card"
            id={`model-card-${model.name}`}
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
        <ModelEditor
          model={editing}
          onClose={() => setEditing(null)}
          onSaved={handleSaved}
        />
      )}
    </div>
  );
}

function ModelEditor({ model, onClose, onSaved }) {
  const [alias, setAlias] = useState(model.alias ?? "");
  const [note, setNote] = useState(model.note ?? "");
  const [coverUrl, setCoverUrl] = useState(model.cover_url);
  const [quant, setQuant] = useState(model.quant);
  const [fetchingQuant, setFetchingQuant] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  async function fetchQuant() {
    setFetchingQuant(true);
    setError(null);
    try {
      const body = await api.fetchModelQuant(model.mode, model.name);
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
      await api.updateModelInfo(model.mode, model.name, { alias, note });
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
      await api.uploadModelCover(model.mode, model.name, file);
      // 加时间戳绕过浏览器缓存
      setCoverUrl(
        `/api/v1/models/${model.mode}/${encodeURIComponent(model.name)}/cover?t=${Date.now()}`,
      );
      onSaved();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <Modal id="model-editor" title={`编辑模型 ${model.name}`} onClose={onClose}>
      <div className="model-editor-main">
        <div
          id="editor-cover"
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
            id="editor-cover-input"
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
            <label htmlFor="editor-alias-input">别名</label>
            <input
              id="editor-alias-input"
              type="text"
              value={alias}
              placeholder="模型别名(同模式内唯一)"
              onChange={(e) => setAlias(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="editor-note-input">备注</label>
            <textarea
              id="editor-note-input"
              value={note}
              placeholder="备注信息"
              onChange={(e) => setNote(e.target.value)}
            />
          </div>
          {error && <div id="editor-error" className="form-error">{error}</div>}
          <div className="row">
            <button
              id="editor-save-btn"
              type="button"
              className="primary"
              disabled={saving}
              onClick={saveInfo}
            >
              {saving ? "保存中……" : "保存"}
            </button>
            <button id="editor-cancel-btn" type="button" className="chip" onClick={onClose}>
              取消
            </button>
          </div>
        </div>
      </div>

      <div id="editor-info-card" className="model-readonly-card">
        <div><span>文件名</span><em title={model.name}>{model.name}</em></div>
        <div><span>类型</span><em>{model.mode}</em></div>
        <div><span>大小</span><em>{formatSize(model.size_bytes)}</em></div>
        <div>
          <span>量化</span>
          {quant ? (
            <em>{quant}</em>
          ) : (
            <button
              id="editor-quant-fetch-btn"
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
