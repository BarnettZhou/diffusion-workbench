import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import Modal from "./Modal";
import { COVER_SPACER } from "./modelCover";

// 支持可选 LoRA 的生图模式(与服务端 domain 校验一致)
const LORA_MODE_LABELS = {
  zit: "Z-Image-Turbo",
  krea2: "Krea 2",
};

// 排序规则与生图模型一致:先按标题(别名),再按文件名
function compareLoras(a, b) {
  const key = (item) => item.alias ?? item.name;
  const byKey = String(key(a)).localeCompare(String(key(b)), undefined, {
    sensitivity: "base",
    numeric: true,
  });
  if (byKey !== 0) return byKey;
  return String(a.name).localeCompare(String(b.name), undefined, {
    sensitivity: "base",
    numeric: true,
  });
}

function formatSize(bytes) {
  if (bytes == null) return "未知";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

export default function LoraSettings({ modes, onResourcesChanged, privacyMode = false }) {
  // 只保留支持 LoRA 的模式;modes 为空(未加载)时回退到全部 LoRA 模式
  const loraModes = (modes ?? []).filter((key) => key in LORA_MODE_LABELS);
  const [mode, setMode] = useState(loraModes[0] ?? "zit");
  const [loras, setLoras] = useState([]);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState(null);
  const [editing, setEditing] = useState(null);
  const modeTabs = loraModes.map((key) => ({ key, label: LORA_MODE_LABELS[key] }));

  useEffect(() => {
    if (loraModes.length && !loraModes.includes(mode)) setMode(loraModes[0]);
  }, [modes]); // eslint-disable-line react-hooks/exhaustive-deps

  async function load(targetMode) {
    try {
      const body = await api.loras(targetMode);
      setLoras([...(body.loras ?? [])].sort(compareLoras));
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    load(mode);
  }, [mode]);

  const keyword = filter.trim().toLowerCase();
  const filteredLoras = keyword
    ? loras.filter(
        (item) =>
          item.name.toLowerCase().includes(keyword) ||
          (item.alias ?? "").toLowerCase().includes(keyword),
      )
    : loras;

  async function handleSaved() {
    await load(mode);
    onResourcesChanged?.();
  }

  return (
    <div id="settings-loras" className="panel">
      <div id="models-toolbar">
        <div id="loras-mode-tabs" className="mode-switch">
          {modeTabs.map((tab) => (
            <button
              key={tab.key}
              id={`loras-mode-${tab.key}`}
              type="button"
              className={mode === tab.key ? "active" : ""}
              onClick={() => setMode(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <input
          id="loras-filter-input"
          type="text"
          value={filter}
          placeholder="筛选 LoRA"
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      {error && <div className="form-error">加载失败:{error}</div>}
      {filteredLoras.length === 0 && !error && (
        <div className="empty-hint">
          {loras.length === 0
            ? "该模式下没有 LoRA(请在 workbench.yaml 的 resources.<mode>.loras 配置候选目录)"
            : "没有匹配的 LoRA"}
        </div>
      )}

      <div id="models-grid">
        {filteredLoras.map((lora) => (
          <div
            className="model-card"
            id={`lora-card-${lora.name}`}
            key={lora.name}
            role="button"
            tabIndex={0}
            onClick={() => setEditing(lora)}
            onKeyDown={(e) => e.key === "Enter" && setEditing(lora)}
          >
            <div className="model-cover">
              {privacyMode ? (
                <div className="model-cover-privacy" aria-label="隐私模式">
                  <img className="model-cover-spacer" src={COVER_SPACER} alt="" />
                  <span>隐私模式</span>
                </div>
              ) : lora.has_cover ? (
                <img src={lora.cover_url} alt={lora.name} loading="lazy" />
              ) : (
                <div className="model-cover-empty">
                  <img className="model-cover-spacer" src={COVER_SPACER} alt="" />
                  <span>暂无封面</span>
                </div>
              )}
            </div>
            <div className="model-info">
              <div className="model-name" title={lora.name}>{lora.alias ?? lora.name}</div>
              <div className="model-sub">
                {lora.alias ? <span className="model-alias">{lora.name}</span> : null}
                <span className="model-mode">{LORA_MODE_LABELS[lora.mode] ?? lora.mode}</span>
              </div>
              <div className="model-sub">
                <span>{formatSize(lora.size_bytes)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {editing && (
        <LoraEditor
          lora={editing}
          onClose={() => setEditing(null)}
          onSaved={handleSaved}
        />
      )}
    </div>
  );
}

function LoraEditor({ lora, onClose, onSaved }) {
  const [title, setTitle] = useState(lora.alias ?? "");
  const [note, setNote] = useState(lora.note ?? "");
  const [coverUrl, setCoverUrl] = useState(lora.cover_url);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  async function saveInfo() {
    setSaving(true);
    setError(null);
    try {
      await api.updateLoraInfo(lora.mode, lora.name, { alias: title, note });
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
      await api.uploadLoraCover(lora.mode, lora.name, file);
      // 加时间戳绕过浏览器缓存
      setCoverUrl(
        `/api/v1/loras/${lora.mode}/${encodeURIComponent(lora.name)}/cover?t=${Date.now()}`,
      );
      onSaved();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <Modal
      id="lora-editor"
      title={`编辑 LoRA ${lora.name}`}
      onClose={onClose}
      // 编辑信息时遮罩点击不关闭,防止误触丢失未保存内容;Esc 仍可关闭
      maskClosable={false}
    >
      <div className="model-editor-main">
        <div
          id="lora-editor-cover"
          className="model-cover editable"
          role="button"
          tabIndex={0}
          title="点击替换封面"
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
        >
          {coverUrl ? (
            <img src={coverUrl} alt={lora.name} />
          ) : (
            <div className="model-cover-empty">
              <img className="model-cover-spacer" src={COVER_SPACER} alt="" />
              <span>暂无封面<br />点击上传</span>
            </div>
          )}
          <input
            ref={fileInputRef}
            id="lora-editor-cover-input"
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
            <label htmlFor="lora-editor-title-input">标题</label>
            <input
              id="lora-editor-title-input"
              type="text"
              value={title}
              placeholder="LoRA 标题(同模式内唯一)"
              onChange={(e) => setTitle(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="lora-editor-note-input">备注</label>
            <textarea
              id="lora-editor-note-input"
              value={note}
              placeholder="备注信息"
              onChange={(e) => setNote(e.target.value)}
            />
          </div>
          {error && <div id="lora-editor-error" className="form-error">{error}</div>}
          <div className="row">
            <button
              id="lora-editor-save-btn"
              type="button"
              className="primary"
              disabled={saving}
              onClick={saveInfo}
            >
              {saving ? "保存中……" : "保存"}
            </button>
            <button id="lora-editor-cancel-btn" type="button" className="chip" onClick={onClose}>
              取消
            </button>
          </div>
        </div>
      </div>

      <div id="lora-editor-info-card" className="model-readonly-card">
        <div><span>文件名</span><em title={lora.name}>{lora.name}</em></div>
        <div><span>类型</span><em>{LORA_MODE_LABELS[lora.mode] ?? lora.mode}</em></div>
        <div><span>大小</span><em>{formatSize(lora.size_bytes)}</em></div>
      </div>
    </Modal>
  );
}
