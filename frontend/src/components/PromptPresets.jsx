import { useState } from "react";
import Modal from "./Modal";
import { useMessage } from "./Message";

export const KIND_LABELS = { positive: "正面", negative: "负面" };

// 设置页「提示词」tab:常用提示词预设的卡片列表。
// 数据存在全局 settings.prompt_presets,增删改均为整表替换式提交。
// 点击卡片在弹窗中查看完整内容并编辑;右上角按钮添加新预设。
export default function PromptPresets({ presets, onUpdate }) {
  const message = useMessage();
  // editing: null 关闭;"new" 新建;否则为被编辑预设的 id
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({ title: "", kind: "positive", text: "" });
  const [saving, setSaving] = useState(false);

  const editingPreset =
    editing && editing !== "new"
      ? (presets.find((item) => item.id === editing) ?? null)
      : null;

  function openAdd() {
    setForm({ title: "", kind: "positive", text: "" });
    setEditing("new");
  }

  function openEdit(preset) {
    setForm({ title: preset.title, kind: preset.kind, text: preset.text });
    setEditing(preset.id);
  }

  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function save() {
    const title = form.title.trim();
    if (!title) {
      message.error("标题不能为空");
      return;
    }
    setSaving(true);
    try {
      const next = editingPreset
        ? presets.map((item) =>
            item.id === editingPreset.id ? { ...item, ...form, title } : item,
          )
        : [...presets, { id: crypto.randomUUID(), ...form, title }];
      await onUpdate({ prompt_presets: next });
      setEditing(null);
      message.success(editingPreset ? "提示词已保存" : "提示词已添加");
    } catch (err) {
      message.error(`保存失败:${err.message}`);
    } finally {
      setSaving(false);
    }
  }

  async function remove() {
    if (!editingPreset) return;
    setSaving(true);
    try {
      await onUpdate({
        prompt_presets: presets.filter((item) => item.id !== editingPreset.id),
      });
      setEditing(null);
      message.success("提示词已删除");
    } catch (err) {
      message.error(`删除失败:${err.message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div id="settings-prompts" className="panel">
      <h2 className="prompt-presets-head">
        提示词
        <button
          id="prompt-preset-add-btn"
          type="button"
          className="chip"
          onClick={openAdd}
        >
          + 添加提示词
        </button>
      </h2>
      {presets.length === 0 ? (
        <p className="prompt-presets-empty">
          还没有提示词预设,点击右上角「添加提示词」创建。
        </p>
      ) : (
        <div className="prompt-presets-grid" id="prompt-presets-grid">
          {presets.map((preset) => (
            <button
              key={preset.id}
              id={`prompt-preset-card-${preset.id}`}
              type="button"
              className="prompt-preset-card"
              onClick={() => openEdit(preset)}
            >
              <span className="prompt-preset-card-head">
                <span className="prompt-preset-title">{preset.title}</span>
                <span className={`prompt-preset-kind ${preset.kind}`}>
                  {KIND_LABELS[preset.kind] ?? preset.kind}
                </span>
              </span>
              <span className="prompt-preset-text">{preset.text}</span>
            </button>
          ))}
        </div>
      )}

      {editing && (
        <Modal
          id="prompt-preset-modal"
          title={editingPreset ? "编辑提示词" : "添加提示词"}
          onClose={() => setEditing(null)}
          closeDisabled={saving}
          footer={
            <>
              {editingPreset && (
                <button
                  id="prompt-preset-delete-btn"
                  type="button"
                  className="chip prompt-preset-delete"
                  disabled={saving}
                  onClick={remove}
                >
                  删除
                </button>
              )}
              <button
                id="prompt-preset-cancel-btn"
                type="button"
                className="chip"
                disabled={saving}
                onClick={() => setEditing(null)}
              >
                取消
              </button>
              <button
                id="prompt-preset-save-btn"
                type="button"
                className="chip"
                disabled={saving}
                onClick={save}
              >
                {saving ? "保存中……" : "保存"}
              </button>
            </>
          }
        >
          <div className="prompt-preset-form">
            <label htmlFor="prompt-preset-title">
              标题
              <input
                id="prompt-preset-title"
                type="text"
                value={form.title}
                autoFocus
                onChange={(e) => setField("title", e.target.value)}
              />
            </label>
            <label htmlFor="prompt-preset-kind">
              类型
              <select
                id="prompt-preset-kind"
                value={form.kind}
                onChange={(e) => setField("kind", e.target.value)}
              >
                <option value="positive">正面</option>
                <option value="negative">负面</option>
              </select>
            </label>
            <label htmlFor="prompt-preset-text">
              提示词
              <textarea
                id="prompt-preset-text"
                value={form.text}
                rows={8}
                onChange={(e) => setField("text", e.target.value)}
              />
            </label>
          </div>
        </Modal>
      )}
    </div>
  );
}
