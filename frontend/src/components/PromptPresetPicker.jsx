import Modal from "./Modal";
import { KIND_LABELS } from "./PromptPresets";

// 工作台提示词预设选择弹窗:按目标类型(正面/负面)过滤设置页维护的预设,
// 点击卡片即用预设内容替换对应输入框的当前内容。
export default function PromptPresetPicker({ kind, presets, onSelect, onClose }) {
  const filtered = presets.filter((item) => item.kind === kind);
  const kindLabel = KIND_LABELS[kind] ?? kind;

  return (
    <Modal
      id="prompt-preset-picker"
      title={`选择${kindLabel}提示词预设`}
      onClose={onClose}
    >
      {filtered.length === 0 ? (
        <p className="preset-picker-empty">
          暂无{kindLabel}提示词预设,可在设置页「提示词」中添加。
        </p>
      ) : (
        <div className="prompt-presets-grid preset-picker-grid">
          {filtered.map((preset) => (
            <button
              key={preset.id}
              id={`preset-pick-${preset.id}`}
              type="button"
              className="prompt-preset-card"
              title="用该预设替换当前内容"
              onClick={() => onSelect(preset)}
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
    </Modal>
  );
}
