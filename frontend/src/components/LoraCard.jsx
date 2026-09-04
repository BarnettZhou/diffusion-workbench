// krea2 模式可选 LoRA 卡片:右上角开/关切换,开启后最多添加 3 个 LoRA。
// 每个槽位 = 模型下拉 + 强度输入 + 移除按钮;强度存原始字符串,提交时才转数值。
const MAX_LORAS = 3;

export default function LoraCard({ enabled, onEnabledChange, loras, onChange, options }) {
  function update(slot, patch) {
    onChange(loras.map((item, index) => (index === slot ? { ...item, ...patch } : item)));
  }
  function remove(slot) {
    onChange(loras.filter((_, index) => index !== slot));
  }
  function add() {
    const used = new Set(loras.map((item) => item.index));
    const first = options.find((item) => !used.has(item.index));
    onChange([...loras, { index: first?.index ?? null, strength: "1" }]);
  }
  return (
    <div id="lora-card" className="size-card-embedded form lora-card">
      <div className="card-header">
        <h2>LoRA</h2>
        <div id="lora-enabled-switch" className="size-mode-switch" role="tablist" aria-label="LoRA 开关">
          <button
            type="button"
            role="tab"
            aria-selected={!enabled}
            className={!enabled ? "active" : ""}
            onClick={() => onEnabledChange(false)}
          >
            关
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={enabled}
            className={enabled ? "active" : ""}
            disabled={!options.length}
            onClick={() => onEnabledChange(true)}
          >
            开
          </button>
        </div>
      </div>
      {enabled && (
        <>
          {loras.map((item, slot) => (
            <div className="row lora-row" id={`lora-row-${slot}`} key={slot}>
              <div className="field lora-model-field">
                <label htmlFor={`lora-select-${slot}`}>LoRA {slot + 1}</label>
                <select
                  id={`lora-select-${slot}`}
                  value={item.index ?? ""}
                  disabled={!options.length}
                  onChange={(e) => update(slot, { index: Number(e.target.value) })}
                >
                  {item.index === null && <option value="">请选择</option>}
                  {options.map((option) => (
                    <option
                      key={option.index}
                      value={option.index}
                      disabled={loras.some((other, index) => index !== slot && other.index === option.index)}
                    >
                      {option.display_name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field lora-strength-field">
                <label htmlFor={`lora-strength-${slot}`}>强度</label>
                <input
                  id={`lora-strength-${slot}`}
                  type="number"
                  min={0}
                  max={2}
                  step="any"
                  value={item.strength}
                  onChange={(e) => update(slot, { strength: e.target.value })}
                />
              </div>
              <button
                id={`lora-remove-${slot}`}
                type="button"
                className="prompt-assist-btn lora-remove-btn"
                title="移除该 LoRA"
                aria-label="移除该 LoRA"
                onClick={() => remove(slot)}
              >
                ×
              </button>
            </div>
          ))}
          {loras.length < MAX_LORAS && (
            <button
              id="lora-add-btn"
              type="button"
              className="chip"
              disabled={!options.length || loras.length >= options.length}
              onClick={add}
            >
              + 添加 LoRA(最多 {MAX_LORAS} 个)
            </button>
          )}
        </>
      )}
    </div>
  );
}
