import { useEffect, useMemo, useState } from "react";

// 宽高倍数的默认约束(与各表单的 LIMITS.size 一致)
const DEFAULT_LIMITS = { min: 256, max: 4096, multiple: 16 };

// 按目标像素(百万像素)与宽高比计算输出尺寸:先按面积开方,再取整到 multiple 的倍数并夹紧到合法区间
export function computeAutoSize(megapixels, ratio, limits = DEFAULT_LIMITS) {
  const { min, max, multiple } = limits;
  const [rw, rh] = ratio;
  const total = megapixels * 1_000_000;
  const clamp = (value) =>
    Math.min(Math.max(Math.round(value / multiple) * multiple, min), max);
  const width = clamp(Math.sqrt((total * rw) / rh));
  const height = clamp(Math.sqrt((total * rh) / rw));
  return { width, height };
}

// 图片尺寸卡片:宽/高输入项集中在同一张卡片里,顶部开关切换两种模式。
// - 手动输入:直接填写宽度/高度,下方是快捷尺寸标签(sizePresets)。
// - 自动计算:填写目标像素(1 = 100 万像素)并选择图片比例(ratioPresets),
//   自动算出合法宽高并写回。宽高始终是父组件的受控字符串,提交校验不变。
// embedded=true 时渲染为内嵌卡片(无边距面板),放在其他表单卡片内部使用。
export default function SizeInputCard({
  width,
  height,
  onWidthChange,
  onHeightChange,
  sizePresets = [],
  ratioPresets = [],
  limits = DEFAULT_LIMITS,
  idPrefix = "size",
  embedded = false,
}) {
  // manual = 手动输入模式;auto = 自动计算模式
  const [mode, setMode] = useState("manual");
  // 目标像素(百万像素),存原始字符串允许中间态;1 表示 100 万像素
  const [targetMp, setTargetMp] = useState("1");
  // 选中的比例下标;比例列表变化后越界时回退到第一项
  const [ratioIndex, setRatioIndex] = useState(0);
  const ratio = ratioPresets.length
    ? ratioPresets[Math.min(ratioIndex, ratioPresets.length - 1)]
    : null;

  // 自动计算结果:目标像素/比例任一非法时为 null(不写回宽高)
  const computed = useMemo(() => {
    if (mode !== "auto" || !ratio) return null;
    const mp = Number(targetMp);
    if (!Number.isFinite(mp) || mp <= 0) return null;
    return computeAutoSize(mp, ratio, limits);
  }, [mode, targetMp, ratio, limits]);

  // 计算结果变化时写回父组件的宽高输入框(保持提交校验/裁剪比例等下游逻辑不变)
  useEffect(() => {
    if (!computed) return;
    onWidthChange(String(computed.width));
    onHeightChange(String(computed.height));
  }, [computed, onWidthChange, onHeightChange]);

  return (
    <div id={`${idPrefix}-card`} className={embedded ? "size-card-embedded form" : "panel form"}>
      <div className="card-header">
        <h2>图片尺寸</h2>
        <div
          id={`${idPrefix}-mode-switch`}
          className="size-mode-switch"
          role="tablist"
          aria-label="尺寸输入模式"
        >
          <button
            id={`${idPrefix}-mode-manual`}
            type="button"
            role="tab"
            aria-selected={mode === "manual"}
            className={mode === "manual" ? "active" : ""}
            onClick={() => setMode("manual")}
          >
            手动输入
          </button>
          <button
            id={`${idPrefix}-mode-auto`}
            type="button"
            role="tab"
            aria-selected={mode === "auto"}
            className={mode === "auto" ? "active" : ""}
            onClick={() => setMode("auto")}
          >
            自动计算
          </button>
        </div>
      </div>

      {mode === "manual" ? (
        <>
          <div className="row" id={`${idPrefix}-row`}>
            <div className="field" id={`field-${idPrefix}-width`}>
              <label htmlFor={`${idPrefix}-width-input`}>宽度 (px)</label>
              <input
                id={`${idPrefix}-width-input`}
                type="number"
                step={limits.multiple}
                min={limits.min}
                max={limits.max}
                value={width}
                onChange={(e) => onWidthChange(e.target.value)}
              />
            </div>
            <div className="field" id={`field-${idPrefix}-height`}>
              <label htmlFor={`${idPrefix}-height-input`}>高度 (px)</label>
              <input
                id={`${idPrefix}-height-input`}
                type="number"
                step={limits.multiple}
                min={limits.min}
                max={limits.max}
                value={height}
                onChange={(e) => onHeightChange(e.target.value)}
              />
            </div>
          </div>
          {(sizePresets ?? []).length > 0 && (
            <div className="preset-row" id={`${idPrefix}-presets`}>
              {sizePresets.map(([presetWidth, presetHeight]) => (
                <button
                  key={`${presetWidth}x${presetHeight}`}
                  id={`${idPrefix}-preset-${presetWidth}x${presetHeight}`}
                  type="button"
                  className="chip"
                  onClick={() => {
                    onWidthChange(String(presetWidth));
                    onHeightChange(String(presetHeight));
                  }}
                >
                  {presetWidth === presetHeight
                    ? `${presetWidth}²`
                    : `${presetWidth}×${presetHeight}`}
                </button>
              ))}
            </div>
          )}
        </>
      ) : (
        <>
          <div className="row" id={`${idPrefix}-auto-row`}>
            <div className="field" id={`field-${idPrefix}-target-mp`}>
              <label htmlFor={`${idPrefix}-target-mp-input`}>目标像素(百万)</label>
              <input
                id={`${idPrefix}-target-mp-input`}
                type="number"
                min={0.1}
                step={0.01}
                title="1 表示 100 万像素"
                value={targetMp}
                onChange={(e) => setTargetMp(e.target.value)}
              />
              <p className="form-hint" id={`${idPrefix}-target-mp-hint`}>
                1 = 100 万像素
              </p>
            </div>
          </div>
          {ratioPresets.length > 0 ? (
            <div className="preset-row" id={`${idPrefix}-ratio-presets`}>
              {ratioPresets.map(([rw, rh], index) => (
                <button
                  key={`${rw}-${rh}`}
                  id={`${idPrefix}-ratio-${rw}-${rh}`}
                  type="button"
                  className={`chip${index === ratioIndex ? " active" : ""}`}
                  aria-pressed={index === ratioIndex}
                  onClick={() => setRatioIndex(index)}
                >
                  {rw}:{rh}
                </button>
              ))}
            </div>
          ) : (
            <p className="form-hint" id={`${idPrefix}-ratio-empty`}>
              暂无可用比例,请先在「设置 → 生图设置 → 图片比例预设」中添加。
            </p>
          )}
          <p className="form-hint" id={`${idPrefix}-computed`}>
            {computed
              ? `输出尺寸:${computed.width} × ${computed.height}`
              : "请输入有效的目标像素"}
          </p>
        </>
      )}
    </div>
  );
}
