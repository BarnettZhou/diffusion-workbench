import { useEffect, useState } from "react";
import { api } from "../api/client";

// 图片放大卡片:顶部 tabs 选择任务开始时使用的放大方法,
// 下方只展示当前方法相关的设置表单。接口不可用时用兜底选项。
const METHOD_TABS = [
  { key: "none", label: "不放大" },
  { key: "resize", label: "resize" },
  { key: "upscale_model", label: "upscale_model" },
  { key: "latent_hires", label: "latent_hires" },
];
const FALLBACK_IMAGE_INTERPOLATIONS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"];
const FALLBACK_LATENT_INTERPOLATIONS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"];
const LIMITS = {
  scale: { min: 1, max: 4 },
  tile: { min: 128, max: 1024, multiple: 32 },
  steps: { min: 1, max: 100 },
};

// 放大卡片的初始状态;数字一律存原始字符串,提交时才校验/转换
export const DEFAULT_UPSCALE = {
  method: "none", // none / resize / upscale_model / latent_hires
  scale: "2",
  interpolationImage: "lanczos",
  interpolationLatent: "bislerp",
  modelIndex: null,
  tile: "512",
  overlap: "32",
  steps: "9",
  startStep: "4",
  cfg: "", // 空字符串表示继承首次采样
  sampler: "",
  scheduler: "",
  seed: "", // 空字符串表示继承任务 seed
};

// 校验放大卡片状态;method === "none" 时无需校验。返回错误文案或 null
export function validateUpscaleValue(upscale) {
  if (upscale.method === "none") return null;
  const scale = Number(upscale.scale);
  if (!Number.isFinite(scale) || scale <= LIMITS.scale.min || scale > LIMITS.scale.max) {
    return `放大倍数必须大于 ${LIMITS.scale.min} 且不超过 ${LIMITS.scale.max}`;
  }
  if (upscale.method === "upscale_model") {
    if (upscale.modelIndex === null) return "请选择放大模型";
    const { min, max, multiple } = LIMITS.tile;
    const tile = Number(upscale.tile);
    if (!Number.isInteger(tile) || tile < min || tile > max || tile % multiple) {
      return `分块大小必须在 ${min}-${max} 之间且是 ${multiple} 的倍数`;
    }
    const overlap = Number(upscale.overlap);
    if (!Number.isInteger(overlap) || overlap < 0 || overlap >= tile / 2) {
      return "分块重叠必须大于等于 0 且小于分块大小的一半";
    }
  }
  if (upscale.method === "latent_hires") {
    const upscaleSteps = Number(upscale.steps);
    if (!Number.isInteger(upscaleSteps) || upscaleSteps < LIMITS.steps.min || upscaleSteps > LIMITS.steps.max) {
      return `总步数必须在 ${LIMITS.steps.min} 到 ${LIMITS.steps.max} 之间`;
    }
    const startStep = Number(upscale.startStep);
    if (!Number.isInteger(startStep) || startStep < 0 || startStep >= upscaleSteps) {
      return "开始步数必须大于等于 0 且小于总步数";
    }
    if (upscale.cfg !== "" && (!Number.isFinite(Number(upscale.cfg)) || Number(upscale.cfg) <= 0)) {
      return "放大 CFG 必须留空(继承)或大于 0";
    }
    if (upscale.seed !== "" && (!Number.isInteger(Number(upscale.seed)) || Number(upscale.seed) < 0)) {
      return "放大 seed 必须留空(继承)或为非负整数";
    }
  }
  return null;
}

// 把放大卡片状态映射为提交参数;不放大时返回 undefined(等价 enabled=false)
export function buildUpscalePayload(upscale) {
  if (upscale.method === "none") return undefined;
  const payload = {
    enabled: true,
    method: upscale.method,
    scale: Number(upscale.scale),
    interpolation:
      upscale.method === "latent_hires"
        ? upscale.interpolationLatent
        : upscale.interpolationImage,
  };
  if (upscale.method === "upscale_model") {
    payload.model_index = upscale.modelIndex;
    payload.tile = Number(upscale.tile);
    payload.overlap = Number(upscale.overlap);
  }
  if (upscale.method === "latent_hires") {
    payload.steps = Number(upscale.steps);
    payload.start_step = Number(upscale.startStep);
    payload.cfg = upscale.cfg === "" ? null : Number(upscale.cfg);
    payload.sampler = upscale.sampler || null;
    payload.scheduler = upscale.scheduler || null;
    payload.seed = upscale.seed === "" ? null : Number(upscale.seed);
  }
  return payload;
}

// allowedMethods 限制可选方法(编辑模式禁用 latent_hires);idPrefix 保证多表单共存时 DOM id 唯一
export default function UpscaleCard({ value, onChange, allowedMethods, idPrefix = "upscale" }) {
  const [options, setOptions] = useState(null);
  const [models, setModels] = useState([]);

  // 拉取放大选项与放大模型列表;失败时保留兜底列表
  useEffect(() => {
    let cancelled = false;
    api
      .upscaleOptions()
      .then((data) => {
        if (!cancelled) setOptions(data);
      })
      .catch(() => {});
    api
      .upscaleModels()
      .then((data) => {
        if (!cancelled) setModels(data.models ?? []);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const imageInterpolations = options?.image_interpolations ?? FALLBACK_IMAGE_INTERPOLATIONS;
  const latentInterpolations = options?.latent_interpolations ?? FALLBACK_LATENT_INTERPOLATIONS;
  const samplerOptions = options?.samplers ?? [];
  const schedulerOptions = options?.schedulers ?? [];
  const tabs = allowedMethods
    ? METHOD_TABS.filter((tab) => allowedMethods.includes(tab.key))
    : METHOD_TABS;

  const patch = (fields) => onChange({ ...value, ...fields });
  const method = value.method;
  // 放大模型默认选中第一项
  const modelIndex =
    value.modelIndex ?? (method === "upscale_model" ? (models[0]?.index ?? null) : null);
  const executedSteps = Number(value.steps) - Number(value.startStep);

  return (
    <div id={`${idPrefix}-card`} className="panel form">
      <h2>图片放大</h2>
      <div id={`${idPrefix}-tabs`} className="upscale-tabs" role="tablist" aria-label="放大方法">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            id={`${idPrefix}-tab-${tab.key}`}
            type="button"
            role="tab"
            aria-selected={method === tab.key}
            className={method === tab.key ? "active" : ""}
            onClick={() => patch({ method: tab.key })}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {method === "none" && (
        <div id={`${idPrefix}-hint`} className="upscale-hint">
          任务只输出原始尺寸图片;切换到其他方法后,任务开始时会用对应方式生成第二张放大图。
        </div>
      )}

      {method !== "none" && (
        <div className="row" id={`${idPrefix}-scale-row`}>
          <div className="field" id={`field-${idPrefix}-scale`}>
            <label htmlFor={`${idPrefix}-scale-input`}>放大倍数</label>
            <input
              id={`${idPrefix}-scale-input`} type="number" step="0.1"
              min={LIMITS.scale.min} max={LIMITS.scale.max}
              value={value.scale}
              onChange={(e) => patch({ scale: e.target.value })}
            />
          </div>
          <div className="field" id={`field-${idPrefix}-interpolation`}>
            <label htmlFor={`${idPrefix}-interpolation-select`}>插值方式</label>
            <select
              id={`${idPrefix}-interpolation-select`}
              value={method === "latent_hires" ? value.interpolationLatent : value.interpolationImage}
              onChange={(e) =>
                patch(
                  method === "latent_hires"
                    ? { interpolationLatent: e.target.value }
                    : { interpolationImage: e.target.value },
                )
              }
            >
              {(method === "latent_hires" ? latentInterpolations : imageInterpolations).map(
                (option) => (
                  <option key={option} value={option}>{option}</option>
                ),
              )}
            </select>
          </div>
        </div>
      )}

      {method === "upscale_model" && (
        <>
          <div className="field" id={`field-${idPrefix}-model`}>
            <label htmlFor={`${idPrefix}-model-select`}>放大模型</label>
            <select
              id={`${idPrefix}-model-select`}
              value={modelIndex ?? ""}
              disabled={!models.length}
              onChange={(e) => patch({ modelIndex: Number(e.target.value) })}
            >
              {models.map((item) => (
                <option key={item.index} value={item.index}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </div>
          <div className="row" id={`${idPrefix}-tile-row`}>
            <div className="field" id={`field-${idPrefix}-tile`}>
              <label htmlFor={`${idPrefix}-tile-input`}>分块大小 Tile</label>
              <input
                id={`${idPrefix}-tile-input`} type="number" step={LIMITS.tile.multiple}
                min={LIMITS.tile.min} max={LIMITS.tile.max}
                value={value.tile}
                onChange={(e) => patch({ tile: e.target.value })}
              />
            </div>
            <div className="field" id={`field-${idPrefix}-overlap`}>
              <label htmlFor={`${idPrefix}-overlap-input`}>分块重叠 Overlap</label>
              <input
                id={`${idPrefix}-overlap-input`} type="number" min={0}
                value={value.overlap}
                onChange={(e) => patch({ overlap: e.target.value })}
              />
            </div>
          </div>
        </>
      )}

      {method === "latent_hires" && (
        <>
          <div className="row" id={`${idPrefix}-steps-row`}>
            <div className="field" id={`field-${idPrefix}-steps`}>
              <label htmlFor={`${idPrefix}-steps-input`}>总步数</label>
              <input
                id={`${idPrefix}-steps-input`} type="number"
                min={LIMITS.steps.min} max={LIMITS.steps.max}
                value={value.steps}
                onChange={(e) => patch({ steps: e.target.value })}
              />
            </div>
            <div className="field" id={`field-${idPrefix}-start-step`}>
              <label htmlFor={`${idPrefix}-start-step-input`}>开始步数</label>
              <input
                id={`${idPrefix}-start-step-input`} type="number" min={0}
                value={value.startStep}
                onChange={(e) => patch({ startStep: e.target.value })}
              />
            </div>
          </div>
          <div
            id={`${idPrefix}-executed-steps`}
            className={`upscale-hint ${executedSteps < 1 ? "upscale-invalid" : ""}`}
          >
            {executedSteps >= 1
              ? `跳过前 ${value.startStep} 步,实际执行 ${executedSteps} 步`
              : "开始步数必须小于总步数"}
          </div>
          <div className="row" id={`${idPrefix}-inherit-row`}>
            <div className="field" id={`field-${idPrefix}-sampler`}>
              <label htmlFor={`${idPrefix}-sampler-select`}>采样器(默认继承首次采样)</label>
              <select
                id={`${idPrefix}-sampler-select`}
                value={value.sampler}
                onChange={(e) => patch({ sampler: e.target.value })}
              >
                <option value="">继承首次采样</option>
                {samplerOptions.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            <div className="field" id={`field-${idPrefix}-scheduler`}>
              <label htmlFor={`${idPrefix}-scheduler-select`}>调度器(默认继承首次采样)</label>
              <select
                id={`${idPrefix}-scheduler-select`}
                value={value.scheduler}
                onChange={(e) => patch({ scheduler: e.target.value })}
              >
                <option value="">继承首次采样</option>
                {schedulerOptions.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="row" id={`${idPrefix}-cfg-seed-row`}>
            <div className="field" id={`field-${idPrefix}-cfg`}>
              <label htmlFor={`${idPrefix}-cfg-input`}>CFG(留空继承)</label>
              <input
                id={`${idPrefix}-cfg-input`} type="number" min={0} step="any"
                placeholder="继承"
                value={value.cfg}
                onChange={(e) => patch({ cfg: e.target.value })}
              />
            </div>
            <div className="field" id={`field-${idPrefix}-seed`}>
              <label htmlFor={`${idPrefix}-seed-input`}>Seed(留空继承任务 seed)</label>
              <input
                id={`${idPrefix}-seed-input`} type="number" min={0} step={1}
                placeholder="继承"
                value={value.seed}
                onChange={(e) => patch({ seed: e.target.value })}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
