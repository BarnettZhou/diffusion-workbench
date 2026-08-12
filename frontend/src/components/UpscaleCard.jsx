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

export default function UpscaleCard({ value, onChange }) {
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

  const patch = (fields) => onChange({ ...value, ...fields });
  const method = value.method;
  // 放大模型默认选中第一项
  const modelIndex =
    value.modelIndex ?? (method === "upscale_model" ? (models[0]?.index ?? null) : null);
  const executedSteps = Number(value.steps) - Number(value.startStep);

  return (
    <div id="upscale-card" className="panel form">
      <h2>图片放大</h2>
      <div id="upscale-tabs" className="upscale-tabs" role="tablist" aria-label="放大方法">
        {METHOD_TABS.map((tab) => (
          <button
            key={tab.key}
            id={`upscale-tab-${tab.key}`}
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
        <div id="upscale-hint" className="upscale-hint">
          任务只输出原始尺寸图片;切换到其他方法后,任务开始时会用对应方式生成第二张放大图。
        </div>
      )}

      {method !== "none" && (
        <div className="row" id="upscale-scale-row">
          <div className="field" id="field-upscale-scale">
            <label htmlFor="upscale-scale-input">放大倍数</label>
            <input
              id="upscale-scale-input" type="number" step="0.1"
              min={LIMITS.scale.min} max={LIMITS.scale.max}
              value={value.scale}
              onChange={(e) => patch({ scale: e.target.value })}
            />
          </div>
          <div className="field" id="field-upscale-interpolation">
            <label htmlFor="upscale-interpolation-select">插值方式</label>
            <select
              id="upscale-interpolation-select"
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
          <div className="field" id="field-upscale-model">
            <label htmlFor="upscale-model-select">放大模型</label>
            <select
              id="upscale-model-select"
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
          <div className="row" id="upscale-tile-row">
            <div className="field" id="field-upscale-tile">
              <label htmlFor="upscale-tile-input">分块大小 Tile</label>
              <input
                id="upscale-tile-input" type="number" step={LIMITS.tile.multiple}
                min={LIMITS.tile.min} max={LIMITS.tile.max}
                value={value.tile}
                onChange={(e) => patch({ tile: e.target.value })}
              />
            </div>
            <div className="field" id="field-upscale-overlap">
              <label htmlFor="upscale-overlap-input">分块重叠 Overlap</label>
              <input
                id="upscale-overlap-input" type="number" min={0}
                value={value.overlap}
                onChange={(e) => patch({ overlap: e.target.value })}
              />
            </div>
          </div>
        </>
      )}

      {method === "latent_hires" && (
        <>
          <div className="row" id="upscale-steps-row">
            <div className="field" id="field-upscale-steps">
              <label htmlFor="upscale-steps-input">总步数</label>
              <input
                id="upscale-steps-input" type="number"
                min={LIMITS.steps.min} max={LIMITS.steps.max}
                value={value.steps}
                onChange={(e) => patch({ steps: e.target.value })}
              />
            </div>
            <div className="field" id="field-upscale-start-step">
              <label htmlFor="upscale-start-step-input">开始步数</label>
              <input
                id="upscale-start-step-input" type="number" min={0}
                value={value.startStep}
                onChange={(e) => patch({ startStep: e.target.value })}
              />
            </div>
          </div>
          <div
            id="upscale-executed-steps"
            className={`upscale-hint ${executedSteps < 1 ? "upscale-invalid" : ""}`}
          >
            {executedSteps >= 1
              ? `跳过前 ${value.startStep} 步,实际执行 ${executedSteps} 步`
              : "开始步数必须小于总步数"}
          </div>
          <div className="row" id="upscale-inherit-row">
            <div className="field" id="field-upscale-sampler">
              <label htmlFor="upscale-sampler-select">采样器(默认继承首次采样)</label>
              <select
                id="upscale-sampler-select"
                value={value.sampler}
                onChange={(e) => patch({ sampler: e.target.value })}
              >
                <option value="">继承首次采样</option>
                {samplerOptions.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            <div className="field" id="field-upscale-scheduler">
              <label htmlFor="upscale-scheduler-select">调度器(默认继承首次采样)</label>
              <select
                id="upscale-scheduler-select"
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
          <div className="row" id="upscale-cfg-seed-row">
            <div className="field" id="field-upscale-cfg">
              <label htmlFor="upscale-cfg-input">CFG(留空继承)</label>
              <input
                id="upscale-cfg-input" type="number" min={0} step="any"
                placeholder="继承"
                value={value.cfg}
                onChange={(e) => patch({ cfg: e.target.value })}
              />
            </div>
            <div className="field" id="field-upscale-seed">
              <label htmlFor="upscale-seed-input">Seed(留空继承任务 seed)</label>
              <input
                id="upscale-seed-input" type="number" min={0} step={1}
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
