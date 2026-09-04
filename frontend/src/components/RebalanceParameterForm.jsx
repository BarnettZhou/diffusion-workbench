import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { normalizeUploadFile } from "./EditParameterForm";
import { useMessage } from "./Message";
import PromptPresetPicker from "./PromptPresetPicker";
import TextEncoderSelector from "./TextEncoderSelector";
import { COVER_SPACER } from "./modelCover";

// 采样器/调度器选项从 /api/v1/sampling-options 拉取,接口不可用时用兜底列表
const FALLBACK_SAMPLERS = ["euler", "dpmpp_2m_sde"];
const FALLBACK_SCHEDULERS = ["simple", "sgm_uniform", "beta"];
const LIMITS = {
  steps: { min: 1, max: 100 },
  count: { min: 1, max: 32 },
  size: { min: 256, max: 4096, multiple: 16 },
};
// token 档位的显示文案;取值与后端契约一致(low/normal/high/max)
const TIER_LABELS = { low: "低 low", normal: "标准 normal", high: "高 high", max: "最高 max" };

// Krea2 参考图重排参数表单。结构参照 EditParameterForm:
// 1-4 张参考图经受控上传拿到 id,每张图带一个 token 档位,缺省全部按 normal。
// resources 取自 App.resources["krea2"],defaults 取自 /api/v1/edit/info 的 rebalance 段。
export default function RebalanceParameterForm({
  resources,
  defaults,
  sizePresets,
  samplingDefaults,
  promptPresets,
  onSubmit,
  remoteEncoderIds = [],
}) {
  const message = useMessage();
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [modelIndex, setModelIndex] = useState(null);
  const [vaeIndex, setVaeIndex] = useState(null);
  const [textEncoderIndex, setTextEncoderIndex] = useState(null);
  const [textEncoderSource, setTextEncoderSource] = useState("local");
  const [remoteTextEncoderId, setRemoteTextEncoderId] = useState("");
  useEffect(() => { if (!remoteTextEncoderId && remoteEncoderIds.length) setRemoteTextEncoderId(remoteEncoderIds[0]); }, [remoteEncoderIds, remoteTextEncoderId]);
  // 数字输入框一律存原始字符串:允许删空和 "-1" 这类中间态,提交时才校验/转换
  const [width, setWidth] = useState("576");
  const [height, setHeight] = useState("576");
  const [steps, setSteps] = useState("8");
  const [count, setCount] = useState("1");
  const [cfg, setCfg] = useState("1");
  const [seed, setSeed] = useState("-1");
  const [sampler, setSampler] = useState("euler");
  const [scheduler, setScheduler] = useState("simple");
  const [samplerOptions, setSamplerOptions] = useState(FALLBACK_SAMPLERS);
  const [schedulerOptions, setSchedulerOptions] = useState(FALLBACK_SCHEDULERS);
  const [optionsReady, setOptionsReady] = useState(false);
  // 参考图列表:[{id, previewUrl, name, tier}];id 为受控上传返回的文件名
  const [references, setReferences] = useState([]);
  const [referenceUploading, setReferenceUploading] = useState(false);
  const referenceInputRef = useRef(null);
  // 预设提示词选择弹窗
  const [presetTarget, setPresetTarget] = useState(null);
  // prompt/负面提示词输入框高度倍率
  const [promptSize, setPromptSize] = useState(1);
  const [negativePromptSize, setNegativePromptSize] = useState(1);
  const [error, setError] = useState(null);
  // 模型封面画廊:label 右侧按钮展开/收起,封面视图 4 列(参照 ParameterForm)
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [modelCovers, setModelCovers] = useState({});
  // 已应用默认参数的 ref:避免每次 defaults/samplingDefaults 更新覆盖用户输入
  const defaultsAppliedRef = useRef(false);

  const models = resources?.models ?? [];
  const vaes = resources?.vaes ?? [];
  const textEncoders = resources?.textEncoders ?? [];
  const maxReferences = defaults?.max_reference_images ?? 4;
  const tokenTiers = defaults?.token_tiers ?? ["low", "normal", "high", "max"];
  // krea2 在 settings 里可以选用 checkpoint loader(自带 VAE),此时不显示 VAE/编码器选择器
  const usesCheckpointVae = resources?.modelLoader === "checkpoint";
  // 模型封面画廊排序:先按别名,再按模型文件名;无别名的模型按文件名参与排序
  const pickerModels = useMemo(() => {
    const compare = (a, b) => {
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
    };
    return [...models].sort(compare);
  }, [models]);

  // 展开模型画廊时拉取 krea2 的封面信息(失败时全部显示"暂无封面")
  useEffect(() => {
    if (!modelPickerOpen) return undefined;
    let cancelled = false;
    api
      .models("krea2")
      .then((body) => {
        if (cancelled) return;
        setModelCovers(
          Object.fromEntries(body.models.map((item) => [item.name, item.cover_url])),
        );
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [modelPickerOpen]);

  // 拉取采样器/调度器选项
  useEffect(() => {
    let cancelled = false;
    api
      .samplingOptions()
      .then((data) => {
        if (cancelled) return;
        if (data.samplers?.length) setSamplerOptions(data.samplers);
        if (data.schedulers?.length) setSchedulerOptions(data.schedulers);
        setOptionsReady(true);
      })
      .catch(() => setOptionsReady(true));
    return () => {
      cancelled = true;
    };
  }, []);

  // 采样选项就位后,首次进入用 krea2 默认采样参数填写
  useEffect(() => {
    if (!optionsReady || defaultsAppliedRef.current) return;
    defaultsAppliedRef.current = true;
    const kreaSampling = samplingDefaults?.krea2 ?? {};
    const nextSteps = Number(kreaSampling.steps ?? 8);
    if (
      Number.isInteger(nextSteps) &&
      nextSteps >= LIMITS.steps.min &&
      nextSteps <= LIMITS.steps.max
    ) {
      setSteps(String(nextSteps));
    }
    const nextCfg = Number(kreaSampling.cfg ?? 1);
    if (Number.isFinite(nextCfg) && nextCfg > 0) setCfg(String(kreaSampling.cfg ?? 1));
    const nextSampler = kreaSampling.sampler ?? "euler";
    if (samplerOptions.includes(nextSampler)) setSampler(nextSampler);
    const nextScheduler = kreaSampling.scheduler ?? "simple";
    if (schedulerOptions.includes(nextScheduler)) setScheduler(nextScheduler);
  }, [samplingDefaults, optionsReady, samplerOptions, schedulerOptions]);

  // 资源列表加载后,当前选择失效时回退到第一项
  useEffect(() => {
    setModelIndex((prev) =>
      models.some((item) => item.index === prev) ? prev : (models[0]?.index ?? null),
    );
    setVaeIndex((prev) =>
      vaes.some((item) => item.index === prev) ? prev : (vaes[0]?.index ?? null),
    );
    setTextEncoderIndex((prev) =>
      textEncoders.some((item) => item.index === prev) ? prev : (textEncoders[0]?.index ?? null),
    );
  }, [resources]); // eslint-disable-line react-hooks/exhaustive-deps

  // 选择文件后立即上传,本地用 object URL 预览;新图默认档位取 defaults.token_tier
  async function handleReferenceSelect(file) {
    if (!file) return;
    if (references.length >= maxReferences) {
      setError(`参考图最多 ${maxReferences} 张`);
      return;
    }
    setError(null);
    setReferenceUploading(true);
    try {
      const uploadFile = await normalizeUploadFile(file);
      const previewUrl = URL.createObjectURL(uploadFile);
      try {
        const saved = await api.uploadEditInputImage(uploadFile);
        setReferences((prev) => [
          ...prev,
          {
            id: saved.id,
            previewUrl,
            name: uploadFile.name,
            tier: defaults?.token_tier ?? "normal",
          },
        ]);
      } catch (err) {
        URL.revokeObjectURL(previewUrl);
        throw err;
      }
    } catch (err) {
      setError(err.message);
      message?.error(err.message);
    } finally {
      setReferenceUploading(false);
    }
  }

  function removeReference(index) {
    setReferences((prev) => {
      const item = prev[index];
      if (item?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(item.previewUrl);
      return prev.filter((_, i) => i !== index);
    });
  }

  function setReferenceTier(index, tier) {
    setReferences((prev) =>
      prev.map((item, i) => (i === index ? { ...item, tier } : item)),
    );
  }

  function resetToDefaults() {
    const kreaSampling = samplingDefaults?.krea2 ?? {};
    if (Number.isInteger(kreaSampling.steps)) setSteps(String(kreaSampling.steps));
    if (kreaSampling.cfg != null) setCfg(String(kreaSampling.cfg));
    if (samplerOptions.includes(kreaSampling.sampler)) setSampler(kreaSampling.sampler);
    if (schedulerOptions.includes(kreaSampling.scheduler)) setScheduler(kreaSampling.scheduler);
    setError(null);
  }

  function validate() {
    if (!models.length || (!usesCheckpointVae && (!vaes.length || !textEncoders.length))) {
      return "资源列表尚未加载";
    }
    if (modelIndex === null || (!usesCheckpointVae && (vaeIndex === null || textEncoderIndex === null))) {
      return usesCheckpointVae ? "请选择模型" : "请选择模型、VAE 和文本编码器";
    }
    if (!prompt.trim()) return "prompt 不能为空";
    if (referenceUploading) return "参考图上传中,请稍候";
    if (references.length < 1) return "请先上传至少 1 张参考图";
    if (references.length > maxReferences) return `参考图最多 ${maxReferences} 张`;
    const { multiple, min, max } = LIMITS.size;
    for (const [label, raw] of [["宽度", width], ["高度", height]]) {
      const value = Number(raw);
      if (!Number.isInteger(value) || value < min || value > max || value % multiple) {
        return `${label}必须是 ${min}-${max} 之间 ${multiple} 的倍数`;
      }
    }
    const stepsNum = Number(steps);
    if (!Number.isInteger(stepsNum) || stepsNum < LIMITS.steps.min || stepsNum > LIMITS.steps.max) {
      return `steps 必须在 ${LIMITS.steps.min} 到 ${LIMITS.steps.max} 之间`;
    }
    const countNum = Number(count);
    if (!Number.isInteger(countNum) || countNum < LIMITS.count.min || countNum > LIMITS.count.max) {
      return `批次数量必须在 ${LIMITS.count.min} 到 ${LIMITS.count.max} 之间`;
    }
    if (!Number.isFinite(Number(cfg)) || Number(cfg) <= 0) return "CFG 必须是大于 0 的数值";
    const seedNum = Number(seed);
    if (!Number.isInteger(seedNum) || seedNum < -1) return "seed 必须是 -1(随机)或非负整数";
    return null;
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const problem = validate();
    setError(problem);
    if (problem) return;
    try {
      await onSubmit({
        model_index: modelIndex,
        vae_index: usesCheckpointVae ? undefined : vaeIndex,
        text_encoder_index: textEncoderIndex,
        text_encoder_source: textEncoderSource,
        remote_text_encoder_id: textEncoderSource === "remote" ? remoteTextEncoderId : undefined,
        prompt: prompt.trim(),
        negative_prompt: negativePrompt,
        reference_image_ids: references.map((item) => item.id),
        reference_image_tokens: references.map((item) => item.tier),
        width: Number(width),
        height: Number(height),
        steps: Number(steps),
        seed: Number(seed),
        count: Number(count),
        cfg: Number(cfg),
        sampler,
        scheduler,
      });
    } catch (err) {
      setError(err.message);
      message?.error(err.message);
    }
  }

  return (
    <form id="rebalance-parameter-form" className="form-stack" onSubmit={handleSubmit}>
      <div id="rebalance-basic-params-card" className="panel form">
        <div className="card-header">
          <h2>Krea2 参考图重排参数</h2>
          <button
            id="rebalance-reset-defaults-btn"
            type="button"
            className="chip"
            title="用 krea2 默认采样参数填写"
            onClick={resetToDefaults}
          >
            重置
          </button>
        </div>

        <div className="field" id="field-rebalance-prompt">
          <div className="label-row">
            <label htmlFor="rebalance-prompt-input">Prompt</label>
            <div className="label-actions">
              <button
                id="rebalance-prompt-preset-btn"
                type="button"
                className="prompt-assist-btn"
                title="选择预设提示词"
                aria-label="选择预设提示词"
                onClick={() => setPresetTarget("positive")}
              >
                预设
              </button>
              <button
                id="rebalance-prompt-size-btn"
                type="button"
                className="prompt-assist-btn"
                title={`切换输入框高度(当前 ${promptSize}x)`}
                aria-label={`切换输入框高度(当前 ${promptSize}x)`}
                onClick={() => setPromptSize((size) => (size % 3) + 1)}
              >
                {promptSize}x
              </button>
              <button
                id="rebalance-prompt-clear-btn"
                type="button"
                className="prompt-assist-btn"
                title="清空 prompt"
                aria-label="清空 prompt"
                disabled={!prompt}
                onClick={() => setPrompt("")}
              >
                清空
              </button>
            </div>
          </div>
          <textarea
            id="rebalance-prompt-input"
            className={promptSize > 1 ? `prompt-size-${promptSize}` : undefined}
            value={prompt}
            placeholder="描述想要参考哪些特征、生成什么画面……"
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div className="field" id="field-rebalance-negative-prompt">
          <div className="label-row">
            <label htmlFor="rebalance-negative-prompt-input">负面提示词 Negative Prompt</label>
            <div className="label-actions">
              <button
                id="rebalance-negative-preset-btn"
                type="button"
                className="prompt-assist-btn"
                title="选择预设提示词"
                aria-label="选择预设提示词"
                onClick={() => setPresetTarget("negative")}
              >
                预设
              </button>
              <button
                id="rebalance-negative-size-btn"
                type="button"
                className="prompt-assist-btn"
                title={`切换输入框高度(当前 ${negativePromptSize}x)`}
                aria-label={`切换输入框高度(当前 ${negativePromptSize}x)`}
                onClick={() => setNegativePromptSize((size) => (size % 3) + 1)}
              >
                {negativePromptSize}x
              </button>
              <button
                id="rebalance-negative-clear-btn"
                type="button"
                className="prompt-assist-btn"
                title="清空负面提示词"
                aria-label="清空负面提示词"
                disabled={!negativePrompt}
                onClick={() => setNegativePrompt("")}
              >
                清空
              </button>
            </div>
          </div>
          <textarea
            id="rebalance-negative-prompt-input"
            className={negativePromptSize > 1 ? `prompt-size-${negativePromptSize}` : undefined}
            value={negativePrompt}
            placeholder="不想出现在画面中的内容(可留空)……"
            onChange={(e) => setNegativePrompt(e.target.value)}
          />
        </div>

        <div className="field" id="field-rebalance-references">
          <div className="label-row">
            <label>参考图(1-{maxReferences} 张,必选)</label>
            <div className="label-actions">
              <button
                id="rebalance-reference-add-btn"
                type="button"
                className="prompt-assist-btn"
                title="添加参考图(png/jpeg/webp,≤32MB)"
                aria-label="添加参考图"
                disabled={referenceUploading || references.length >= maxReferences}
                onClick={() => referenceInputRef.current?.click()}
              >
                添加
              </button>
            </div>
          </div>
          <input
            ref={referenceInputRef}
            id="rebalance-reference-input"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            hidden
            disabled={referenceUploading}
            onChange={(e) => {
              handleReferenceSelect(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          {references.length === 0 ? (
            <div
              id="rebalance-reference-empty"
              className="video-input-image"
              role="button"
              tabIndex={0}
              title="点击选择参考图(png/jpeg/webp,≤32MB)"
              onClick={() => referenceInputRef.current?.click()}
              onKeyDown={(e) => e.key === "Enter" && referenceInputRef.current?.click()}
            >
              <div className="video-input-image-empty">
                {referenceUploading ? "上传中……" : "点击上传参考图"}
              </div>
            </div>
          ) : (
            <div id="rebalance-reference-grid" className="reference-grid">
              {references.map((item, index) => (
                <div
                  key={item.id}
                  id={`rebalance-reference-${index + 1}`}
                  className="reference-item"
                >
                  <img src={item.previewUrl} alt={item.name} />
                  <div className="reference-item-bar">
                    <select
                      id={`rebalance-reference-tier-${index + 1}`}
                      title="该参考图占用的 token 档位"
                      value={item.tier}
                      onChange={(e) => setReferenceTier(index, e.target.value)}
                    >
                      {tokenTiers.map((tier) => (
                        <option key={tier} value={tier}>
                          {TIER_LABELS[tier] ?? tier}
                        </option>
                      ))}
                    </select>
                    <button
                      id={`rebalance-reference-remove-${index + 1}`}
                      type="button"
                      className="prompt-assist-btn"
                      title="移除该参考图"
                      aria-label="移除该参考图"
                      onClick={() => removeReference(index)}
                    >
                      移除
                    </button>
                  </div>
                </div>
              ))}
              {references.length < maxReferences && (
                <button
                  id="rebalance-reference-add-card"
                  type="button"
                  className="reference-add-card"
                  title="添加参考图"
                  disabled={referenceUploading}
                  onClick={() => referenceInputRef.current?.click()}
                >
                  {referenceUploading ? "上传中……" : "+"}
                </button>
              )}
            </div>
          )}
          {references.length > 0 && (
            <p className="form-hint" id="rebalance-reference-hint">
              token 档位越高,该参考图占用的上下文越多、影响越强。
            </p>
          )}
        </div>

        <div className="field" id="field-rebalance-model">
          <div className="label-row">
            <label htmlFor="rebalance-model-select">模型 Checkpoint</label>
            <button
              id="rebalance-model-picker-toggle"
              type="button"
              className="prompt-assist-btn"
              title={modelPickerOpen ? "收起模型画廊" : "查看模型画廊"}
              aria-label={modelPickerOpen ? "收起模型画廊" : "查看模型画廊"}
              aria-expanded={modelPickerOpen}
              onClick={() => setModelPickerOpen((open) => !open)}
            >
              ▦
            </button>
          </div>
          <select
            id="rebalance-model-select"
            value={modelIndex ?? ""}
            disabled={!models.length}
            onChange={(e) => setModelIndex(Number(e.target.value))}
          >
            {models.map((item) => (
              <option key={item.index} value={item.index}>
                {item.display_name}
              </option>
            ))}
          </select>
        </div>

        {modelPickerOpen && (
          <div id="rebalance-model-picker" className="model-picker">
            {pickerModels.map((item) => {
              const cover = modelCovers[item.name];
              return (
                <button
                  key={item.index}
                  id={`rebalance-model-pick-${item.index}`}
                  type="button"
                  className={`model-card picker-card ${modelIndex === item.index ? "active" : ""}`}
                  title={item.display_name}
                  onClick={() => setModelIndex(item.index)}
                >
                  <div className="model-cover">
                    {cover ? (
                      <img src={cover} alt={item.name} loading="lazy" />
                    ) : (
                      <div className="model-cover-empty">
                        <img className="model-cover-spacer" src={COVER_SPACER} alt="" />
                        <span>暂无封面</span>
                      </div>
                    )}
                  </div>
                  <div className="model-info">
                    <div className="model-name">{item.display_name}</div>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {!usesCheckpointVae && (
          <div className="field" id="field-rebalance-vae">
            <label htmlFor="rebalance-vae-select">VAE</label>
            <select
              id="rebalance-vae-select"
              value={vaeIndex ?? ""}
              disabled={!vaes.length}
              onChange={(e) => setVaeIndex(Number(e.target.value))}
            >
              {vaes.map((item) => (
                <option key={item.index} value={item.index}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </div>
        )}

        {!usesCheckpointVae && (
          <TextEncoderSelector idPrefix="rebalance" localEncoders={textEncoders} localIndex={textEncoderIndex} onLocalIndexChange={setTextEncoderIndex} remoteIds={remoteEncoderIds} source={textEncoderSource} onSourceChange={setTextEncoderSource} remoteId={remoteTextEncoderId} onRemoteIdChange={setRemoteTextEncoderId} />
        )}

        <div className="row" id="rebalance-size-row">
          <div className="field" id="field-rebalance-width">
            <label htmlFor="rebalance-width-input">宽度 (px)</label>
            <input
              id="rebalance-width-input" type="number" step={LIMITS.size.multiple}
              min={LIMITS.size.min} max={LIMITS.size.max}
              value={width} onChange={(e) => setWidth(e.target.value)}
            />
          </div>
          <div className="field" id="field-rebalance-height">
            <label htmlFor="rebalance-height-input">高度 (px)</label>
            <input
              id="rebalance-height-input" type="number" step={LIMITS.size.multiple}
              min={LIMITS.size.min} max={LIMITS.size.max}
              value={height} onChange={(e) => setHeight(e.target.value)}
            />
          </div>
        </div>

        {(sizePresets ?? []).length > 0 && (
          <div className="preset-row" id="rebalance-size-presets">
            {sizePresets.map(([presetWidth, presetHeight]) => (
              <button
                key={`${presetWidth}x${presetHeight}`}
                id={`rebalance-preset-${presetWidth}x${presetHeight}`}
                type="button"
                className="chip"
                onClick={() => {
                  setWidth(String(presetWidth));
                  setHeight(String(presetHeight));
                }}
              >
                {presetWidth === presetHeight
                  ? `${presetWidth}²`
                  : `${presetWidth}×${presetHeight}`}
              </button>
            ))}
          </div>
        )}

        <div className="row" id="rebalance-steps-count-row">
          <div className="field" id="field-rebalance-steps">
            <label htmlFor="rebalance-steps-input">采样步数 Steps</label>
            <input
              id="rebalance-steps-input" type="number"
              min={LIMITS.steps.min} max={LIMITS.steps.max}
              value={steps} onChange={(e) => setSteps(e.target.value)}
            />
          </div>
          <div className="field" id="field-rebalance-count">
            <label htmlFor="rebalance-count-input">批次数量</label>
            <input
              id="rebalance-count-input" type="number"
              min={LIMITS.count.min} max={LIMITS.count.max}
              value={count} onChange={(e) => setCount(e.target.value)}
            />
          </div>
        </div>

        <div className="row" id="rebalance-fixed-params-row">
          <div className="field" id="field-rebalance-sampler">
            <label htmlFor="rebalance-sampler-select">采样器 Sampler</label>
            <select
              id="rebalance-sampler-select"
              value={sampler}
              onChange={(e) => setSampler(e.target.value)}
            >
              {samplerOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          <div className="field" id="field-rebalance-scheduler">
            <label htmlFor="rebalance-scheduler-select">调度器 Scheduler</label>
            <select
              id="rebalance-scheduler-select"
              value={scheduler}
              onChange={(e) => setScheduler(e.target.value)}
            >
              {schedulerOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          <div className="field" id="field-rebalance-cfg">
            <label htmlFor="rebalance-cfg-input">CFG</label>
            <input
              id="rebalance-cfg-input" type="number" min={0} step="any"
              value={cfg} onChange={(e) => setCfg(e.target.value)}
            />
          </div>
          <div className="field" id="field-rebalance-seed">
            <label htmlFor="rebalance-seed-input">种子 Seed</label>
            <input
              id="rebalance-seed-input" type="number" min={-1} step={1}
              title="-1 表示随机"
              value={seed} onChange={(e) => setSeed(e.target.value)}
            />
          </div>
        </div>
      </div>

      {error && <div id="rebalance-form-error" className="form-error">{error}</div>}

      {presetTarget && (
        <PromptPresetPicker
          kind={presetTarget}
          presets={promptPresets ?? []}
          onClose={() => setPresetTarget(null)}
          onSelect={(preset) => {
            if (presetTarget === "positive") setPrompt(preset.text);
            else setNegativePrompt(preset.text);
            setPresetTarget(null);
          }}
        />
      )}
    </form>
  );
}
