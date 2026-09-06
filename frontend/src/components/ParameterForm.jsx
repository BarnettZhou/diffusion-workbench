import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import PromptAssistModal from "./PromptAssistModal";
import PromptPresetPicker from "./PromptPresetPicker";
import UpscaleCard, { DEFAULT_UPSCALE, validateUpscaleValue, buildUpscalePayload } from "./UpscaleCard";
import SizeInputCard from "./SizeInputCard";
import TextEncoderSelector from "./TextEncoderSelector";
import LoraCard from "./LoraCard";
import { COVER_SPACER } from "./modelCover";

// 采样器/调度器选项从 /api/v1/sampling-options 拉取,接口不可用时用兜底列表
const FALLBACK_SAMPLERS = ["euler", "dpmpp_2m_sde"];
const FALLBACK_SCHEDULERS = ["simple", "sgm_uniform", "beta"];
const LIMITS = {
  steps: { min: 1, max: 100 },
  count: { min: 1, max: 32 },
  size: { min: 256, max: 4096, multiple: 16 },
};
// 支持可选 LoRA 的文生图模式(与服务端 domain 校验一致)
const LORA_MODES = new Set(["krea2", "zit"]);

export default function ParameterForm({ mode, onModeChange, resources, sizePresets, ratioPresets = [], promptPresets, prefill, samplingDefaults, llmSettings, onSelectLlmModel, onSubmit, remoteEncoderIds = [], privacyMode = false }) {
  // 模型/VAE 选择按 mode 分开保存,切换 tab 恢复各 mode 上次选中项
  const [selections, setSelections] = useState({});
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [textEncoderSource, setTextEncoderSource] = useState("local");
  const [remoteTextEncoderId, setRemoteTextEncoderId] = useState(remoteEncoderIds[0] ?? "");
  useEffect(() => { if (!remoteTextEncoderId && remoteEncoderIds.length) setRemoteTextEncoderId(remoteEncoderIds[0]); }, [remoteEncoderIds, remoteTextEncoderId]);
  // 数字输入框一律存原始字符串:允许删空和 "-1" 这类中间态,提交时才校验/转换
  const [width, setWidth] = useState("576");
  const [height, setHeight] = useState("576");
  const [steps, setSteps] = useState("8");
  const [count, setCount] = useState("1");
  const [cfg, setCfg] = useState("1");
  // seed:-1 表示随机;相册"发送到工作台"时替换为图片的 seed
  const [seed, setSeed] = useState("-1");
  const [sampler, setSampler] = useState("euler");
  const [scheduler, setScheduler] = useState("simple");
  const [samplerOptions, setSamplerOptions] = useState(FALLBACK_SAMPLERS);
  const [schedulerOptions, setSchedulerOptions] = useState(FALLBACK_SCHEDULERS);
  // 采样器/调度器选项是否已拉取完成(首次应用模式默认参数的前置条件)
  const [optionsReady, setOptionsReady] = useState(false);
  const [upscale, setUpscale] = useState(DEFAULT_UPSCALE);
  // krea2 / zit 模式可选 LoRA:默认关闭;loras 每项为 {index, strength 字符串}
  const [loraEnabled, setLoraEnabled] = useState(false);
  const [loras, setLoras] = useState([]);
  const [error, setError] = useState(null);
  // 模型封面选择器:label 右侧按钮展开/收起,封面视图 4 列
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [modelCovers, setModelCovers] = useState({});
  // 快捷提示词:对话式弹窗(见 PromptAssistModal)
  const [assistOpen, setAssistOpen] = useState(false);
  // 预设提示词选择弹窗:null 关闭;"positive"/"negative" 表示替换目标输入框
  const [presetTarget, setPresetTarget] = useState(null);
  // prompt 输入框高度倍率,循环 1x → 2x → 3x
  const [promptSize, setPromptSize] = useState(1);
  // 负面提示词输入框高度倍率,循环 1x → 2x → 3x
  const [negativePromptSize, setNegativePromptSize] = useState(1);

  const models = resources?.[mode]?.models ?? [];
  const vaes = resources?.[mode]?.vaes ?? [];
  const textEncoders = resources?.[mode]?.textEncoders ?? [];
  const loraOptions = resources?.[mode]?.loras ?? [];
  // 模型下拉(#model-select)与封面选择器共用的排序:先按别名(重命名后的名称),
  // 再按模型文件名;无别名的模型按文件名参与排序
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
  const usesCheckpointVae = resources?.[mode]?.modelLoader === "checkpoint";
  const availableModes = Object.keys(resources ?? {});
  const modelIndex = selections[mode]?.modelIndex ?? null;
  const vaeIndex = selections[mode]?.vaeIndex ?? null;
  const textEncoderIndex = selections[mode]?.textEncoderIndex ?? null;
  // 相册"发送到工作台"带过来的模型/VAE 文件名,等资源列表就位后匹配成 index
  const pendingNamesRef = useRef(null);

  // 拉取全部可用采样器/调度器(44 + 9),失败时保留兜底列表
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

  // 展开模型选择器时拉取该模式的封面信息(失败时全部显示"暂无封面")
  useEffect(() => {
    if (!modelPickerOpen) return undefined;
    let cancelled = false;
    api
      .models(mode)
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
  }, [modelPickerOpen, mode]);

  // 应用相册预填:非法字段保留当前值,模式非法时回退 zit
  useEffect(() => {
    if (!prefill) return;
    const targetMode = availableModes.includes(prefill.mode)
      ? prefill.mode
      : (availableModes[0] ?? "zit");
    pendingNamesRef.current = {
      mode: targetMode,
      modelName: prefill.modelName,
      vaeName: prefill.vaeName,
    };
    onModeChange(targetMode);
    setPrompt(prefill.prompt ?? "");
    setNegativePrompt(prefill.negativePrompt ?? "");
    const { multiple, min, max } = LIMITS.size;
    for (const [value, setter] of [[prefill.width, setWidth], [prefill.height, setHeight]]) {
      if (Number.isInteger(value) && value >= min && value <= max && value % multiple === 0) {
        setter(String(value));
      }
    }
    if (
      Number.isInteger(prefill.steps) &&
      prefill.steps >= LIMITS.steps.min &&
      prefill.steps <= LIMITS.steps.max
    ) {
      setSteps(String(prefill.steps));
    }
    if (Number.isFinite(Number(prefill.cfg)) && Number(prefill.cfg) > 0) {
      setCfg(String(prefill.cfg));
    }
    // 图片的 seed 替换默认的 -1(随机);非法值保留当前值
    const prefillSeed = Number(prefill.seed);
    if (Number.isInteger(prefillSeed) && prefillSeed >= -1) {
      setSeed(String(prefillSeed));
    }
    if (samplerOptions.includes(prefill.sampler)) setSampler(prefill.sampler);
    if (schedulerOptions.includes(prefill.scheduler)) setScheduler(prefill.scheduler);
    setError(null);
  }, [prefill]);

  function setModelIndex(index) {
    setSelections((prev) => ({ ...prev, [mode]: { ...prev[mode], modelIndex: index } }));
  }
  function setVaeIndex(index) {
    setSelections((prev) => ({ ...prev, [mode]: { ...prev[mode], vaeIndex: index } }));
  }
  function setTextEncoderIndex(index) {
    setSelections((prev) => ({ ...prev, [mode]: { ...prev[mode], textEncoderIndex: index } }));
  }

  // 资源列表加载或模式切换后,该 mode 无有效选择时默认选中第一项;
  // 有相册预填的待匹配文件名时优先按文件名匹配
  useEffect(() => {
    setSelections((prev) => {
      const pending = pendingNamesRef.current;
      if (pending && pending.mode === mode) {
        const model = models.find((item) => item.name === pending.modelName);
        const vae = vaes.find((item) => item.name === pending.vaeName);
        if (model || vae) {
          pendingNamesRef.current = null;
          return {
            ...prev,
            [mode]: {
              modelIndex: model?.index ?? (models[0]?.index ?? null),
              vaeIndex: vae?.index ?? (vaes[0]?.index ?? null),
              textEncoderIndex: textEncoders[0]?.index ?? null,
            },
          };
        }
        // 列表已加载但文件名匹配不上(资源已删除/改名),放弃匹配走默认逻辑
        if (models.length && (usesCheckpointVae || vaes.length)) pendingNamesRef.current = null;
      }
      const current = prev[mode] ?? {};
      const modelValid = models.some((item) => item.index === current.modelIndex);
      const vaeValid = vaes.some((item) => item.index === current.vaeIndex);
      const teValid = textEncoders.some((item) => item.index === current.textEncoderIndex);
      if (modelValid && (usesCheckpointVae || (vaeValid && teValid))) return prev;
      return {
        ...prev,
        [mode]: {
          modelIndex: modelValid ? current.modelIndex : (models[0]?.index ?? null),
          vaeIndex: vaeValid ? current.vaeIndex : (vaes[0]?.index ?? null),
          textEncoderIndex: teValid ? current.textEncoderIndex : (textEncoders[0]?.index ?? null),
        },
      };
    });
  }, [mode, resources]); // eslint-disable-line react-hooks/exhaustive-deps

  // 用当前模式在设置页配置的默认采样参数填写表单;缺项回落内置默认值
  function resetToDefaults() {
    const defaults = samplingDefaults?.[mode] ?? {};
    const nextSteps = Number(defaults.steps ?? 8);
    if (
      Number.isInteger(nextSteps) &&
      nextSteps >= LIMITS.steps.min &&
      nextSteps <= LIMITS.steps.max
    ) {
      setSteps(String(nextSteps));
    }
    const nextCfg = Number(defaults.cfg ?? 1);
    if (Number.isFinite(nextCfg) && nextCfg > 0) setCfg(String(defaults.cfg ?? 1));
    const nextSampler = defaults.sampler ?? "euler";
    if (samplerOptions.includes(nextSampler)) setSampler(nextSampler);
    const nextScheduler = defaults.scheduler ?? "simple";
    if (schedulerOptions.includes(nextScheduler)) setScheduler(nextScheduler);
    setError(null);
  }

  // 首次进入某个模式 tab 时,用该模式在设置页配置的默认采样参数填写一次;
  // 之后切模式/手动修改不再覆盖,相册预填优先于默认参数
  const defaultsAppliedRef = useRef(new Set());
  useEffect(() => {
    if (!samplingDefaults || !optionsReady) return;
    if (defaultsAppliedRef.current.has(mode)) return;
    defaultsAppliedRef.current.add(mode);
    if (prefill && prefill.mode === mode) return;
    resetToDefaults();
  }, [mode, samplingDefaults, optionsReady, prefill]);

  function validate() {
    const { multiple, min, max } = LIMITS.size;
    if (!models.length || (!usesCheckpointVae && (!vaes.length || !textEncoders.length))) {
      return "资源列表尚未加载";
    }
    if (modelIndex === null || (!usesCheckpointVae && (vaeIndex === null || textEncoderIndex === null))) {
      return usesCheckpointVae ? "请选择模型" : "请选择模型、VAE 和文本编码器";
    }
    if (!prompt.trim()) return "prompt 不能为空";
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
    if (LORA_MODES.has(mode) && loraEnabled) {
      for (const [slot, item] of loras.entries()) {
        if (item.index === null) return `请选择 LoRA ${slot + 1} 的模型`;
        const strength = Number(item.strength);
        if (!Number.isFinite(strength) || strength < 0 || strength > 2) {
          return `LoRA ${slot + 1} 强度必须是 0 到 2 之间的数值`;
        }
      }
    }
    return validateUpscaleValue(upscale);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const problem = validate();
    setError(problem);
    if (problem) return;
    try {
      await onSubmit({
        mode,
        model_index: modelIndex,
        vae_index: usesCheckpointVae ? undefined : vaeIndex,
        text_encoder_index: usesCheckpointVae ? undefined : textEncoderIndex,
        prompt: prompt.trim(),
        negative_prompt: negativePrompt,
        width: Number(width),
        height: Number(height),
        steps: Number(steps),
        seed: Number(seed),
        count: Number(count),
        cfg: Number(cfg),
        sampler,
        scheduler,
        text_encoder_source: textEncoderSource,
        remote_text_encoder_id: textEncoderSource === "remote" ? remoteTextEncoderId : undefined,
        loras:
          LORA_MODES.has(mode) && loraEnabled && loras.length
            ? loras.map((item) => ({ index: item.index, strength: Number(item.strength) }))
            : undefined,
        upscale: buildUpscalePayload(upscale),
      });
    } catch (err) {
      setError(err.message);
    }
  }

  function handleAssistUse(positive, negative) {
    setPrompt(positive);
    setNegativePrompt(negative);
    setAssistOpen(false);
  }

  return (
    <form id="parameter-form" className="form-stack" onSubmit={handleSubmit}>
      <div id="basic-params-card" className="panel form">
        <div className="card-header">
          <h2>基础参数</h2>
          <button
            id="reset-defaults-btn"
            type="button"
            className="chip"
            title="用当前模式的默认采样参数填写"
            onClick={resetToDefaults}
          >
            重置
          </button>
        </div>
      <div className="field" id="field-prompt">
        <div className="label-row">
          <label htmlFor="prompt-input">Prompt</label>
          <div className="label-actions">
            <button
              id="prompt-preset-btn"
              type="button"
              className="prompt-assist-btn"
              title="选择预设提示词"
              aria-label="选择预设提示词"
              onClick={() => setPresetTarget("positive")}
            >
              预设
            </button>
            <button
              id="prompt-size-btn"
              type="button"
              className="prompt-assist-btn"
              title={`切换输入框高度(当前 ${promptSize}x)`}
              aria-label={`切换输入框高度(当前 ${promptSize}x)`}
              onClick={() => setPromptSize((size) => (size % 3) + 1)}
            >
              {promptSize}x
            </button>
            <button
              id="prompt-clear-btn"
              type="button"
              className="prompt-assist-btn"
              title="清空 prompt"
              aria-label="清空 prompt"
              disabled={!prompt}
              onClick={() => setPrompt("")}
            >
              清空
            </button>
            <button
              id="prompt-assist-btn"
              type="button"
              className="prompt-assist-btn"
              title="快捷生成提示词"
              aria-label="快捷生成提示词"
              onClick={() => setAssistOpen(true)}
            >
              ⚡
            </button>
          </div>
        </div>
        <textarea
          id="prompt-input"
          className={promptSize > 1 ? `prompt-size-${promptSize}` : undefined}
          value={prompt}
          placeholder="描述你想要生成的画面……"
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>

      <div className="field" id="field-negative-prompt">
        <div className="label-row">
          <label htmlFor="negative-prompt-input">负面提示词 Negative Prompt</label>
          <div className="label-actions">
            <button
              id="negative-preset-btn"
              type="button"
              className="prompt-assist-btn"
              title="选择预设提示词"
              aria-label="选择预设提示词"
              onClick={() => setPresetTarget("negative")}
            >
              预设
            </button>
            <button
              id="negative-size-btn"
              type="button"
              className="prompt-assist-btn"
              title={`切换输入框高度(当前 ${negativePromptSize}x)`}
              aria-label={`切换输入框高度(当前 ${negativePromptSize}x)`}
              onClick={() => setNegativePromptSize((size) => (size % 3) + 1)}
            >
              {negativePromptSize}x
            </button>
            <button
              id="negative-clear-btn"
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
          id="negative-prompt-input"
          className={negativePromptSize > 1 ? `prompt-size-${negativePromptSize}` : undefined}
          value={negativePrompt}
          placeholder="不想出现在画面中的内容(可留空)……"
          onChange={(e) => setNegativePrompt(e.target.value)}
        />
      </div>

      <div className="field" id="field-model">
        <div className="label-row">
          <label htmlFor="model-select">模型 Checkpoint</label>
          <button
            id="model-picker-toggle"
            type="button"
            className="prompt-assist-btn"
            title={modelPickerOpen ? "收起模型列表" : "查看模型列表"}
            aria-label={modelPickerOpen ? "收起模型列表" : "查看模型列表"}
            aria-expanded={modelPickerOpen}
            onClick={() => setModelPickerOpen((open) => !open)}
          >
            ▦
          </button>
        </div>
        <select
          id="model-select"
          value={modelIndex ?? ""}
          disabled={!models.length}
          onChange={(e) => setModelIndex(Number(e.target.value))}
        >
          {pickerModels.map((item) => (
            <option key={item.index} value={item.index}>
              {item.display_name}
            </option>
          ))}
        </select>
      </div>

      {modelPickerOpen && (
        <div id="model-picker" className="model-picker">
          {pickerModels.map((item) => {
            const cover = modelCovers[item.name];
            return (
              <button
                key={item.index}
                id={`model-pick-${item.index}`}
                type="button"
                className={`model-card picker-card ${modelIndex === item.index ? "active" : ""}`}
                title={item.display_name}
                onClick={() => setModelIndex(item.index)}
              >
                <div className="model-cover">
                  {privacyMode ? (
                    <div className="model-cover-privacy" aria-label="隐私模式">
                      <img className="model-cover-spacer" src={COVER_SPACER} alt="" />
                      <span>隐私模式</span>
                    </div>
                  ) : cover ? (
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

      {!usesCheckpointVae && <div className="field" id="field-vae">
        <label htmlFor="vae-select">VAE</label>
        <select
          id="vae-select"
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
      </div>}

      {!usesCheckpointVae && <TextEncoderSelector idPrefix="" localEncoders={textEncoders} localIndex={textEncoderIndex} onLocalIndexChange={setTextEncoderIndex} remoteIds={remoteEncoderIds} source={textEncoderSource} onSourceChange={setTextEncoderSource} remoteId={remoteTextEncoderId} onRemoteIdChange={setRemoteTextEncoderId} />}

      <SizeInputCard
        idPrefix="size"
        embedded
        width={width}
        height={height}
        onWidthChange={setWidth}
        onHeightChange={setHeight}
        sizePresets={sizePresets ?? []}
        ratioPresets={ratioPresets ?? []}
        limits={LIMITS.size}
      />

      {LORA_MODES.has(mode) && (
        <LoraCard
          enabled={loraEnabled}
          onEnabledChange={setLoraEnabled}
          loras={loras}
          onChange={setLoras}
          options={loraOptions}
        />
      )}

      <div className="row" id="steps-count-row">
        <div className="field" id="field-steps">
          <label htmlFor="steps-input">采样步数 Steps</label>
          <input
            id="steps-input" type="number"
            min={LIMITS.steps.min} max={LIMITS.steps.max}
            value={steps} onChange={(e) => setSteps(e.target.value)}
          />
        </div>
        <div className="field" id="field-count">
          <label htmlFor="count-input">批次数量</label>
          <input
            id="count-input" type="number"
            min={LIMITS.count.min} max={LIMITS.count.max}
            value={count} onChange={(e) => setCount(e.target.value)}
          />
        </div>
      </div>

      <div className="row" id="fixed-params-row">
        <div className="field" id="field-sampler">
          <label htmlFor="sampler-select">采样器 Sampler</label>
          <select
            id="sampler-select"
            value={sampler}
            onChange={(e) => setSampler(e.target.value)}
          >
            {samplerOptions.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div className="field" id="field-scheduler">
          <label htmlFor="scheduler-select">调度器 Scheduler</label>
          <select
            id="scheduler-select"
            value={scheduler}
            onChange={(e) => setScheduler(e.target.value)}
          >
            {schedulerOptions.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div className="field" id="field-cfg">
          <label htmlFor="cfg-input">CFG</label>
          <input
            id="cfg-input" type="number" min={0} step="any"
            value={cfg} onChange={(e) => setCfg(e.target.value)}
          />
        </div>
        <div className="field" id="field-seed">
          <label htmlFor="seed-input">种子 Seed</label>
          <input
            id="seed-input" type="number" min={-1} step={1}
            title="-1 表示随机"
            value={seed} onChange={(e) => setSeed(e.target.value)}
          />
        </div>
      </div>
      </div>

      <UpscaleCard value={upscale} onChange={setUpscale} />

      {error && <div id="form-error" className="form-error">{error}</div>}

      <PromptAssistModal
        open={assistOpen}
        onClose={() => setAssistOpen(false)}
        onUse={handleAssistUse}
        llmSettings={llmSettings}
        onSelectModel={onSelectLlmModel}
      />
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
