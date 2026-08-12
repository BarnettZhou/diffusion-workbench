import { useEffect, useMemo, useState } from "react";
import LLMRecords from "./LLMRecords";
import ModelSettings from "./ModelSettings";
import VideoModelSettings from "./VideoModelSettings";
import PromptPresets from "./PromptPresets";
import { useMessage } from "./Message";
import { api } from "../api/client";

// 设置项注册表:后续新设置页只需在这里加一项。
const SETTINGS_TABS = [
  { key: "general", label: "生图设置" },
  { key: "models", label: "生图模型" },
  { key: "video", label: "视频设置" },
  { key: "videoModels", label: "视频模型" },
  { key: "llm", label: "大模型" },
  { key: "prompts", label: "提示词" },
];

export default function SettingsPage({ settings, modes, onUpdate, onResourcesChanged }) {
  const [active, setActive] = useState("general");

  return (
    <main id="settings-page">
      <nav id="settings-nav">
        {SETTINGS_TABS.map((item) => (
          <button
            key={item.key}
            id={`settings-tab-${item.key}`}
            type="button"
            className={active === item.key ? "active" : ""}
            onClick={() => setActive(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      <section id="settings-content">
        {active === "general" && (
          <GeneralSettings settings={settings} modes={modes} onUpdate={onUpdate} />
        )}
        {active === "models" && (
          <ModelSettings modes={modes} onResourcesChanged={onResourcesChanged} />
        )}
        {active === "video" && (
          <VideoSettings settings={settings} onUpdate={onUpdate} />
        )}
        {active === "videoModels" && <VideoModelSettings />}
        {active === "llm" && (
          <LLMTab settings={settings} onUpdate={onUpdate} />
        )}
        {active === "prompts" && (
          <PromptPresets
            presets={settings?.prompt_presets ?? []}
            onUpdate={onUpdate}
          />
        )}
      </section>
    </main>
  );
}

// 大模型页内部的二级 tabs:大模型设置 / 请求记录。
// 两个面板同时挂载、仅切换显隐,避免切换 tab 丢失未保存的表单内容。
const LLM_SUB_TABS = [
  { key: "settings", label: "大模型设置" },
  { key: "records", label: "请求记录" },
];

function LLMTab({ settings, onUpdate }) {
  const [sub, setSub] = useState("settings");

  return (
    <div id="settings-llm-tab">
      <nav id="llm-subnav">
        {LLM_SUB_TABS.map((item) => (
          <button
            key={item.key}
            id={`llm-subtab-${item.key}`}
            type="button"
            className={sub === item.key ? "active" : ""}
            onClick={() => setSub(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      <div hidden={sub !== "settings"}>
        <LLMSettings settings={settings} onUpdate={onUpdate} />
      </div>
      <div hidden={sub !== "records"}>
        <LLMRecords />
      </div>
    </div>
  );
}

// 后端 settings.llm 缺失或缺键时的前端兜底(服务端保存时也会补齐默认值)
const LLM_DEFAULTS = {
  interface: "ollama",
  base_url: "http://127.0.0.1:11434",
  api_key: "",
  model: "",
  system_prompt: "",
  sd_system_prompt: "",
  format_prompt: "",
  language_prompt: "",
  think: false,
  think_effort: "",
};

function LLMSettings({ settings, onUpdate }) {
  const llm = settings?.llm ?? null;
  const message = useMessage();
  const [form, setForm] = useState(() => ({ ...LLM_DEFAULTS, ...(llm ?? {}) }));
  const [saving, setSaving] = useState(false);

  // 已保存快照:与表单对比得出「未保存」状态
  const savedSnapshot = useMemo(
    () => JSON.stringify({ ...LLM_DEFAULTS, ...(llm ?? {}) }),
    [llm],
  );
  const dirty = JSON.stringify(form) !== savedSnapshot;

  // 初次进入及保存成功后,以后端返回的最新 settings.llm 为准同步表单
  useEffect(() => {
    setForm({ ...LLM_DEFAULTS, ...(llm ?? {}) });
  }, [llm]);

  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function save() {
    setSaving(true);
    try {
      await onUpdate({ llm: { ...form } });
      // 保存成功后 settings.llm 更新,dirty 随之消除
      message.success("大模型设置已保存");
    } catch (err) {
      message.error(`保存失败:${err.message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div id="settings-llm" className="panel">
      <h2>
        大模型
        {dirty && (
          <span id="llm-unsaved-badge" className="unsaved-badge">
            未保存
          </span>
        )}
      </h2>
      <div className="settings-item" id="settings-item-llm-interface">
        <label className="settings-item-label" htmlFor="llm-interface">接口类型</label>
        <p className="settings-item-desc">
          提示词快捷生成使用的大模型接口:Ollama 本地服务或 OpenAI 兼容接口。
        </p>
        <select
          id="llm-interface"
          value={form.interface}
          onChange={(e) => setField("interface", e.target.value)}
        >
          <option value="ollama">Ollama</option>
          <option value="openai">OpenAI 兼容</option>
        </select>
      </div>
      <div className="settings-item" id="settings-item-llm-base-url">
        <label className="settings-item-label" htmlFor="llm-base-url">Base URL</label>
        <input
          id="llm-base-url"
          type="text"
          value={form.base_url}
          placeholder="http://127.0.0.1:11434"
          onChange={(e) => setField("base_url", e.target.value)}
        />
      </div>
      <div className="settings-item" id="settings-item-llm-api-key">
        <label className="settings-item-label" htmlFor="llm-api-key">API Key</label>
        <p className="settings-item-desc">OpenAI 兼容接口需要;Ollama 可留空。</p>
        <input
          id="llm-api-key"
          type="password"
          value={form.api_key}
          autoComplete="off"
          onChange={(e) => setField("api_key", e.target.value)}
        />
      </div>
      <div className="settings-item" id="settings-item-llm-model">
        <label className="settings-item-label" htmlFor="llm-model">模型 ID</label>
        <input
          id="llm-model"
          type="text"
          value={form.model}
          placeholder="例如 qwen2.5:7b 或 gpt-4o-mini"
          onChange={(e) => setField("model", e.target.value)}
        />
      </div>
      <div className="settings-item" id="settings-item-llm-think">
        <label className="settings-item-label" htmlFor="llm-think">思考 Thinking</label>
        <p className="settings-item-desc">
          生成提示词一般无需思考,关闭可明显加快响应;Ollama 原生支持,
          OpenAI 兼容接口的强度通过 reasoning_effort 下发(服务端不支持时请保持关闭)。
        </p>
        <div className="settings-think-row">
          <label className="settings-toggle">
            <input
              id="llm-think"
              type="checkbox"
              checked={form.think}
              onChange={(e) => setField("think", e.target.checked)}
            />
            启用思考
          </label>
          <input
            id="llm-think-effort"
            type="text"
            value={form.think_effort}
            disabled={!form.think}
            placeholder="强度,如 low / medium / high / max"
            onChange={(e) => setField("think_effort", e.target.value)}
          />
        </div>
      </div>
      <div className="settings-item" id="settings-item-llm-system-prompt">
        <span className="settings-item-label">提示词·用户自定义</span>
        <p className="settings-item-desc">
          按目标模型分别附加画风偏好、常用质量词和提示词组织要求。
        </p>
        <div className="settings-prompt-style-grid">
          <label className="settings-prompt-style-field" htmlFor="llm-system-prompt">
            <span>FLUX 风格</span>
            <textarea
              id="llm-system-prompt"
              value={form.system_prompt}
              onChange={(e) => setField("system_prompt", e.target.value)}
            />
          </label>
          <label
            className="settings-prompt-style-field"
            htmlFor="llm-sd-system-prompt"
          >
            <span>SD 风格</span>
            <textarea
              id="llm-sd-system-prompt"
              value={form.sd_system_prompt}
              onChange={(e) => setField("sd_system_prompt", e.target.value)}
            />
          </label>
        </div>
      </div>
      <div className="settings-item" id="settings-item-llm-format-prompt">
        <label className="settings-item-label" htmlFor="llm-format-prompt">提示词·输出格式规范</label>
        <p className="settings-item-desc">
          约束大模型输出正向/负面提示词的格式,修改前建议先备份预设值。
        </p>
        <textarea
          id="llm-format-prompt"
          value={form.format_prompt}
          onChange={(e) => setField("format_prompt", e.target.value)}
        />
      </div>
      <div className="settings-item" id="settings-item-llm-language-prompt">
        <label className="settings-item-label" htmlFor="llm-language-prompt">提示词·输出语言</label>
        <p className="settings-item-desc">
          输出语言要求;其中的 {"{language}"} 占位符会被替换为弹窗中选择的目标语言(ENG / 中文)。
        </p>
        <textarea
          id="llm-language-prompt"
          value={form.language_prompt}
          onChange={(e) => setField("language_prompt", e.target.value)}
        />
      </div>
      <div className="settings-item" id="settings-item-llm-actions">
        <button
          id="llm-save-btn"
          type="button"
          className="chip"
          disabled={saving}
          onClick={save}
        >
          {saving ? "保存中……" : "保存"}
        </button>
      </div>
    </div>
  );
}

function GeneralSettings({ settings, modes, onUpdate }) {
  return (
    <div id="settings-general" className="panel">
      <h2>通用</h2>
      <div className="settings-item" id="settings-item-size-presets">
        <label className="settings-item-label">图片尺寸标签</label>
        <p className="settings-item-desc">
          工作台表单中宽高输入框下方的快捷尺寸标签。
        </p>
        <SizePresetEditor
          presets={settings?.size_presets ?? []}
          onUpdate={onUpdate}
        />
      </div>
      <SamplingDefaultsSettings
        modes={modes}
        settings={settings}
        onUpdate={onUpdate}
      />
    </div>
  );
}

function VideoSettings({ settings, onUpdate }) {
  return (
    <div id="settings-video" className="panel">
      <h2>视频设置</h2>
      <div className="settings-item" id="settings-item-wan-video-size-presets">
        <label className="settings-item-label">Wan 系列尺寸标签</label>
        <p className="settings-item-desc">
          wan 系列视频模型表单中宽高输入框下方的快捷尺寸标签;宽高必须是 16 的倍数。
        </p>
        <SizePresetEditor
          presets={settings?.wan_video_size_presets ?? []}
          field="wan_video_size_presets"
          idPrefix="wan-video-"
          multiple={16}
          onUpdate={onUpdate}
        />
      </div>
      <div className="settings-item" id="settings-item-minimax-video-size-presets">
        <label className="settings-item-label">MiniMax 系列尺寸标签</label>
        <p className="settings-item-desc">
          MiniMax 系列视频模型表单中宽高输入框下方的快捷尺寸标签;宽高必须是 32 的倍数。
        </p>
        <SizePresetEditor
          presets={settings?.minimax_video_size_presets ?? []}
          field="minimax_video_size_presets"
          idPrefix="minimax-video-"
          multiple={32}
          onUpdate={onUpdate}
        />
      </div>
    </div>
  );
}

// field 指定写入的设置项键名;idPrefix 用于区分图片/视频多处编辑器的 DOM id;
// multiple 是该系列模型要求的宽高倍数(wan 为 16,MiniMax 为 32)
function SizePresetEditor({ presets, onUpdate, field = "size_presets", idPrefix = "", multiple = 16 }) {
  const [adding, setAdding] = useState(false);
  const [width, setWidth] = useState(576);
  const [height, setHeight] = useState(576);
  const [error, setError] = useState(null);

  function removePreset(target) {
    onUpdate({
      [field]: presets.filter(
        ([w, h]) => !(w === target[0] && h === target[1]),
      ),
    });
  }

  function savePreset() {
    for (const [label, value] of [["宽度", width], ["高度", height]]) {
      if (!Number.isInteger(value) || value <= 0 || value % multiple) {
        setError(`${label}必须是正整数且为 ${multiple} 的倍数`);
        return;
      }
    }
    if (presets.some(([w, h]) => w === width && h === height)) {
      setError("该尺寸标签已存在");
      return;
    }
    setError(null);
    onUpdate({ [field]: [...presets, [width, height]] });
    setAdding(false);
  }

  return (
    <div className="preset-editor" id={`${idPrefix}size-preset-editor`}>
      <div className="preset-editor-tags">
        {presets.map(([w, h]) => (
          <span className="preset-tag" key={`${w}x${h}`} id={`${idPrefix}size-tag-${w}x${h}`}>
            {w === h ? `${w}²` : `${w}×${h}`}
            <button
              id={`${idPrefix}delete-tag-${w}x${h}`}
              type="button"
              className="preset-tag-delete"
              aria-label={`删除 ${w}×${h}`}
              onClick={() => removePreset([w, h])}
            >
              ×
            </button>
          </span>
        ))}
        {!adding && (
          <button
            id={`${idPrefix}add-preset-btn`}
            type="button"
            className="chip"
            onClick={() => {
              setError(null);
              setAdding(true);
            }}
          >
            + 添加
          </button>
        )}
      </div>

      {adding && (
        <div className="preset-add-form" id={`${idPrefix}preset-add-form`}>
          <input
            id={`${idPrefix}preset-width-input`}
            type="number"
            step={multiple}
            min={multiple}
            value={width}
            onChange={(e) => setWidth(Number(e.target.value))}
            placeholder="宽度"
          />
          <span className="preset-add-x">×</span>
          <input
            id={`${idPrefix}preset-height-input`}
            type="number"
            step={multiple}
            min={multiple}
            value={height}
            onChange={(e) => setHeight(Number(e.target.value))}
            placeholder="高度"
          />
          <button id={`${idPrefix}preset-save-btn`} type="button" className="chip" onClick={savePreset}>
            保存
          </button>
          <button
            id={`${idPrefix}preset-cancel-btn`}
            type="button"
            className="chip"
            onClick={() => setAdding(false)}
          >
            取消
          </button>
        </div>
      )}
      {error && <div id={`${idPrefix}preset-form-error`} className="form-error">{error}</div>}
    </div>
  );
}

// 各模式默认采样参数:后端 settings.sampling_defaults 缺失/缺模式时的前端兜底,
// 数值与工作台表单初始值保持一致
const SAMPLING_PARAM_FALLBACK = { steps: 8, sampler: "euler", scheduler: "simple", cfg: "1" };
const FALLBACK_SAMPLERS = ["euler", "dpmpp_2m_sde"];
const FALLBACK_SCHEDULERS = ["simple", "sgm_uniform", "beta"];
const SAMPLING_MODE_LABELS = { zit: "ZIT", zib: "ZIB", krea2: "Krea2", sdxl: "SDXL" };

function buildSamplingForm(modes, stored) {
  return Object.fromEntries(
    (modes ?? []).map((mode) => {
      const saved = stored?.[mode] ?? {};
      return [mode, {
        steps: Number.isInteger(saved.steps) ? saved.steps : SAMPLING_PARAM_FALLBACK.steps,
        sampler: saved.sampler ?? SAMPLING_PARAM_FALLBACK.sampler,
        scheduler: saved.scheduler ?? SAMPLING_PARAM_FALLBACK.scheduler,
        cfg: saved.cfg != null ? String(saved.cfg) : SAMPLING_PARAM_FALLBACK.cfg,
      }];
    }),
  );
}

function SamplingDefaultsSettings({ modes, settings, onUpdate }) {
  const message = useMessage();
  const stored = settings?.sampling_defaults ?? null;
  const [form, setForm] = useState(() => buildSamplingForm(modes, stored));
  const [saving, setSaving] = useState(false);
  const [samplerOptions, setSamplerOptions] = useState(FALLBACK_SAMPLERS);
  const [schedulerOptions, setSchedulerOptions] = useState(FALLBACK_SCHEDULERS);

  // 已保存快照:与表单对比得出「未保存」状态
  const savedSnapshot = useMemo(
    () => JSON.stringify(buildSamplingForm(modes, stored)),
    [modes, stored],
  );
  const dirty = JSON.stringify(form) !== savedSnapshot;

  // 保存成功或模式列表变化后,以后端最新 settings 为准同步表单
  useEffect(() => {
    setForm(buildSamplingForm(modes, stored));
  }, [modes, stored]);

  // 采样器/调度器选项与工作台表单同源,接口不可用时用兜底列表
  useEffect(() => {
    let cancelled = false;
    api
      .samplingOptions()
      .then((data) => {
        if (cancelled) return;
        if (data.samplers?.length) setSamplerOptions(data.samplers);
        if (data.schedulers?.length) setSchedulerOptions(data.schedulers);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  function setField(mode, key, value) {
    setForm((prev) => ({ ...prev, [mode]: { ...prev[mode], [key]: value } }));
  }

  async function save() {
    // 提交前做与工作台表单一致的校验
    for (const mode of modes ?? []) {
      const item = form[mode];
      const label = SAMPLING_MODE_LABELS[mode] ?? mode;
      if (!Number.isInteger(item.steps) || item.steps < 1 || item.steps > 100) {
        message.error(`${label}:采样步数必须是 1-100 的整数`);
        return;
      }
      if (!Number.isFinite(Number(item.cfg)) || Number(item.cfg) <= 0) {
        message.error(`${label}:CFG 必须是大于 0 的数值`);
        return;
      }
    }
    setSaving(true);
    try {
      const payload = Object.fromEntries(
        (modes ?? []).map((mode) => [
          mode,
          { ...form[mode], cfg: Number(form[mode].cfg) },
        ]),
      );
      await onUpdate({ sampling_defaults: payload });
      message.success("默认采样参数已保存");
    } catch (err) {
      message.error(`保存失败:${err.message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="settings-item" id="settings-item-sampling-defaults">
      <span className="settings-item-label">
        各模式默认采样参数
        {dirty && <span className="unsaved-badge">未保存</span>}
      </span>
      <p className="settings-item-desc">
        工作台表单「基础参数」右侧的重置按钮,会用当前模式对应的这组默认值填写表单。
      </p>
      {(modes ?? []).map((mode) => (
        <div className="sampling-defaults-row" id={`sampling-defaults-${mode}`} key={mode}>
          <span className="sampling-defaults-mode">{SAMPLING_MODE_LABELS[mode] ?? mode}</span>
          <label htmlFor={`sampling-default-${mode}-steps`}>
            步数
            <input
              id={`sampling-default-${mode}-steps`}
              type="number"
              min={1}
              max={100}
              value={form[mode]?.steps ?? ""}
              onChange={(e) => setField(mode, "steps", Number(e.target.value))}
            />
          </label>
          <label htmlFor={`sampling-default-${mode}-sampler`}>
            采样器
            <select
              id={`sampling-default-${mode}-sampler`}
              value={form[mode]?.sampler ?? ""}
              onChange={(e) => setField(mode, "sampler", e.target.value)}
            >
              {samplerOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
          <label htmlFor={`sampling-default-${mode}-scheduler`}>
            调度器
            <select
              id={`sampling-default-${mode}-scheduler`}
              value={form[mode]?.scheduler ?? ""}
              onChange={(e) => setField(mode, "scheduler", e.target.value)}
            >
              {schedulerOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
          <label htmlFor={`sampling-default-${mode}-cfg`}>
            CFG
            <input
              id={`sampling-default-${mode}-cfg`}
              type="number"
              min={0}
              step="any"
              value={form[mode]?.cfg ?? ""}
              onChange={(e) => setField(mode, "cfg", e.target.value)}
            />
          </label>
        </div>
      ))}
      <div className="sampling-defaults-actions">
        <button
          id="sampling-defaults-save-btn"
          type="button"
          className="chip"
          disabled={saving}
          onClick={save}
        >
          {saving ? "保存中……" : "保存"}
        </button>
      </div>
    </div>
  );
}
