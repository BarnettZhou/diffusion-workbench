import { useEffect, useMemo, useState } from "react";
import LLMRecords from "./LLMRecords";
import CaptionRecords from "./CaptionRecords";
import ModelSettings from "./ModelSettings";
import VideoModelSettings from "./VideoModelSettings";
import PromptPresets from "./PromptPresets";
import RemoteModelSelect from "./RemoteModelSelect";
import { useMessage } from "./Message";
import { api } from "../api/client";

// 设置项注册表:后续新设置页只需在这里加一项。
const SETTINGS_TABS = [
  { key: "general", label: "生图设置" },
  { key: "models", label: "生图模型" },
  { key: "video", label: "视频设置" },
  { key: "videoModels", label: "视频模型" },
  { key: "llm", label: "大模型" },
  { key: "caption", label: "图片反推" },
  { key: "prompts", label: "提示词" },
];

export default function SettingsPage({ settings, modes, onUpdate, onResourcesChanged, privacyMode = false }) {
  const [active, setActive] = useState("general");
  // 子页首次激活才挂载,之后保持挂载:避免打开设置页时全部面板并发拉取,
  // 同时保住已访问子页未保存的表单内容
  const [mounted, setMounted] = useState(["general"]);
  useEffect(() => {
    setMounted((prev) => (prev.includes(active) ? prev : [...prev, active]));
  }, [active]);
  const panelClass = (key) => (active === key ? "tab-contents" : "tab-hidden");

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
        {mounted.includes("general") && (
          <div className={panelClass("general")}>
            <GeneralSettings settings={settings} modes={modes} onUpdate={onUpdate} />
          </div>
        )}
        {mounted.includes("models") && (
          <div className={panelClass("models")}>
            <ModelSettings
              modes={modes}
              onResourcesChanged={onResourcesChanged}
              privacyMode={privacyMode}
            />
          </div>
        )}
        {mounted.includes("video") && (
          <div className={panelClass("video")}>
            <VideoSettings settings={settings} onUpdate={onUpdate} />
          </div>
        )}
        {mounted.includes("videoModels") && (
          <div className={panelClass("videoModels")}>
            <VideoModelSettings privacyMode={privacyMode} />
          </div>
        )}
        {mounted.includes("llm") && (
          <div className={panelClass("llm")}>
            <LLMTab settings={settings} onUpdate={onUpdate} />
          </div>
        )}
        {mounted.includes("caption") && (
          <div className={panelClass("caption")}>
            <CaptionApiTab settings={settings} onUpdate={onUpdate} />
          </div>
        )}
        {mounted.includes("prompts") && (
          <div className={panelClass("prompts")}>
            <PromptPresets
              presets={settings?.prompt_presets ?? []}
              onUpdate={onUpdate}
            />
          </div>
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

// 后端 settings.llm 缺失或缺键时的前端兜底(服务端保存时也会补齐默认值)。
// 多端点结构:共享字段(系统提示词 / 思考等)+ endpoints 列表 + selected 指针。
const LLM_DEFAULTS = {
  endpoints: [],
  selected: { endpoint_id: "", model: "" },
  system_prompt: "",
  sd_system_prompt: "",
  format_prompt: "",
  language_prompt: "",
  think: false,
  think_effort: "",
};

// 端点卡片编辑器:渲染一组端点(name/interface/base_url/api_key/models),
// 「获取模型列表」按钮调 api.remoteModels 拉取该端点支持的模型列表;
// onChange 回传新的 endpoints 数组。
function EndpointListEditor({ endpoints, onChange, idPrefix }) {
  const message = useMessage();
  const list = Array.isArray(endpoints) ? endpoints : [];
  // 每张卡片的「获取模型」loading 状态:key=端点 id,true 时按钮 disabled
  const [loading, setLoading] = useState({});
  // 模型输入框(手动添加)的草稿:key=端点 id
  const [modelDraft, setModelDraft] = useState({});

  function update(next) {
    onChange?.(next);
  }

  function addEndpoint() {
    const localId = `local-${Date.now()}`;
    update([
      ...list,
      {
        id: localId,
        name: "",
        interface: "ollama",
        base_url: "http://127.0.0.1:11434",
        api_key: "",
        models: [],
      },
    ]);
  }

  function removeEndpoint(id) {
    update(list.filter((item) => item.id !== id));
  }

  function patchEndpoint(id, patch) {
    update(list.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }

  async function fetchModels(endpoint) {
    if (!endpoint.base_url?.trim()) {
      message.warning("请先填写 Base URL");
      return;
    }
    setLoading((prev) => ({ ...prev, [endpoint.id]: true }));
    try {
      const result = await api.remoteModels({
        interface: endpoint.interface,
        base_url: endpoint.base_url,
        api_key: endpoint.api_key,
      });
      const models = Array.isArray(result.models) ? result.models : [];
      patchEndpoint(endpoint.id, { models });
      if (models.length === 0) {
        message.warning("连接成功,但服务返回了 0 个模型");
      } else {
        message.success(`获取到 ${models.length} 个模型`);
      }
    } catch (err) {
      message.error(`获取模型失败:${err.message}`);
    } finally {
      setLoading((prev) => ({ ...prev, [endpoint.id]: false }));
    }
  }

  function addModelManual(endpoint) {
    const draft = (modelDraft[endpoint.id] ?? "").trim();
    if (!draft) return;
    const exists = (endpoint.models ?? []).some((m) => m === draft);
    if (exists) {
      message.warning("该模型已在列表里");
      return;
    }
    patchEndpoint(endpoint.id, { models: [...(endpoint.models ?? []), draft] });
    setModelDraft((prev) => ({ ...prev, [endpoint.id]: "" }));
  }

  function removeModel(endpoint, modelName) {
    const next = (endpoint.models ?? []).filter((m) => m !== modelName);
    patchEndpoint(endpoint.id, { models: next });
  }

  return (
    <div className="endpoint-list-editor" id={`${idPrefix}-endpoint-list`}>
      {list.map((endpoint, index) => {
        const models = Array.isArray(endpoint.models) ? endpoint.models : [];
        const isFetching = Boolean(loading[endpoint.id]);
        const draft = modelDraft[endpoint.id] ?? "";
        const cardId = `${idPrefix}-endpoint-${index}`;
        return (
          <div className="endpoint-card" id={cardId} key={endpoint.id}>
            <div className="endpoint-card-head">
              <span className="endpoint-card-title">
                端点 #{index + 1}
                {endpoint.name?.trim() ? `·${endpoint.name.trim()}` : ""}
              </span>
              <button
                id={`${cardId}-delete`}
                type="button"
                className="chip endpoint-delete-btn"
                title="删除该端点"
                aria-label="删除该端点"
                onClick={() => removeEndpoint(endpoint.id)}
              >
                删除端点
              </button>
            </div>
            <div className="endpoint-card-body">
              <div className="settings-item" id={`${cardId}-name`}>
                <label className="settings-item-label" htmlFor={`${cardId}-name-input`}>
                  名称(可选)
                </label>
                <input
                  id={`${cardId}-name-input`}
                  type="text"
                  value={endpoint.name ?? ""}
                  placeholder="留空时显示 Base URL"
                  onChange={(e) => patchEndpoint(endpoint.id, { name: e.target.value })}
                />
              </div>
              <div className="settings-item" id={`${cardId}-interface`}>
                <label className="settings-item-label" htmlFor={`${cardId}-interface-input`}>
                  接口类型
                </label>
                <select
                  id={`${cardId}-interface-input`}
                  value={endpoint.interface ?? "ollama"}
                  onChange={(e) => patchEndpoint(endpoint.id, { interface: e.target.value })}
                >
                  <option value="ollama">Ollama</option>
                  <option value="openai">OpenAI 兼容</option>
                </select>
              </div>
              <div className="settings-item" id={`${cardId}-base-url`}>
                <label className="settings-item-label" htmlFor={`${cardId}-base-url-input`}>
                  Base URL
                </label>
                <input
                  id={`${cardId}-base-url-input`}
                  type="text"
                  value={endpoint.base_url ?? ""}
                  placeholder={
                    endpoint.interface === "openai"
                      ? "http://host:port/v1"
                      : "http://127.0.0.1:11434"
                  }
                  onChange={(e) => patchEndpoint(endpoint.id, { base_url: e.target.value })}
                />
              </div>
              <div className="settings-item" id={`${cardId}-api-key`}>
                <label className="settings-item-label" htmlFor={`${cardId}-api-key-input`}>
                  API Key
                </label>
                <input
                  id={`${cardId}-api-key-input`}
                  type="password"
                  value={endpoint.api_key ?? ""}
                  autoComplete="off"
                  placeholder="OpenAI 兼容接口需要;Ollama 可留空"
                  onChange={(e) => patchEndpoint(endpoint.id, { api_key: e.target.value })}
                />
              </div>
              <div className="settings-item" id={`${cardId}-models`}>
                <span className="settings-item-label">模型列表</span>
                <div className="settings-item-row">
                  <button
                    id={`${cardId}-fetch-models`}
                    type="button"
                    className="chip"
                    disabled={isFetching}
                    onClick={() => fetchModels(endpoint)}
                  >
                    {isFetching ? "获取中……" : "获取模型列表"}
                  </button>
                </div>
                <div className="preset-editor-tags endpoint-model-tags">
                  {models.map((name) => (
                    <span
                      className="preset-tag"
                      id={`${cardId}-model-${name}`}
                      key={name}
                    >
                      {name}
                      <button
                        id={`${cardId}-delete-model-${name}`}
                        type="button"
                        className="preset-tag-delete"
                        aria-label={`删除模型 ${name}`}
                        onClick={() => removeModel(endpoint, name)}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                  <span className="endpoint-model-add">
                    <input
                      id={`${cardId}-model-input`}
                      type="text"
                      value={draft}
                      placeholder="手动添加模型"
                      onChange={(e) =>
                        setModelDraft((prev) => ({ ...prev, [endpoint.id]: e.target.value }))
                      }
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                          e.preventDefault();
                          addModelManual(endpoint);
                        }
                      }}
                    />
                    <button
                      id={`${cardId}-add-model`}
                      type="button"
                      className="chip"
                      onClick={() => addModelManual(endpoint)}
                    >
                      添加
                    </button>
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
      <button
        id={`${idPrefix}-add-endpoint`}
        type="button"
        className="chip endpoint-add-btn"
        onClick={addEndpoint}
      >
        + 添加端点
      </button>
    </div>
  );
}

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
      <div className="settings-item" id="settings-item-llm-endpoints">
        <span className="settings-item-label">大模型端点</span>
        <p className="settings-item-desc">
          支持配置多个 Ollama / OpenAI 兼容端点;默认模型下拉里的选项来自这里。
        </p>
        <EndpointListEditor
          endpoints={form.endpoints}
          onChange={(next) => setField("endpoints", next)}
          idPrefix="llm"
        />
      </div>
      <div className="settings-item" id="settings-item-llm-selected">
        <label className="settings-item-label" htmlFor="llm-selected-model">
          默认模型
        </label>
        <p className="settings-item-desc">
          快捷生成提示词使用的默认模型,可在弹窗里临时切换;保存后端会自动规范化端点 id。
        </p>
        <RemoteModelSelect
          idPrefix="llm-selected-model"
          endpoints={form.endpoints}
          selected={form.selected}
          onChange={(next) => setField("selected", next)}
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

// 图片反推页内部的二级 tabs:反推设置 / 请求记录(与 LLMTab 同构)。
// 两个面板同时挂载、仅切换显隐,避免切换 tab 丢失未保存的表单内容。
const CAPTION_SUB_TABS = [
  { key: "settings", label: "反推设置" },
  { key: "records", label: "请求记录" },
];

function CaptionApiTab({ settings, onUpdate }) {
  const [sub, setSub] = useState("settings");

  return (
    <div id="settings-caption-tab">
      <nav id="caption-subnav">
        {CAPTION_SUB_TABS.map((item) => (
          <button
            key={item.key}
            id={`caption-subtab-${item.key}`}
            type="button"
            className={sub === item.key ? "active" : ""}
            onClick={() => setSub(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      <div hidden={sub !== "settings"}>
        <CaptionApiSettings settings={settings} onUpdate={onUpdate} />
      </div>
      <div hidden={sub !== "records"}>
        <CaptionRecords />
      </div>
    </div>
  );
}

// 后端 settings.caption_api 缺失或缺键时的前端兜底(服务端保存时也会补齐默认值;
// prompt 默认值与本地反推的系统提示词一致,由服务端补齐)
const CAPTION_API_DEFAULTS = {
  endpoints: [],
  selected: { endpoint_id: "", model: "" },
  prompt: "",
  think: false,
};

// 图片反推 API 设置:图片反推 tab 的「API 反推」使用的外部视觉模型接口配置。
// 结构与 LLMSettings 一致:多端点 + 共享字段;不再有「测试连通性」按钮,改用端点上的「获取模型列表」。
function CaptionApiSettings({ settings, onUpdate }) {
  const captionApi = settings?.caption_api ?? null;
  const message = useMessage();
  const [form, setForm] = useState(() => ({
    ...CAPTION_API_DEFAULTS,
    ...(captionApi ?? {}),
  }));
  const [saving, setSaving] = useState(false);

  // 已保存快照:与表单对比得出「未保存」状态
  const savedSnapshot = useMemo(
    () => JSON.stringify({ ...CAPTION_API_DEFAULTS, ...(captionApi ?? {}) }),
    [captionApi],
  );
  const dirty = JSON.stringify(form) !== savedSnapshot;

  // 初次进入及保存成功后,以后端返回的最新 settings.caption_api 为准同步表单
  useEffect(() => {
    setForm({ ...CAPTION_API_DEFAULTS, ...(captionApi ?? {}) });
  }, [captionApi]);

  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function save() {
    setSaving(true);
    try {
      await onUpdate({ caption_api: { ...form } });
      // 保存成功后 settings.caption_api 更新,dirty 随之消除
      message.success("图片反推 API 设置已保存");
    } catch (err) {
      message.error(`保存失败:${err.message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div id="settings-caption-api" className="panel">
      <h2>
        图片反推 API
        {dirty && (
          <span id="caption-api-unsaved-badge" className="unsaved-badge">
            未保存
          </span>
        )}
      </h2>
      <div className="settings-item" id="settings-item-caption-api-endpoints">
        <span className="settings-item-label">视觉模型端点</span>
        <p className="settings-item-desc">
          支持配置多个 Ollama / OpenAI 兼容端点,每个端点可手动添加或「获取模型列表」拉取。
        </p>
        <EndpointListEditor
          endpoints={form.endpoints}
          onChange={(next) => setField("endpoints", next)}
          idPrefix="caption-api"
        />
      </div>
      <div className="settings-item" id="settings-item-caption-api-selected">
        <label className="settings-item-label" htmlFor="caption-api-selected-model">
          默认模型
        </label>
        <p className="settings-item-desc">
          「API 反推」使用的默认模型,需为支持图片输入的视觉模型。
        </p>
        <RemoteModelSelect
          idPrefix="caption-api-selected-model"
          endpoints={form.endpoints}
          selected={form.selected}
          onChange={(next) => setField("selected", next)}
        />
      </div>
      <div className="settings-item" id="settings-item-caption-api-think">
        <label className="settings-item-label" htmlFor="caption-api-think">思考 Thinking</label>
        <p className="settings-item-desc">
          反推一般无需思考,关闭可明显加快响应;Ollama 原生支持,
          OpenAI 兼容接口关闭时通过 enable_thinking=false 下发(服务端不支持时请保持开启)。
        </p>
        <label className="settings-toggle">
          <input
            id="caption-api-think"
            type="checkbox"
            checked={form.think}
            onChange={(e) => setField("think", e.target.checked)}
          />
          启用思考
        </label>
      </div>
      <div className="settings-item" id="settings-item-caption-api-prompt">
        <label className="settings-item-label" htmlFor="caption-api-prompt">提示词</label>
        <p className="settings-item-desc">
          API 反推使用的系统提示词,默认值与本地反推一致;留空时回退默认值。
        </p>
        <textarea
          id="caption-api-prompt"
          rows={10}
          value={form.prompt}
          onChange={(e) => setField("prompt", e.target.value)}
        />
      </div>
      <div className="settings-item" id="settings-item-caption-api-actions">
        <button
          id="caption-api-save-btn"
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
      <div className="settings-item" id="settings-item-aspect-ratio-presets">
        <label className="settings-item-label">图片比例预设</label>
        <p className="settings-item-desc">
          尺寸卡片「自动计算」模式中可选的图片比例(宽:高),配合目标像素自动算出宽高。
        </p>
        <SizePresetEditor
          presets={settings?.aspect_ratio_presets ?? []}
          field="aspect_ratio_presets"
          idPrefix="aspect-ratio-"
          multiple={1}
          display="ratio"
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
      <div className="settings-item" id="settings-item-ref2va-limits">
        <label className="settings-item-label">Ref2VA 输入上限</label>
        <p className="settings-item-desc">
          Ref2VA 表单中参考图片/视频/音频的最多可添加数量;本地算力有限,可按需调低。
          合法范围:图片 0-9、视频 0-3、音频 0-3,总数不超过 12。
        </p>
        <Ref2vaLimitsEditor limits={settings?.ref2va_limits} onUpdate={onUpdate} />
      </div>
    </div>
  );
}

const REF2VA_LIMIT_FIELDS = [
  ["max_images", "参考图片上限", 9],
  ["max_videos", "参考视频上限", 3],
  ["max_audios", "参考音频上限", 3],
];

// Ref2VA 表单输入上限编辑器;单项修改即提交,后端用默认值合并其余键
function Ref2vaLimitsEditor({ limits, onUpdate }) {
  const current = {
    max_images: limits?.max_images ?? 3,
    max_videos: limits?.max_videos ?? 1,
    max_audios: limits?.max_audios ?? 1,
  };
  return (
    <div className="ref2va-limits-editor">
      {REF2VA_LIMIT_FIELDS.map(([key, label, max]) => (
        <label className="ref2va-limit-field" key={key} id={`settings-ref2va-${key}`}>
          <span>{label}</span>
          <input
            type="number"
            min={0}
            max={max}
            step={1}
            value={current[key]}
            onChange={(e) => {
              const value = Number(e.target.value);
              if (!Number.isInteger(value)) return;
              onUpdate({ ref2va_limits: { [key]: value } });
            }}
          />
        </label>
      ))}
    </div>
  );
}

// field 指定写入的设置项键名;idPrefix 用于区分图片/视频多处编辑器的 DOM id;
// multiple 是该系列模型要求的宽高倍数(wan 为 16,MiniMax 为 32,比例为 1);
// display="ratio" 时按宽高比展示(3:4),否则按尺寸展示(576×768)
function SizePresetEditor({ presets, onUpdate, field = "size_presets", idPrefix = "", multiple = 16, display = "size" }) {
  const [adding, setAdding] = useState(false);
  const [width, setWidth] = useState(576);
  const [height, setHeight] = useState(576);
  const [error, setError] = useState(null);

  const formatTag = (w, h) =>
    display === "ratio" ? `${w}:${h}` : w === h ? `${w}²` : `${w}×${h}`;
  const separator = display === "ratio" ? ":" : "×";

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
        setError(
          multiple > 1
            ? `${label}必须是正整数且为 ${multiple} 的倍数`
            : `${label}必须是正整数`,
        );
        return;
      }
    }
    if (presets.some(([w, h]) => w === width && h === height)) {
      setError(display === "ratio" ? "该比例已存在" : "该尺寸标签已存在");
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
            {formatTag(w, h)}
            <button
              id={`${idPrefix}delete-tag-${w}x${h}`}
              type="button"
              className="preset-tag-delete"
              aria-label={`删除 ${formatTag(w, h)}`}
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
          <span className="preset-add-x">{separator}</span>
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
