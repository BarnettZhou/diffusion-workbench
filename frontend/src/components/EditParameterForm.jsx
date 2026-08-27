import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import ImageCropModal from "./ImageCropModal";
import { useMessage } from "./Message";
import PromptPresetPicker from "./PromptPresetPicker";

// 采样器/调度器选项从 /api/v1/sampling-options 拉取,接口不可用时用兜底列表
const FALLBACK_SAMPLERS = ["euler", "dpmpp_2m_sde"];
const FALLBACK_SCHEDULERS = ["simple", "sgm_uniform", "beta"];
const LIMITS = {
  steps: { min: 1, max: 100 },
  count: { min: 1, max: 32 },
  size: { min: 256, max: 4096, multiple: 16 },
};

// 服务端接受的输入图片类型(与后端白名单一致)
const UPLOAD_TYPES = ["image/png", "image/jpeg", "image/webp"];

// 手机相册选出的图片可能是 HEIC 或没有 Content-Type:浏览器能解码的(如 iOS 的 HEIC)
// 统一转成 PNG 再上传;浏览器也解码不了时给出明确错误。RebalanceParameterForm 复用。
export async function normalizeUploadFile(file) {
  if (UPLOAD_TYPES.includes(file.type)) return file;
  let bitmap = null;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error("该图片格式无法识别(如 HEIC),请先转换为 PNG/JPEG/WebP");
  }
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  bitmap.close();
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("图片转换失败,请换一张图片");
  const base = file.name.replace(/\.[^.]+$/, "") || "input";
  return new File([blob], `${base}.png`, { type: "image/png" });
}

// Krea2 图像编辑参数表单。结构参照 ParameterForm,字段对齐后端契约。
// resources 取自 App.resources["krea2"],defaults 取自 /api/v1/edit/info。
// 输入图片沿用视频表单的受控上传模式 + ImageCropModal 裁剪。
export default function EditParameterForm({
  resources,
  defaults,
  sizePresets,
  samplingDefaults,
  promptPresets,
  inputImagePrefill = null,
  onSubmit,
}) {
  const message = useMessage();
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [modelIndex, setModelIndex] = useState(null);
  const [vaeIndex, setVaeIndex] = useState(null);
  const [textEncoderIndex, setTextEncoderIndex] = useState(null);
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
  // 编辑专属参数:grounding_px=0 表示使用原图尺寸;ref_boost=1.0 表示关闭参考加强
  const [groundingPx, setGroundingPx] = useState("768");
  const [refBoost, setRefBoost] = useState("1");
  // 输入图片:{id, previewUrl, name, file};file 保留原始 File 用于再次裁剪
  const [inputImage, setInputImage] = useState(null);
  const [inputImageUploading, setInputImageUploading] = useState(false);
  const inputImageRef = useRef(null);
  // 裁剪弹窗开关;比例取自当前宽度/高度输入值
  const [cropOpen, setCropOpen] = useState(false);
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
  // krea2 在 settings 里可以选用 checkpoint loader(自带 VAE),此时不显示 VAE 选择器
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

  // defaults 与采样选项就位后,首次进入用编辑默认参数 + krea2 默认采样参数填写
  useEffect(() => {
    if (!defaults || !optionsReady || defaultsAppliedRef.current) return;
    defaultsAppliedRef.current = true;
    const editDefaults = defaults ?? {};
    if (Number.isFinite(Number(editDefaults.grounding_px))) {
      setGroundingPx(String(Math.max(0, Math.floor(Number(editDefaults.grounding_px)))));
    }
    if (Number.isFinite(Number(editDefaults.ref_boost))) {
      setRefBoost(String(Number(editDefaults.ref_boost)));
    }
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
  }, [defaults, samplingDefaults, optionsReady, samplerOptions, schedulerOptions]);

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

  // 相册"发送到图片编辑"带过来的输入图片:已在服务端导入,直接引用受控 URL,不走本地上传
  useEffect(() => {
    if (!inputImagePrefill) return;
    setInputImage((prev) => {
      revokePreview(prev);
      return {
        id: inputImagePrefill.id,
        previewUrl: inputImagePrefill.url,
        name: inputImagePrefill.name,
      };
    });
  }, [inputImagePrefill]);

  // 选择文件后立即上传,本地用 object URL 预览;保存 File 以便后续重新裁剪
  async function handleInputImageSelect(file) {
    if (!file) return;
    setError(null);
    setInputImageUploading(true);
    try {
      const uploadFile = await normalizeUploadFile(file);
      const previewUrl = URL.createObjectURL(uploadFile);
      try {
        const saved = await api.uploadEditInputImage(uploadFile);
        setInputImage((prev) => {
          revokePreview(prev);
          return { id: saved.id, previewUrl, name: uploadFile.name, file: uploadFile };
        });
      } catch (err) {
        URL.revokeObjectURL(previewUrl);
        throw err;
      }
    } catch (err) {
      setError(err.message);
      // 移动端表单较长,底部内联错误可能不在视野内,同步弹全局 toast
      message?.error(err.message);
    } finally {
      setInputImageUploading(false);
    }
  }

  function clearInputImage() {
    setInputImage((prev) => {
      revokePreview(prev);
      return null;
    });
  }

  function revokePreview(item) {
    if (item?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(item.previewUrl);
  }

  // 打开裁剪弹窗:比例取自当前宽度/高度输入值,无效时先提示
  function openCrop() {
    const widthNum = Number(width);
    const heightNum = Number(height);
    if (!Number.isFinite(widthNum) || !Number.isFinite(heightNum) || widthNum <= 0 || heightNum <= 0) {
      setError("裁剪比例取自宽度和高度,请先把这两个值填为有效数值");
      return;
    }
    setError(null);
    setCropOpen(true);
  }

  function resetToDefaults() {
    if (defaults) {
      if (Number.isFinite(Number(defaults.grounding_px))) {
        setGroundingPx(String(Math.max(0, Math.floor(Number(defaults.grounding_px)))));
      }
      if (Number.isFinite(Number(defaults.ref_boost))) {
        setRefBoost(String(Number(defaults.ref_boost)));
      }
    }
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
    if (!inputImage) return "请先上传输入图片";
    if (inputImageUploading) return "输入图片上传中,请稍候";
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
    const groundingNum = Number(groundingPx);
    if (!Number.isFinite(groundingNum) || groundingNum < 0) {
      return "grounding_px 必须大于等于 0(0 表示使用原图尺寸)";
    }
    const refBoostNum = Number(refBoost);
    if (!Number.isFinite(refBoostNum) || refBoostNum <= 0) {
      return "ref_boost 必须大于 0";
    }
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
        prompt: prompt.trim(),
        negative_prompt: negativePrompt,
        input_image_id: inputImage.id,
        width: Number(width),
        height: Number(height),
        steps: Number(steps),
        seed: Number(seed),
        count: Number(count),
        cfg: Number(cfg),
        sampler,
        scheduler,
        grounding_px: Number(groundingPx),
        ref_boost: Number(refBoost),
      });
    } catch (err) {
      setError(err.message);
      message?.error(err.message);
    }
  }

  return (
    <form id="edit-parameter-form" className="form-stack" onSubmit={handleSubmit}>
      <div id="edit-basic-params-card" className="panel form">
        <div className="card-header">
          <h2>Krea2 图像编辑参数</h2>
          <button
            id="edit-reset-defaults-btn"
            type="button"
            className="chip"
            title="用编辑默认参数 + krea2 默认采样参数填写"
            onClick={resetToDefaults}
          >
            重置
          </button>
        </div>

        <div className="field" id="field-edit-prompt">
          <div className="label-row">
            <label htmlFor="edit-prompt-input">Prompt</label>
            <div className="label-actions">
              <button
                id="edit-prompt-preset-btn"
                type="button"
                className="prompt-assist-btn"
                title="选择预设提示词"
                aria-label="选择预设提示词"
                onClick={() => setPresetTarget("positive")}
              >
                预设
              </button>
              <button
                id="edit-prompt-size-btn"
                type="button"
                className="prompt-assist-btn"
                title={`切换输入框高度(当前 ${promptSize}x)`}
                aria-label={`切换输入框高度(当前 ${promptSize}x)`}
                onClick={() => setPromptSize((size) => (size % 3) + 1)}
              >
                {promptSize}x
              </button>
              <button
                id="edit-prompt-clear-btn"
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
            id="edit-prompt-input"
            className={promptSize > 1 ? `prompt-size-${promptSize}` : undefined}
            value={prompt}
            placeholder="描述你想要的编辑效果……"
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div className="field" id="field-edit-negative-prompt">
          <div className="label-row">
            <label htmlFor="edit-negative-prompt-input">负面提示词 Negative Prompt</label>
            <div className="label-actions">
              <button
                id="edit-negative-preset-btn"
                type="button"
                className="prompt-assist-btn"
                title="选择预设提示词"
                aria-label="选择预设提示词"
                onClick={() => setPresetTarget("negative")}
              >
                预设
              </button>
              <button
                id="edit-negative-size-btn"
                type="button"
                className="prompt-assist-btn"
                title={`切换输入框高度(当前 ${negativePromptSize}x)`}
                aria-label={`切换输入框高度(当前 ${negativePromptSize}x)`}
                onClick={() => setNegativePromptSize((size) => (size % 3) + 1)}
              >
                {negativePromptSize}x
              </button>
              <button
                id="edit-negative-clear-btn"
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
            id="edit-negative-prompt-input"
            className={negativePromptSize > 1 ? `prompt-size-${negativePromptSize}` : undefined}
            value={negativePrompt}
            placeholder="不想出现在画面中的内容(可留空)……"
            onChange={(e) => setNegativePrompt(e.target.value)}
          />
        </div>

        <div className="field" id="field-edit-input-image">
          <div className="label-row">
            <label htmlFor="edit-input-image-input">输入图片(必选)</label>
            <div className="label-actions">
              {inputImage && (
                <button
                  id="edit-input-image-crop-btn"
                  type="button"
                  className="prompt-assist-btn"
                  title="按当前宽度×高度比例裁剪输入图片"
                  aria-label="裁剪输入图片"
                  onClick={openCrop}
                >
                  裁剪
                </button>
              )}
              {inputImage && (
                <button
                  id="edit-input-image-clear-btn"
                  type="button"
                  className="prompt-assist-btn"
                  title="移除输入图片"
                  aria-label="移除输入图片"
                  onClick={clearInputImage}
                >
                  移除
                </button>
              )}
            </div>
          </div>
          <div
            id="edit-input-image-drop"
            className={`video-input-image${inputImage ? " has-image" : ""}`}
            role="button"
            tabIndex={0}
            title="点击选择图片(png/jpeg/webp,≤32MB)"
            onClick={() => inputImageRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && inputImageRef.current?.click()}
          >
            {inputImage ? (
              <img src={inputImage.previewUrl} alt={inputImage.name} />
            ) : (
              <div className="video-input-image-empty">点击上传要编辑的图片</div>
            )}
            {inputImageUploading && (
              <div className="video-input-image-loading">上传中……</div>
            )}
            <input
              ref={inputImageRef}
              id="edit-input-image-input"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              aria-required="true"
              hidden
              disabled={inputImageUploading}
              onChange={(e) => {
                handleInputImageSelect(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </div>
          {inputImage && (
            <p className="form-hint" id="edit-input-image-hint">
              已上传:{inputImage.name},可在「裁剪」中按当前宽度×高度重新构图。
            </p>
          )}
        </div>

        <div className="field" id="field-edit-model">
          <div className="label-row">
            <label htmlFor="edit-model-select">模型 Checkpoint</label>
            <button
              id="edit-model-picker-toggle"
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
            id="edit-model-select"
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
          <div id="edit-model-picker" className="model-picker">
            {pickerModels.map((item) => {
              const cover = modelCovers[item.name];
              return (
                <button
                  key={item.index}
                  id={`edit-model-pick-${item.index}`}
                  type="button"
                  className={`model-card picker-card ${modelIndex === item.index ? "active" : ""}`}
                  title={item.display_name}
                  onClick={() => setModelIndex(item.index)}
                >
                  <div className="model-cover">
                    {cover ? (
                      <img src={cover} alt={item.name} loading="lazy" />
                    ) : (
                      <div className="model-cover-empty">暂无封面</div>
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
          <div className="field" id="field-edit-vae">
            <label htmlFor="edit-vae-select">VAE</label>
            <select
              id="edit-vae-select"
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
          <div className="field" id="field-edit-text-encoder">
            <label htmlFor="edit-text-encoder-select">文本编码器 Text Encoder</label>
            <select
              id="edit-text-encoder-select"
              value={textEncoderIndex ?? ""}
              disabled={!textEncoders.length}
              onChange={(e) => setTextEncoderIndex(Number(e.target.value))}
            >
              {textEncoders.map((item) => (
                <option key={item.index} value={item.index}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="row" id="edit-size-row">
          <div className="field" id="field-edit-width">
            <label htmlFor="edit-width-input">宽度 (px)</label>
            <input
              id="edit-width-input" type="number" step={LIMITS.size.multiple}
              min={LIMITS.size.min} max={LIMITS.size.max}
              value={width} onChange={(e) => setWidth(e.target.value)}
            />
          </div>
          <div className="field" id="field-edit-height">
            <label htmlFor="edit-height-input">高度 (px)</label>
            <input
              id="edit-height-input" type="number" step={LIMITS.size.multiple}
              min={LIMITS.size.min} max={LIMITS.size.max}
              value={height} onChange={(e) => setHeight(e.target.value)}
            />
          </div>
        </div>

        {(sizePresets ?? []).length > 0 && (
          <div className="preset-row" id="edit-size-presets">
            {sizePresets.map(([presetWidth, presetHeight]) => (
              <button
                key={`${presetWidth}x${presetHeight}`}
                id={`edit-preset-${presetWidth}x${presetHeight}`}
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

        <div className="row" id="edit-steps-count-row">
          <div className="field" id="field-edit-steps">
            <label htmlFor="edit-steps-input">采样步数 Steps</label>
            <input
              id="edit-steps-input" type="number"
              min={LIMITS.steps.min} max={LIMITS.steps.max}
              value={steps} onChange={(e) => setSteps(e.target.value)}
            />
          </div>
          <div className="field" id="field-edit-count">
            <label htmlFor="edit-count-input">批次数量</label>
            <input
              id="edit-count-input" type="number"
              min={LIMITS.count.min} max={LIMITS.count.max}
              value={count} onChange={(e) => setCount(e.target.value)}
            />
          </div>
        </div>

        <div className="row" id="edit-fixed-params-row">
          <div className="field" id="field-edit-sampler">
            <label htmlFor="edit-sampler-select">采样器 Sampler</label>
            <select
              id="edit-sampler-select"
              value={sampler}
              onChange={(e) => setSampler(e.target.value)}
            >
              {samplerOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          <div className="field" id="field-edit-scheduler">
            <label htmlFor="edit-scheduler-select">调度器 Scheduler</label>
            <select
              id="edit-scheduler-select"
              value={scheduler}
              onChange={(e) => setScheduler(e.target.value)}
            >
              {schedulerOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          <div className="field" id="field-edit-cfg">
            <label htmlFor="edit-cfg-input">CFG</label>
            <input
              id="edit-cfg-input" type="number" min={0} step="any"
              value={cfg} onChange={(e) => setCfg(e.target.value)}
            />
          </div>
          <div className="field" id="field-edit-seed">
            <label htmlFor="edit-seed-input">种子 Seed</label>
            <input
              id="edit-seed-input" type="number" min={-1} step={1}
              title="-1 表示随机"
              value={seed} onChange={(e) => setSeed(e.target.value)}
            />
          </div>
        </div>

        <div className="row" id="edit-grounding-ref-row">
          <div className="field" id="field-edit-grounding">
            <label htmlFor="edit-grounding-input">Grounding 像素</label>
            <input
              id="edit-grounding-input" type="number" min={0} step={32}
              title="参考图缩放后的短边像素,0 表示使用原图尺寸"
              value={groundingPx} onChange={(e) => setGroundingPx(e.target.value)}
            />
            <p className="form-hint" id="edit-grounding-hint">0 = 原图尺寸</p>
          </div>
          <div className="field" id="field-edit-ref-boost">
            <label htmlFor="edit-ref-boost-input">Ref Boost</label>
            <input
              id="edit-ref-boost-input" type="number" min={0} step="0.05"
              title="参考图加强度,1.0 为关闭,大于 1 更贴近参考图"
              value={refBoost} onChange={(e) => setRefBoost(e.target.value)}
            />
            <p className="form-hint" id="edit-ref-boost-hint">1.0 = 关闭,&gt;1 更贴近参考图</p>
          </div>
        </div>
      </div>

      {error && <div id="edit-form-error" className="form-error">{error}</div>}

      {cropOpen && inputImage && (
        <ImageCropModal
          src={inputImage.previewUrl}
          ratio={Number(width) / Number(height)}
          onClose={() => setCropOpen(false)}
          onConfirm={(file) => {
            setCropOpen(false);
            handleInputImageSelect(file);
          }}
        />
      )}

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