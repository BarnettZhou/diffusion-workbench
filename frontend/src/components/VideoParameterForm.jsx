import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import PromptPresetPicker from "./PromptPresetPicker";

// 采样器/调度器选项与图片表单同源(/api/v1/sampling-options),接口不可用时用兜底列表
const FALLBACK_SAMPLERS = ["uni_pc", "euler", "dpmpp_2m_sde"];
const FALLBACK_SCHEDULERS = ["simple", "sgm_uniform", "beta"];
const LIMITS = {
  steps: { min: 1, max: 100 },
  count: { min: 1, max: 8 },
  size: { min: 16, max: 4096, multiple: 16 },
  duration: { min: 1, max: 60 },
  fps: { min: 1, max: 60 },
};

// 视频生成参数表单。结构与 ParameterForm 一致(卡片 + field/row 布局),
// videoModel 与 resources 由 App 下发;切 tab 只隐藏不卸载,表单现场保留。
export default function VideoParameterForm({
  videoModel,
  modelInfo,
  resources,
  inputImagePrefill,
  paramsPrefill,
  promptPresets,
  sizePresets,
  onSubmit,
}) {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [modelIndex, setModelIndex] = useState(null);
  const [highModelIndex, setHighModelIndex] = useState(null);
  const [lowModelIndex, setLowModelIndex] = useState(null);
  const [vaeIndex, setVaeIndex] = useState(null);
  // 数字输入框一律存原始字符串:允许删空和 "-1" 这类中间态,提交时才校验/转换
  const [width, setWidth] = useState("704");
  const [height, setHeight] = useState("960");
  const [duration, setDuration] = useState("5");
  const [fps, setFps] = useState("24");
  const [steps, setSteps] = useState("20");
  const [count, setCount] = useState("1");
  const [cfg, setCfg] = useState("5");
  const [shift, setShift] = useState("8");
  const [latentMultiplier, setLatentMultiplier] = useState("1");
  const [sampler, setSampler] = useState("uni_pc");
  const [scheduler, setScheduler] = useState("simple");
  const [seed, setSeed] = useState("-1");
  const [samplerOptions, setSamplerOptions] = useState(FALLBACK_SAMPLERS);
  const [schedulerOptions, setSchedulerOptions] = useState(FALLBACK_SCHEDULERS);
  const [error, setError] = useState(null);
  // I2V 输入图片:{id, previewUrl, name};选择文件后立即走受控上传接口
  const [inputImage, setInputImage] = useState(null);
  const [inputImageUploading, setInputImageUploading] = useState(false);
  const inputImageRef = useRef(null);
  const [referenceImage, setReferenceImage] = useState(null);
  const [referenceImageUploading, setReferenceImageUploading] = useState(false);
  const referenceImageRef = useRef(null);
  // 预设提示词选择弹窗:null 关闭;"positive"/"negative" 表示替换目标输入框
  const [presetTarget, setPresetTarget] = useState(null);
  // prompt/负面提示词输入框高度倍率,循环 1x → 2x → 3x
  const [promptSize, setPromptSize] = useState(1);
  const [negativePromptSize, setNegativePromptSize] = useState(1);

  const models = resources?.models ?? [];
  const vaes = resources?.vaes ?? [];
  const usesDualModels = videoModel === "wan2.2-i2v-14b";
  const isMiniMaxH3 = videoModel === "minimax-h3";
  const isRef2va = isMiniMaxH3 && models.find((item) => item.index === modelIndex)?.name.toLowerCase().includes("ref2va");
  const highModels = models.filter((item) =>
    item.name.toLowerCase().includes("_high_noise_"),
  );
  const lowModels = models.filter((item) =>
    item.name.toLowerCase().includes("_low_noise_"),
  );
  function pairedModelIndex(item, candidates, fromMarker, toMarker) {
    if (!item) return null;
    const pairName = item.name.replace(fromMarker, toMarker);
    return candidates.find((candidate) => candidate.name === pairName)?.index ?? null;
  }
  // 兼容尚未返回能力字段的旧 API；14B 始终只允许 I2V。
  const requiresInputImage =
    modelInfo?.requires_input_image ?? videoModel === "wan2.2-i2v-14b";
  const videoModelLabel = modelInfo?.label ?? videoModel;

  // 总帧数 length = 时长 × 帧率 + 1,Wan latent 要求满足 4n+1(即乘积是 4 的倍数);
  // 输入被删空或是中间态时不计算,由提交时的校验拦截
  const durationNum = Number(duration);
  const fpsNum = Number(fps);
  const framesComputable = Number.isInteger(durationNum) && Number.isInteger(fpsNum);
  const wanTotalFrames = durationNum * fpsNum + 1;
  const h3BaseFrames = Math.max(5, Math.round(durationNum * 24));
  const h3TotalFrames = h3BaseFrames + ((5 - (h3BaseFrames % 17) + 17) % 17);
  const totalFrames = isMiniMaxH3 ? h3TotalFrames : wanTotalFrames;
  const framesValid =
    framesComputable &&
    (isMiniMaxH3 ? fpsNum === 24 && totalFrames % 17 === 5 : (totalFrames - 1) % 4 === 0);

  // 拉取全部可用采样器/调度器,失败时保留兜底列表
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

  // 资源加载或视频模型切换后,当前选择失效时默认选中第一项
  useEffect(() => {
    setModelIndex((prev) =>
      models.some((item) => item.index === prev) ? prev : (models[0]?.index ?? null),
    );
    setHighModelIndex((prev) =>
      highModels.some((item) => item.index === prev)
        ? prev
        : (highModels[0]?.index ?? null),
    );
    setLowModelIndex((prev) =>
      lowModels.some((item) => item.index === prev)
        ? prev
        : (lowModels[0]?.index ?? null),
    );
    setVaeIndex((prev) =>
      vaes.some((item) => item.index === prev) ? prev : (vaes[0]?.index ?? null),
    );
  }, [resources]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // 切回 FL2VA 时清除不再适用的 Ref2VA 参考图，避免表单进入不可提交状态。
    if ((!isMiniMaxH3 || (modelIndex !== null && !isRef2va)) && referenceImage) {
      clearReferenceImage();
    }
  }, [isMiniMaxH3, isRef2va, modelIndex]);

  // 按模型家族写入已验证的默认采样参数。
  useEffect(() => {
    if (videoModel === "minimax-h3") {
      setWidth("608");
      setHeight("352");
      setDuration("5");
      setFps("24");
      setSteps("8");
      setCfg("1");
      setShift("12");
      setSampler("res_multistep");
    } else {
      setSampler(videoModel === "wan2.2-i2v-14b" ? "euler" : "uni_pc");
    }
  }, [videoModel]);

  // 相册视频抽屉"发送到工作台"带过来的生成参数预填;
  // 模型/VAE 按文件名匹配当前分类的资源,匹配不到保留默认选中
  useEffect(() => {
    if (!paramsPrefill) return;
    setPrompt(paramsPrefill.prompt ?? "");
    setNegativePrompt(paramsPrefill.negative_prompt ?? "");
    if (Number.isInteger(paramsPrefill.width)) setWidth(String(paramsPrefill.width));
    if (Number.isInteger(paramsPrefill.height)) setHeight(String(paramsPrefill.height));
    if (Number.isInteger(paramsPrefill.duration_seconds)) {
      setDuration(String(paramsPrefill.duration_seconds));
    }
    if (Number.isInteger(paramsPrefill.fps)) setFps(String(paramsPrefill.fps));
    if (Number.isInteger(paramsPrefill.steps)) setSteps(String(paramsPrefill.steps));
    if (paramsPrefill.cfg != null) setCfg(String(paramsPrefill.cfg));
    if (paramsPrefill.shift != null) setShift(String(paramsPrefill.shift));
    if (paramsPrefill.latent_multiplier != null) {
      setLatentMultiplier(String(paramsPrefill.latent_multiplier));
    }
    if (paramsPrefill.sampler) setSampler(paramsPrefill.sampler);
    if (paramsPrefill.scheduler) setScheduler(paramsPrefill.scheduler);
    if (Number.isInteger(paramsPrefill.seed)) setSeed(String(paramsPrefill.seed));
    const model = models.find((item) => item.name === paramsPrefill.model_name);
    if (model) {
      setModelIndex(model.index);
      if (model.name.toLowerCase().includes("_high_noise_")) {
        setHighModelIndex(model.index);
        const lowName = model.name.replace("_high_noise_", "_low_noise_");
        const lowModel = lowModels.find((item) => item.name === lowName);
        if (lowModel) setLowModelIndex(lowModel.index);
      }
    }
    const vae = vaes.find((item) => item.name === paramsPrefill.vae_name);
    if (vae) setVaeIndex(vae.index);
  }, [paramsPrefill, resources]); // eslint-disable-line react-hooks/exhaustive-deps

  // 相册"发送到视频生成"带过来的输入图片:直接引用受控 URL,不走本地上传
  useEffect(() => {
    if (!inputImagePrefill) return;
    setInputImage((prev) => {
      if (prev?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(prev.previewUrl);
      return {
        id: inputImagePrefill.id,
        previewUrl: inputImagePrefill.url,
        name: inputImagePrefill.name,
      };
    });
  }, [inputImagePrefill]);

  // 选择文件后立即上传,本地用 object URL 预览;清除时仅移除引用,服务端文件保留
  async function handleInputImageSelect(file) {
    if (!file) return;
    setError(null);
    setInputImageUploading(true);
    const previewUrl = URL.createObjectURL(file);
    try {
      const saved = await api.uploadVideoInputImage(file);
      setInputImage((prev) => {
        if (prev?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(prev.previewUrl);
        return { id: saved.id, previewUrl, name: file.name };
      });
    } catch (err) {
      URL.revokeObjectURL(previewUrl);
      setError(err.message);
    } finally {
      setInputImageUploading(false);
    }
  }

  async function handleReferenceImageSelect(file) {
    if (!file) return;
    setError(null);
    setReferenceImageUploading(true);
    const previewUrl = URL.createObjectURL(file);
    try {
      const saved = await api.uploadVideoInputImage(file);
      setReferenceImage({ id: saved.id, previewUrl, name: file.name });
    } catch (err) {
      URL.revokeObjectURL(previewUrl);
      setError(err.message);
    } finally {
      setReferenceImageUploading(false);
    }
  }

  function clearReferenceImage() {
    setReferenceImage((prev) => {
      if (prev?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(prev.previewUrl);
      return null;
    });
  }

  function clearInputImage() {
    setInputImage((prev) => {
      if (prev?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(prev.previewUrl);
      return null;
    });
  }

  // 用内置默认采样参数填写表单(对齐图片表单的「重置」)
  function resetToDefaults() {
    setSteps(isMiniMaxH3 ? "8" : "20");
    setCfg(isMiniMaxH3 ? "1" : "5");
    setShift(isMiniMaxH3 ? "12" : "8");
    setLatentMultiplier("1");
    setSampler(
      isMiniMaxH3
        ? "res_multistep"
        : videoModel === "wan2.2-i2v-14b"
          ? "euler"
          : "uni_pc",
    );
    setScheduler("simple");
    setError(null);
  }

  function validate() {
    if (!models.length || !vaes.length) return "资源列表尚未加载";
    if (usesDualModels) {
      if (!highModels.length || !lowModels.length) {
        return "缺少 high-noise 或 low-noise 模型";
      }
      if (highModelIndex === null || lowModelIndex === null || vaeIndex === null) {
        return "请选择 high-noise、low-noise 模型和 VAE";
      }
    } else if (modelIndex === null || vaeIndex === null) {
      return "请选择模型和 VAE";
    }
    if (!prompt.trim()) return "prompt 不能为空";
    if (inputImageUploading || referenceImageUploading) return "图片上传中,请稍候";
    if (isRef2va && !referenceImage) return "Ref2VA 必须提供一张参考图片";
    if (!isRef2va && referenceImage) return "只有 Ref2VA 模型可以使用参考图片";
    if (referenceImage && inputImage) return "首帧输入与 Ref2VA 参考图不能同时提供";
    if (requiresInputImage && !inputImage) {
      return `${videoModelLabel} 必须提供输入图片`;
    }
    const { multiple, min, max } = LIMITS.size;
    for (const [label, raw] of [["宽度", width], ["高度", height]]) {
      const value = Number(raw);
      const requiredMultiple = isMiniMaxH3 ? 32 : multiple;
      if (!Number.isInteger(value) || value < min || value > max || value % requiredMultiple) {
        return `${label}必须是 ${min}-${max} 之间 ${requiredMultiple} 的倍数`;
      }
    }
    if (!Number.isInteger(durationNum) || durationNum < LIMITS.duration.min || durationNum > LIMITS.duration.max) {
      return `视频长度必须是 ${LIMITS.duration.min}-${LIMITS.duration.max} 秒的整数`;
    }
    if (!Number.isInteger(fpsNum) || fpsNum < LIMITS.fps.min || fpsNum > LIMITS.fps.max) {
      return `帧率必须是 ${LIMITS.fps.min}-${LIMITS.fps.max} 的整数`;
    }
    if (isMiniMaxH3 && fpsNum !== 24) return "MiniMax H3 帧率固定为 24";
    const stepsNum = Number(steps);
    if (!Number.isInteger(stepsNum) || stepsNum < LIMITS.steps.min || stepsNum > LIMITS.steps.max) {
      return `steps 必须在 ${LIMITS.steps.min} 到 ${LIMITS.steps.max} 之间`;
    }
    const countNum = Number(count);
    if (!Number.isInteger(countNum) || countNum < LIMITS.count.min || countNum > LIMITS.count.max) {
      return `批次数量必须在 ${LIMITS.count.min} 到 ${LIMITS.count.max} 之间`;
    }
    if (!Number.isFinite(Number(cfg)) || Number(cfg) <= 0) return "CFG 必须是大于 0 的数值";
    if (isMiniMaxH3 && Number(cfg) !== 1) return "MiniMax H3 CFG 固定为 1";
    if (!Number.isFinite(Number(shift)) || Number(shift) < 0 || Number(shift) > 100) {
      return "Shift 必须在 0 到 100 之间";
    }
    if (!Number.isFinite(Number(latentMultiplier)) || Number(latentMultiplier) <= 0) {
      return "Latent multiplier 必须是大于 0 的数值";
    }
    const seedNum = Number(seed);
    if (seed.trim() === "" || !Number.isInteger(seedNum) || seedNum < -1) {
      return "seed 必须是 -1(随机)或非负整数";
    }
    return null;
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const problem = validate();
    setError(problem);
    if (problem) return;
    try {
      const payload = {
        video_model: videoModel,
        vae_index: vaeIndex,
        prompt: prompt.trim(),
        negative_prompt: negativePrompt,
        input_image_id: inputImage?.id ?? null,
        reference_image_id: referenceImage?.id ?? null,
        width: Number(width),
        height: Number(height),
        duration_seconds: durationNum,
        fps: fpsNum,
        steps: Number(steps),
        seed: Number(seed),
        count: Number(count),
        cfg: Number(cfg),
        shift: Number(shift),
        latent_multiplier: Number(latentMultiplier),
        sampler,
        scheduler,
      };
      if (usesDualModels) {
        payload.high_model_index = highModelIndex;
        payload.low_model_index = lowModelIndex;
      } else {
        payload.model_index = modelIndex;
      }
      await onSubmit(payload);
    } catch (err) {
      // 服务端校验(如总帧数须满足 4n+1)返回的 422 错误直接展示
      setError(err.message);
    }
  }

  return (
    <form id="video-parameter-form" className="form-stack" onSubmit={handleSubmit}>
      <div id="video-basic-params-card" className="panel form">
        <div className="card-header">
          <h2>基础参数</h2>
          <button
            id="video-reset-defaults-btn"
            type="button"
            className="chip"
            title="用默认采样参数填写"
            onClick={resetToDefaults}
          >
            重置
          </button>
        </div>
      <div className="field" id="field-video-prompt">
        <div className="label-row">
          <label htmlFor="video-prompt-input">Prompt</label>
          <div className="label-actions">
            <button
              id="video-prompt-preset-btn"
              type="button"
              className="prompt-assist-btn"
              title="选择预设提示词"
              aria-label="选择预设提示词"
              onClick={() => setPresetTarget("positive")}
            >
              预设
            </button>
            <button
              id="video-prompt-size-btn"
              type="button"
              className="prompt-assist-btn"
              title={`切换输入框高度(当前 ${promptSize}x)`}
              aria-label={`切换输入框高度(当前 ${promptSize}x)`}
              onClick={() => setPromptSize((size) => (size % 3) + 1)}
            >
              {promptSize}x
            </button>
            <button
              id="video-prompt-clear-btn"
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
          id="video-prompt-input"
          className={promptSize > 1 ? `prompt-size-${promptSize}` : undefined}
          value={prompt}
          placeholder="描述你想要生成的视频画面……"
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>

      {isMiniMaxH3 && (
        <div className="field" id="field-video-reference-image">
          <div className="label-row">
            <label htmlFor="video-reference-image-input">Ref2VA 参考图片</label>
            {referenceImage && <button type="button" className="prompt-assist-btn" onClick={clearReferenceImage}>移除</button>}
          </div>
          <div className={`video-input-image${referenceImage ? " has-image" : ""}`} role="button" tabIndex={0}
            onClick={() => referenceImageRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && referenceImageRef.current?.click()}>
            {referenceImage ? <img src={referenceImage.previewUrl} alt={referenceImage.name} /> : <div className="video-input-image-empty">{isRef2va ? "点击上传一张参考图片" : "选择 Ref2VA 模型后可上传参考图片"}</div>}
            <input ref={referenceImageRef} id="video-reference-image-input" type="file"
              accept="image/png,image/jpeg,image/webp" hidden disabled={referenceImageUploading || !isRef2va}
              onChange={(e) => { handleReferenceImageSelect(e.target.files?.[0]); e.target.value = ""; }} />
          </div>
        </div>
      )}

      <div className="field" id="field-video-negative-prompt">
        <div className="label-row">
          <label htmlFor="video-negative-prompt-input">负面提示词 Negative Prompt</label>
          <div className="label-actions">
            <button
              id="video-negative-preset-btn"
              type="button"
              className="prompt-assist-btn"
              title="选择预设提示词"
              aria-label="选择预设提示词"
              onClick={() => setPresetTarget("negative")}
            >
              预设
            </button>
            <button
              id="video-negative-size-btn"
              type="button"
              className="prompt-assist-btn"
              title={`切换输入框高度(当前 ${negativePromptSize}x)`}
              aria-label={`切换输入框高度(当前 ${negativePromptSize}x)`}
              onClick={() => setNegativePromptSize((size) => (size % 3) + 1)}
            >
              {negativePromptSize}x
            </button>
            <button
              id="video-negative-clear-btn"
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
          id="video-negative-prompt-input"
          className={negativePromptSize > 1 ? `prompt-size-${negativePromptSize}` : undefined}
          value={negativePrompt}
          placeholder="不想出现在画面中的内容(可留空)……"
          onChange={(e) => setNegativePrompt(e.target.value)}
        />
      </div>

      <div className="field" id="field-video-input-image">
        <div className="label-row">
          <label htmlFor="video-input-image-input">
            {requiresInputImage ? "输入图片(必选,图生视频)" : "输入图片(可选,图生视频)"}
          </label>
          <div className="label-actions">
            {inputImage && (
              <button
                id="video-input-image-clear-btn"
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
          id="video-input-image-drop"
          className={`video-input-image${inputImage ? " has-image" : ""}`}
          role="button"
          tabIndex={0}
          title="点击选择图片(png/jpeg/webp,≤10MB)"
          onClick={() => inputImageRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && inputImageRef.current?.click()}
        >
          {inputImage ? (
            <img src={inputImage.previewUrl} alt={inputImage.name} />
          ) : (
            <div className="video-input-image-empty">
              {inputImageUploading
                ? "上传中……"
                : requiresInputImage
                  ? "此模型仅支持图生视频;点击上传输入图片"
                  : "不设置为文生视频;点击上传图片做图生视频"}
            </div>
          )}
          <input
            ref={inputImageRef}
            id="video-input-image-input"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            aria-required={requiresInputImage}
            hidden
            disabled={inputImageUploading}
            onChange={(e) => {
              handleInputImageSelect(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>
        {inputImage && (
          <p className="form-hint" id="video-input-image-hint">
            已上传:{inputImage.name},本次提交将按图生视频(I2V)生成。
          </p>
        )}
      </div>

      {usesDualModels ? (
        <div className="row" id="video-dual-model-row">
          <div className="field" id="field-video-high-model">
            <label htmlFor="video-high-model-select">High Noise 模型</label>
            <select
              id="video-high-model-select"
              value={highModelIndex ?? ""}
              disabled={!highModels.length}
              onChange={(e) => {
                const nextIndex = Number(e.target.value);
                const nextModel = highModels.find((item) => item.index === nextIndex);
                setHighModelIndex(nextIndex);
                setLowModelIndex(
                  pairedModelIndex(nextModel, lowModels, "_high_noise_", "_low_noise_"),
                );
              }}
            >
              {highModels.map((item) => (
                <option key={item.index} value={item.index}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </div>
          <div className="field" id="field-video-low-model">
            <label htmlFor="video-low-model-select">Low Noise 模型</label>
            <select
              id="video-low-model-select"
              value={lowModelIndex ?? ""}
              disabled={!lowModels.length}
              onChange={(e) => {
                const nextIndex = Number(e.target.value);
                const nextModel = lowModels.find((item) => item.index === nextIndex);
                setLowModelIndex(nextIndex);
                setHighModelIndex(
                  pairedModelIndex(nextModel, highModels, "_low_noise_", "_high_noise_"),
                );
              }}
            >
              {lowModels.map((item) => (
                <option key={item.index} value={item.index}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </div>
        </div>
      ) : (
        <div className="field" id="field-video-model">
          <label htmlFor="video-model-select">模型 Checkpoint</label>
          <select
            id="video-model-select"
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
      )}

      <div className="field" id="field-video-vae">
        <label htmlFor="video-vae-select">VAE</label>
        <select
          id="video-vae-select"
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

      <div className="row" id="video-size-row">
        <div className="field" id="field-video-width">
          <label htmlFor="video-width-input">宽度 (px)</label>
          <input
            id="video-width-input" type="number" step={isMiniMaxH3 ? 32 : LIMITS.size.multiple}
            min={LIMITS.size.min} max={LIMITS.size.max}
            value={width} onChange={(e) => setWidth(e.target.value)}
          />
        </div>
        <div className="field" id="field-video-height">
          <label htmlFor="video-height-input">高度 (px)</label>
          <input
            id="video-height-input" type="number" step={isMiniMaxH3 ? 32 : LIMITS.size.multiple}
            min={LIMITS.size.min} max={LIMITS.size.max}
            value={height} onChange={(e) => setHeight(e.target.value)}
          />
        </div>
      </div>

      <div className="preset-row" id="video-size-presets">
        {(sizePresets ?? []).map(([presetWidth, presetHeight]) => (
          <button
            key={`${presetWidth}x${presetHeight}`}
            id={`video-preset-${presetWidth}x${presetHeight}`}
            type="button"
            className="chip"
            onClick={() => { setWidth(String(presetWidth)); setHeight(String(presetHeight)); }}
          >
            {presetWidth === presetHeight
              ? `${presetWidth}²`
              : `${presetWidth}×${presetHeight}`}
          </button>
        ))}
      </div>

      <div className="row" id="video-duration-fps-row">
        <div className="field" id="field-video-duration">
          <label htmlFor="video-duration-input">视频长度 (秒)</label>
          <input
            id="video-duration-input" type="number" step={1}
            min={LIMITS.duration.min} max={LIMITS.duration.max}
            value={duration} onChange={(e) => setDuration(e.target.value)}
          />
        </div>
        <div className="field" id="field-video-fps">
          <label htmlFor="video-fps-input">帧率 FPS</label>
          <input
            id="video-fps-input" type="number" step={1}
            min={LIMITS.fps.min} max={LIMITS.fps.max}
            disabled={isMiniMaxH3}
            value={fps} onChange={(e) => setFps(e.target.value)}
          />
        </div>
      </div>
      <p className="form-hint" id="video-frames-hint">
        {isMiniMaxH3 ? "对齐后总帧数" : `总帧数 length = ${duration} × ${fps} + 1`} = <strong>{framesComputable ? totalFrames : "-"}</strong>
        {framesValid ? (
          isMiniMaxH3 ? "(满足 17n+5)" : "(满足 4n+1)"
        ) : (
          <span className="form-hint-error">
            {isMiniMaxH3 ? ",MiniMax H3 帧率固定为 24" : ",不满足 4n+1:时长 × 帧率 必须是 4 的倍数"}
          </span>
        )}
      </p>

      <div className="row" id="video-steps-count-row">
        <div className="field" id="field-video-steps">
          <label htmlFor="video-steps-input">采样步数 Steps</label>
          <input
            id="video-steps-input" type="number"
            min={LIMITS.steps.min} max={LIMITS.steps.max}
            value={steps} onChange={(e) => setSteps(e.target.value)}
          />
        </div>
        <div className="field" id="field-video-count">
          <label htmlFor="video-count-input">批次数量</label>
          <input
            id="video-count-input" type="number"
            min={LIMITS.count.min} max={LIMITS.count.max}
            value={count} onChange={(e) => setCount(e.target.value)}
          />
        </div>
        <div className="field" id="field-video-seed">
          <label htmlFor="video-seed-input">Seed(-1 随机)</label>
          <input
            id="video-seed-input" type="number" step={1} min={-1}
            value={seed} onChange={(e) => setSeed(e.target.value)}
          />
        </div>
      </div>

      <div className="row" id="video-fixed-params-row">
        <div className="field" id="field-video-sampler">
          <label htmlFor="video-sampler-select">采样器 Sampler</label>
          <select
            id="video-sampler-select"
            value={sampler}
            onChange={(e) => setSampler(e.target.value)}
          >
            {samplerOptions.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div className="field" id="field-video-scheduler">
          <label htmlFor="video-scheduler-select">调度器 Scheduler</label>
          <select
            id="video-scheduler-select"
            value={scheduler}
            onChange={(e) => setScheduler(e.target.value)}
          >
            {schedulerOptions.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div className="field" id="field-video-cfg">
          <label htmlFor="video-cfg-input">CFG</label>
          <input
            id="video-cfg-input" type="number" min={0} step="any"
            disabled={isMiniMaxH3}
            value={cfg} onChange={(e) => setCfg(e.target.value)}
          />
        </div>
        <div className="field" id="field-video-shift">
          <label htmlFor="video-shift-input">Shift (0–100)</label>
          <input
            id="video-shift-input" type="number" min={0} max={100} step="any"
            value={shift} onChange={(e) => setShift(e.target.value)}
          />
        </div>
        <div className="field" id="field-video-latent-multiplier">
          <label htmlFor="video-latent-multiplier-input">Latent multiplier</label>
          <input
            id="video-latent-multiplier-input" type="number" min={0} step="any"
            value={latentMultiplier} onChange={(e) => setLatentMultiplier(e.target.value)}
          />
        </div>
      </div>
      </div>

      {error && <div id="video-form-error" className="form-error">{error}</div>}

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
