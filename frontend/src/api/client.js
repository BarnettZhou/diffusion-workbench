// API 客户端。默认同源(由 vite dev proxy 转发到 127.0.0.1:8188);
// 如需直连其他地址,构建时设置 VITE_API_BASE。
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (Array.isArray(body.detail)) {
        // pydantic 校验错误的 detail 是对象数组,提取 msg 拼成可读文本
        detail = body.detail.map((item) => item?.msg ?? String(item)).join("; ");
      } else if (body.detail) {
        detail = String(body.detail);
      }
    } catch {
      /* 保留默认错误信息 */
    }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  get: (path) => request(path),
  status: () => request("/api/v1/status"),
  modes: () => request("/api/v1/modes"),
  samplingOptions: () => request("/api/v1/sampling-options"),
  remoteEncoders: () => request("/api/v1/remote-encoders"),
  upscaleOptions: () => request("/api/v1/upscale-options"),
  upscaleModels: () => request("/api/v1/upscale-models"),
  resources: (mode, kind) => request(`/api/v1/resources/${mode}/${kind}`),
  jobs: (limit = 50) => request(`/api/v1/jobs?limit=${limit}`),
  job: (id) => request(`/api/v1/jobs/${id}`),
  submit: (payload) =>
    request("/api/v1/jobs", { method: "POST", body: JSON.stringify(payload) }),
  stop: () => request("/api/v1/control/stop", { method: "POST" }),
  skip: () => request("/api/v1/control/skip", { method: "POST" }),
  // 释放 Worker 已加载的模型资源;任务或队列非空时后端返回 409
  releaseResources: () => request("/api/v1/control/release", { method: "POST" }),
  imageUrl: (jobId) => `${API_BASE}/api/v1/images/${jobId}`,
  upscaledImageUrl: (jobId) => `${API_BASE}/api/v1/images/${jobId}/upscaled`,
  imageMetadata: (jobId) => request(`/api/v1/images/${jobId}/metadata`),
  album: (limit, cursor, dir = "output", subdir = null) =>
    request(
      `/api/v1/album?limit=${limit}&dir=${encodeURIComponent(dir)}${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}${subdir ? `&subdir=${encodeURIComponent(subdir)}` : ""}`,
    ),
  albumSubdirs: (dir = "output") =>
    request(`/api/v1/album/subdirs?dir=${encodeURIComponent(dir)}`),
  albumImageUrl: (id, dir = "output") =>
    `${API_BASE}/api/v1/album/image/${encodeURIComponent(id)}?dir=${encodeURIComponent(dir)}`,
  albumMetadataUrl: (id, dir = "output") =>
    `/api/v1/album/image/${encodeURIComponent(id)}/metadata?dir=${encodeURIComponent(dir)}`,
  // 视频封面(服务端 ffmpeg 抽首帧):iOS 不渲染 <video> 首帧,封面须用静态图
  albumPosterUrl: (id, dir = "output") =>
    `${API_BASE}/api/v1/album/image/${encodeURIComponent(id)}/poster?dir=${encodeURIComponent(dir)}`,
  // 图片缩略图(服务端懒生成 200×200 JPEG):grid 用,lightbox 仍用 albumImageUrl 取原图
  albumThumbnailUrl: (id, dir = "output") =>
    `${API_BASE}/api/v1/album/image/${encodeURIComponent(id)}/thumbnail?dir=${encodeURIComponent(dir)}`,
  deleteAlbumImage: (id, dir = "output") =>
    request(
      `/api/v1/album/image/${encodeURIComponent(id)}?dir=${encodeURIComponent(dir)}`,
      { method: "DELETE" },
    ),
  // 批量删除:返回 {deleted: [relpath], failed: [{relpath, reason}]}
  batchDeleteAlbumImages: (relpaths, dir = "output") =>
    request("/api/v1/album/batch-delete", {
      method: "POST",
      body: JSON.stringify({ dir, relpaths }),
    }),
  albumDirs: () => request("/api/v1/album/dirs"),
  addAlbumDir: (payload) =>
    request("/api/v1/album/dirs", { method: "POST", body: JSON.stringify(payload) }),
  renameAlbumDir: (id, payload) =>
    request(`/api/v1/album/dirs/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  deleteAlbumDir: (id) =>
    request(`/api/v1/album/dirs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  settings: () => request("/api/v1/settings"),
  updateSettings: (patch) =>
    request("/api/v1/settings", { method: "PUT", body: JSON.stringify(patch) }),
  promptAssist: (payload) =>
    request("/api/v1/prompt-assist", { method: "POST", body: JSON.stringify(payload) }),
  // 对话式提示词生成:返回原始 Response,由调用方按 SSE 流式读取
  promptAssistChat: (payload) =>
    fetch(`${API_BASE}/api/v1/prompt-assist/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  llmRequests: (page, pageSize = 10) =>
    request(`/api/v1/llm/requests?page=${page}&page_size=${pageSize}`),
  models: (mode) => request(`/api/v1/models/${mode}`),
  updateModelInfo: (mode, name, payload) =>
    request(`/api/v1/models/${mode}/${encodeURIComponent(name)}/info`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  fetchModelQuant: (mode, name) =>
    request(`/api/v1/models/${mode}/${encodeURIComponent(name)}/quant`, {
      method: "POST",
    }),
  uploadModelCover: (mode, name, file) =>
    fetch(
      `${API_BASE}/api/v1/models/${mode}/${encodeURIComponent(name)}/cover`,
      { method: "PUT", headers: { "Content-Type": file.type }, body: file },
    ).then((response) => {
      if (!response.ok) throw new Error(`封面上传失败(HTTP ${response.status})`);
      return response.json();
    }),
  // ---------- 生图 LoRA(krea2 / zit 可选 LoRA 画册) ----------
  loras: (mode) => request(`/api/v1/loras/${mode}`),
  updateLoraInfo: (mode, name, payload) =>
    request(`/api/v1/loras/${mode}/${encodeURIComponent(name)}/info`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  uploadLoraCover: (mode, name, file) =>
    fetch(
      `${API_BASE}/api/v1/loras/${mode}/${encodeURIComponent(name)}/cover`,
      { method: "PUT", headers: { "Content-Type": file.type }, body: file },
    ).then((response) => {
      if (!response.ok) throw new Error(`封面上传失败(HTTP ${response.status})`);
      return response.json();
    }),
  // ---------- 视频生成 ----------
  // 视频模型分类与能力:{video_model,label,generation_types,requires_input_image}
  videoModels: () => request("/api/v1/video/models"),
  // 指定视频模型的资源:{"models": [...], "vaes": [...], "text_encoders": [...]}
  videoResources: (videoModel) =>
    request(`/api/v1/video/models/${encodeURIComponent(videoModel)}/resources`),
  submitVideo: (payload) =>
    request("/api/v1/video/jobs", { method: "POST", body: JSON.stringify(payload) }),
  videoJob: (id) => request(`/api/v1/video/jobs/${id}`),
  videoUrl: (jobId) => `${API_BASE}/api/v1/videos/${jobId}`,
  // 受控上传 I2V 输入图片(二进制 body),返回 {id, url}
  uploadVideoInputImage: (file) =>
    fetch(`${API_BASE}/api/v1/video/input-images`, {
      method: "POST",
      headers: { "Content-Type": file.type },
      body: file,
    }).then(async (response) => {
      if (!response.ok) {
        let detail = `输入图片上传失败(HTTP ${response.status})`;
        try {
          const body = await response.json();
          if (body.detail) detail = String(body.detail);
        } catch { /* 保留默认错误信息 */ }
        throw new Error(detail);
      }
      return response.json();
    }),
  // 受控上传 Ref2VA 参考视频(二进制 body),返回 {id, url}
  uploadVideoInputVideo: (file) =>
    fetch(`${API_BASE}/api/v1/video/input-videos`, {
      method: "POST",
      headers: { "Content-Type": file.type },
      body: file,
    }).then(async (response) => {
      if (!response.ok) {
        let detail = `参考视频上传失败(HTTP ${response.status})`;
        try {
          const body = await response.json();
          if (body.detail) detail = String(body.detail);
        } catch { /* 保留默认错误信息 */ }
        throw new Error(detail);
      }
      return response.json();
    }),
  // 受控上传 Ref2VA 参考音频(二进制 body),返回 {id, url}
  uploadVideoInputAudio: (file) =>
    fetch(`${API_BASE}/api/v1/video/input-audios`, {
      method: "POST",
      headers: { "Content-Type": file.type },
      body: file,
    }).then(async (response) => {
      if (!response.ok) {
        let detail = `参考音频上传失败(HTTP ${response.status})`;
        try {
          const body = await response.json();
          if (body.detail) detail = String(body.detail);
        } catch { /* 保留默认错误信息 */ }
        throw new Error(detail);
      }
      return response.json();
    }),
  // 把相册图片直接导入为 I2V 输入图片(服务端本地复制),返回 {id, url}
  importVideoInputFromAlbum: (id, dir = "output") =>
    request("/api/v1/video/input-images/from-album", {
      method: "POST",
      body: JSON.stringify({ id, dir }),
    }),
  // 把生成/编辑任务的输出图直接导入为 I2V 输入图片(服务端本地复制),返回 {id, url}
  importVideoInputFromJob: (jobId) =>
    request("/api/v1/video/input-images/from-job", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId }),
    }),
  // ---------- Krea2 图像编辑 ----------
  // 编辑能力开关与默认值:{"enabled": bool, "defaults": {"grounding_px", "ref_boost"}}
  editInfo: () => request("/api/v1/edit/info"),
  // 受控上传编辑输入图片(二进制 body),返回 {id, url};与视频上传同构
  uploadEditInputImage: (file) =>
    fetch(`${API_BASE}/api/v1/edit/input-images`, {
      method: "POST",
      headers: { "Content-Type": file.type },
      body: file,
    }).then(async (response) => {
      if (!response.ok) {
        let detail = `输入图片上传失败(HTTP ${response.status})`;
        try {
          const body = await response.json();
          if (body.detail) detail = String(body.detail);
        } catch { /* 保留默认错误信息 */ }
        throw new Error(detail);
      }
      return response.json();
    }),
  // 把相册图片直接导入为编辑输入图片(服务端本地复制),返回 {id, url}
  importEditInputFromAlbum: (id, dir = "output") =>
    request("/api/v1/edit/input-images/from-album", {
      method: "POST",
      body: JSON.stringify({ id, dir }),
    }),
  // 把生成/编辑任务的输出图直接导入为编辑输入图片(服务端本地复制),返回 {id, url}
  importEditInputFromJob: (jobId) =>
    request("/api/v1/edit/input-images/from-job", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId }),
    }),
  // 提交编辑任务;服务端按 krea2 资源列表 + 加载 edit_lora 生成 edit-krea2 任务
  submitEdit: (payload) =>
    request("/api/v1/edit/jobs", { method: "POST", body: JSON.stringify(payload) }),
  // 提交参考图重排任务;复用 krea2 资源,参考图为受控上传 id(1-4 张)
  submitRebalance: (payload) =>
    request("/api/v1/edit/rebalance-jobs", { method: "POST", body: JSON.stringify(payload) }),
  // ---------- 图片反推 ----------
  // 同步反推(复用 krea2 Qwen3-VL);输入图片走 uploadEditInputImage 的受控 id
  caption: (payload) =>
    request("/api/v1/caption", { method: "POST", body: JSON.stringify(payload) }),
  // API 反推:走设置页配置的外部视觉模型接口(Ollama / OpenAI 兼容),不占用本地 GPU
  captionRemote: (payload) =>
    request("/api/v1/caption/remote", { method: "POST", body: JSON.stringify(payload) }),
  // API 反推请求记录(分页)
  captionRemoteRequests: (page, pageSize = 10) =>
    request(`/api/v1/caption/remote/requests?page=${page}&page_size=${pageSize}`),
  // 远端模型发现:body {interface, base_url, api_key} → {"models": [str]}
  remoteModels: (payload) =>
    request("/api/v1/remote/models", { method: "POST", body: JSON.stringify(payload) }),
  // ---------- 设置页:视频模型卡片(镜像图片 models 端点,不支持 alias) ----------
  videoModelCards: (videoModel) =>
    request(`/api/v1/video-models/${encodeURIComponent(videoModel)}`),
  updateVideoModelInfo: (videoModel, name, payload) =>
    request(
      `/api/v1/video-models/${encodeURIComponent(videoModel)}/${encodeURIComponent(name)}/info`,
      { method: "PUT", body: JSON.stringify(payload) },
    ),
  fetchVideoModelQuant: (videoModel, name) =>
    request(
      `/api/v1/video-models/${encodeURIComponent(videoModel)}/${encodeURIComponent(name)}/quant`,
      { method: "POST" },
    ),
  uploadVideoModelCover: (videoModel, name, file) =>
    fetch(
      `${API_BASE}/api/v1/video-models/${encodeURIComponent(videoModel)}/${encodeURIComponent(name)}/cover`,
      { method: "PUT", headers: { "Content-Type": file.type }, body: file },
    ).then((response) => {
      if (!response.ok) throw new Error(`封面上传失败(HTTP ${response.status})`);
      return response.json();
    }),
};

export function eventsUrl() {
  const base = API_BASE || location.origin;
  return `${base.replace(/^http/, "ws")}/api/v1/events`;
}
