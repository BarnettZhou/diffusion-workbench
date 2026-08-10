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
      if (body.detail) detail = String(body.detail);
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
  deleteAlbumImage: (id, dir = "output") =>
    request(
      `/api/v1/album/image/${encodeURIComponent(id)}?dir=${encodeURIComponent(dir)}`,
      { method: "DELETE" },
    ),
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
  // ---------- 视频生成 ----------
  // 视频模型分类与能力:{video_model,label,generation_types,requires_input_image}
  videoModels: () => request("/api/v1/video/models"),
  // 指定视频模型的资源:{"models": [{index,name,display_name,alias}], "vaes": [...]}
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
  // 把相册图片直接导入为 I2V 输入图片(服务端本地复制),返回 {id, url}
  importVideoInputFromAlbum: (id, dir = "output") =>
    request("/api/v1/video/input-images/from-album", {
      method: "POST",
      body: JSON.stringify({ id, dir }),
    }),
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
