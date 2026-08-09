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
  imageUrl: (jobId) => `${API_BASE}/api/v1/images/${jobId}`,
  upscaledImageUrl: (jobId) => `${API_BASE}/api/v1/images/${jobId}/upscaled`,
  imageMetadata: (jobId) => request(`/api/v1/images/${jobId}/metadata`),
  album: (limit, cursor, dir = "output") =>
    request(
      `/api/v1/album?limit=${limit}&dir=${encodeURIComponent(dir)}${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`,
    ),
  albumImageUrl: (id, dir = "output") =>
    `${API_BASE}/api/v1/album/image/${encodeURIComponent(id)}?dir=${encodeURIComponent(dir)}`,
  albumMetadataUrl: (id, dir = "output") =>
    `/api/v1/album/image/${encodeURIComponent(id)}/metadata?dir=${encodeURIComponent(dir)}`,
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
};

export function eventsUrl() {
  const base = API_BASE || location.origin;
  return `${base.replace(/^http/, "ws")}/api/v1/events`;
}
