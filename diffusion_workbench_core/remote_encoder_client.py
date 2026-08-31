"""远端 ComfyUI text encoder 的轻量 HTTP/JSON 客户端。

协议使用单次 POST 请求和 base64 编码的 torch.save payload，避免主项目强制引入
grpcio；服务端包可独立复制到 Mac 运行。
"""

import base64
import io
import json
import time
import urllib.error
import urllib.request


def pack(value, torch):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def unpack(value, torch):
    raw = base64.b64decode(value.encode("ascii"))
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


class RemoteEncoderError(RuntimeError):
    pass


class RemoteEncoderClient:
    def __init__(self, host, port, *, connect_timeout=5.0, request_timeout=60.0, torch):
        self.base_url = f"http://{host}:{int(port)}"
        self.connect_timeout = float(connect_timeout)
        self.request_timeout = float(request_timeout)
        self.torch = torch
        self.current_encoder = None

    def _post(self, path, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last = None
        for delay in (0.0, 1.0, 2.0):
            if delay:
                time.sleep(delay)
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict) or not result.get("ok", False):
                    raise RemoteEncoderError(result.get("error", "远端编码服务返回失败"))
                return result
            except (urllib.error.URLError, TimeoutError, OSError, RemoteEncoderError) as exc:
                last = exc
                if isinstance(exc, RemoteEncoderError):
                    break
        raise RemoteEncoderError(f"远端编码服务不可用: {last}") from last

    def ping(self):
        return self._post("/v1/ping", {})

    def load_clip(self, encoder_id, clip_type):
        result = self._post("/v1/load_clip", {"encoder_id": encoder_id, "clip_type": clip_type})
        self.current_encoder = (encoder_id, clip_type)
        return result

    def encode_text(self, prompt, negative_prompt, *, cfg, reuse_negative_at_cfg_one, cache_scope):
        result = self._post("/v1/encode_text", {
            "encoder_id": self.current_encoder[0] if self.current_encoder else None,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "cfg": float(cfg),
            "reuse_negative_at_cfg_one": bool(reuse_negative_at_cfg_one),
            "cache_scope": cache_scope,
        })
        return unpack(result["positive"], self.torch), unpack(result["negative"], self.torch), result

    def encode_grounded(self, prompt, negative_prompt, image, image_b, *, grounding_px, cfg, reuse_negative_at_cfg_one):
        result = self._post("/v1/encode_grounded", {
            "encoder_id": self.current_encoder[0] if self.current_encoder else None,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": pack(image.cpu(), self.torch),
            "image_b": pack(image_b.cpu(), self.torch) if image_b is not None else None,
            "grounding_px": int(grounding_px),
            "cfg": float(cfg),
            "reuse_negative_at_cfg_one": bool(reuse_negative_at_cfg_one),
        })
        return unpack(result["positive"], self.torch), unpack(result["negative"], self.torch), result

    def release(self):
        try:
            self._post("/v1/release", {})
        finally:
            self.current_encoder = None
