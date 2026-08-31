"""运行在 Mac 源码版 ComfyUI Python 中的远端文本编码服务。"""

import argparse
import base64
import importlib.util
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def unpack(value, torch):
    return torch.load(io.BytesIO(base64.b64decode(value)), map_location="cpu", weights_only=True)


def pack(value, torch):
    out = io.BytesIO()
    torch.save(value, out)
    return base64.b64encode(out.getvalue()).decode("ascii")


class Encoder:
    def __init__(self, args):
        root = Path(args.comfy_root).expanduser().resolve()
        sys.path.insert(0, str(root))
        import comfy.options
        comfy.options.args_parsing = False
        import comfy.cli_args
        comfy.cli_args.args.disable_xformers = True
        comfy.cli_args.args.use_pytorch_cross_attention = True
        import folder_paths, nodes, torch
        self.torch = torch
        self.folder_paths = folder_paths
        self.nodes = nodes
        self.encoder_id = args.encoder_id
        self.clip_path = Path(args.clip_path).expanduser().resolve()
        self.clip = None
        self.lock = threading.RLock()
        self.krea2edit = None

    def load_clip(self, encoder_id, clip_type):
        with self.lock:
            if encoder_id != self.encoder_id:
                raise ValueError(f"未知 encoder_id: {encoder_id}")
            if self.clip is not None:
                return False
            if not self.clip_path.is_file():
                raise FileNotFoundError(f"找不到 text encoder: {self.clip_path}")
            self.folder_paths.add_model_folder_path("text_encoders", str(self.clip_path.parent), is_default=True)
            name = self.clip_path.name
            self.clip = self.nodes.CLIPLoader().load_clip(name, clip_type)[0]
            return True

    def _nodes(self):
        if self.krea2edit is not None:
            return self.krea2edit
        root = Path(self.folder_paths.__file__).resolve().parent / "custom_nodes" / "comfyui-krea2edit"
        entry = root / "__init__.py"
        spec = importlib.util.spec_from_file_location("remote_krea2edit", entry, submodule_search_locations=[str(root)])
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载节点: {entry}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["remote_krea2edit"] = module
        spec.loader.exec_module(module)
        self.krea2edit = module
        return module

    def text(self, prompt, negative_prompt, cfg, reuse):
        with self.lock, self.torch.inference_mode():
            enc = self.nodes.CLIPTextEncode()
            positive = enc.encode(self.clip, prompt)[0]
            negative = positive if reuse and cfg == 1.0 else enc.encode(self.clip, negative_prompt)[0]
            return positive, negative

    def grounded(self, prompt, negative_prompt, image, image_b, grounding_px, cfg, reuse):
        with self.lock, self.torch.inference_mode():
            mapping = self._nodes().NODE_CLASS_MAPPINGS
            node = mapping["Krea2EditGroundedEncode"]()
            positive = node.encode(self.clip, prompt, image=image, image_b=image_b, grounding_px=int(grounding_px), system_prompt="")[0]
            if reuse and cfg == 1.0:
                negative = positive
            elif not negative_prompt:
                negative = self.nodes.ConditioningZeroOut().zero_out(positive)[0]
            else:
                negative = node.encode(self.clip, negative_prompt, image=image, image_b=image_b, grounding_px=int(grounding_px), system_prompt="")[0]
            return positive, negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", default="~/comfyui")
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    encoder = Encoder(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return
        def send_json(self, payload, status=200):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/v1/ping":
                    self.send_json({"ok": True, "encoder_id": encoder.encoder_id}); return
                if self.path == "/v1/load_clip":
                    loaded = encoder.load_clip(body.get("encoder_id"), body.get("clip_type")); self.send_json({"ok": True, "loaded": loaded}); return
                if self.path == "/v1/release":
                    encoder.clip = None; self.send_json({"ok": True}); return
                if self.path == "/v1/encode_text":
                    p, n = encoder.text(body["prompt"], body.get("negative_prompt", ""), body.get("cfg", 1), body.get("reuse_negative_at_cfg_one", False)); self.send_json({"ok": True, "positive": pack(p, encoder.torch), "negative": pack(n, encoder.torch)}); return
                if self.path == "/v1/encode_grounded":
                    image = unpack(body["image"], encoder.torch); image_b = unpack(body["image_b"], encoder.torch) if body.get("image_b") else None
                    p, n = encoder.grounded(body["prompt"], body.get("negative_prompt", ""), image, image_b, body.get("grounding_px", 768), body.get("cfg", 1), body.get("reuse_negative_at_cfg_one", False)); self.send_json({"ok": True, "positive": pack(p, encoder.torch), "negative": pack(n, encoder.torch)}); return
                raise ValueError("未知请求路径")
            except Exception as exc:
                self.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)

    print(f"remote encoder listening on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
