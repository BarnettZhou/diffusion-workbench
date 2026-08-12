import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, PngImagePlugin


PNG_METADATA_KEY = "diffusion_workbench"
PNG_METADATA_SCHEMA_VERSION = 4
SUPPORTED_PNG_METADATA_SCHEMA_VERSIONS = {1, 2, 3, PNG_METADATA_SCHEMA_VERSION}
# ComfyUI 原生生成 PNG 把 API prompt 节点图写进这个 tEXt 块
COMFYUI_METADATA_KEY = "prompt"


class ResourceFingerprintCache:
    """Cache full-file hashes while detecting files replaced at the same path."""

    def __init__(self) -> None:
        self._sha256: dict[tuple[str, int, int], str] = {}

    def describe(self, value: str | Path) -> dict[str, str | int]:
        path = Path(value).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        digest = self._sha256.get(key)
        if digest is None:
            digest = _sha256_file(path)
            self._sha256 = {
                cached_key: cached_digest
                for cached_key, cached_digest in self._sha256.items()
                if cached_key[0] != str(path)
            }
            self._sha256[key] = digest
        return {
            "filename": path.name,
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
            "sha256": digest,
        }


def build_generation_metadata(
    command: Mapping[str, Any],
    runtime_versions: Mapping[str, str | None],
    performance: Mapping[str, float] | None = None,
    resources: Mapping[str, Mapping[str, str | int]] | None = None,
    artifact_kind: str = "original",
    artifact_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build the stable, versioned payload embedded in generated PNG files."""

    artifact_width, artifact_height = artifact_size or (
        int(command["width"]),
        int(command["height"]),
    )
    metadata = {
        "schema_version": PNG_METADATA_SCHEMA_VERSION,
        "generator": {
            "name": "diffusion-workbench",
            "version": command.get("workbench_version", "unknown"),
        },
        "job": {
            "id": command["job_id"],
            "batch_id": command.get("batch_id"),
        },
        "artifact": {
            "kind": artifact_kind,
            "width": int(artifact_width),
            "height": int(artifact_height),
        },
        "parameters": {
            "mode": command["mode"],
            "prompt": command["prompt"],
            "negative_prompt": command.get("negative_prompt", ""),
            "width": int(command["width"]),
            "height": int(command["height"]),
            "steps": int(command["steps"]),
            "seed": int(command["seed"]),
            "cfg": float(command["cfg"]),
            "sampler": command["sampler"],
            "scheduler": command["scheduler"],
            "denoise": 1.0,
            "negative_conditioning": (
                "encoded_negative_prompt"
                if command["mode"] == "zib" or float(command["cfg"]) != 1.0
                else "positive_reused"
            ),
            "upscale": dict(command.get("upscale") or {"enabled": False}),
        },
        "resources": {
            "diffusion_model": _resource_or_default(
                resources, "diffusion_model", command["model_path"]
            ),
            "vae": _optional_resource_or_default(
                resources, "vae", command.get("vae_path")
            ),
            "text_encoder": _optional_resource_or_default(
                resources, "text_encoder", command.get("text_encoder_path")
            ),
            "clip_type": command["clip_type"],
            "model_loader": command.get("model_loader", "components"),
        },
        "runtime": dict(runtime_versions),
    }
    if performance is not None:
        metadata["performance"] = dict(performance)
    if resources is not None and "upscale_model" in resources:
        metadata["resources"]["upscale_model"] = dict(resources["upscale_model"])
    return metadata


def create_png_info(metadata: Mapping[str, Any]) -> PngImagePlugin.PngInfo:
    png_info = PngImagePlugin.PngInfo()
    png_info.add_itxt(
        PNG_METADATA_KEY,
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
    )
    return png_info


def read_generation_metadata(path: str | Path) -> dict[str, Any] | None:
    """Read and validate diffusion-workbench metadata from a PNG image."""

    with Image.open(path) as image:
        raw = image.info.get(PNG_METADATA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("PNG 中的 diffusion-workbench 元数据不是文本")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("PNG 中的 diffusion-workbench 元数据不是有效 JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("PNG 中的 diffusion-workbench 元数据必须是对象")
    schema_version = metadata.get("schema_version")
    if schema_version not in SUPPORTED_PNG_METADATA_SCHEMA_VERSIONS:
        raise ValueError(
            "不支持的 diffusion-workbench PNG 元数据版本: "
            f"{schema_version!r}"
        )
    return metadata


def _resource(value: str | Path) -> dict[str, str]:
    path = Path(value)
    return {"filename": path.name, "path": str(path)}


def _resource_or_default(
    resources: Mapping[str, Mapping[str, str | int]] | None,
    key: str,
    value: str | Path,
) -> dict[str, str | int]:
    return dict(resources[key]) if resources is not None else _resource(value)


def _optional_resource_or_default(
    resources: Mapping[str, Mapping[str, str | int]] | None,
    key: str,
    value: str | Path | None,
) -> dict[str, str | int] | None:
    if resources is not None:
        resource = resources.get(key)
        return dict(resource) if resource is not None else None
    return _resource(value) if value else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_comfyui_metadata(path: str | Path) -> dict[str, Any] | None:
    """Read generation parameters from a PNG produced by ComfyUI itself.

    ComfyUI 把 API prompt 形式的完整节点图写进 PNG 的 ``prompt`` tEXt 块，
    本函数从中提取加载器（UNET/VAE/CLIP）名称与第一个采样器的参数，
    返回与相册展示对齐的公开形状；取不到的字段为 None。
    非 ComfyUI 生成的 PNG（没有 ``prompt`` 块或格式不符）返回 None。
    """

    with Image.open(path) as image:
        raw = image.info.get(COMFYUI_METADATA_KEY)
        size = image.size
    if raw is None or not isinstance(raw, str):
        return None
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(graph, dict):
        return None
    nodes = {
        str(node_id): node
        for node_id, node in graph.items()
        if isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and isinstance(node.get("inputs"), dict)
    }
    if not nodes:
        return None

    sampler = _first_node(nodes, "KSampler", "KSamplerAdvanced")
    unet = _first_node(nodes, "UNETLoader")
    vae = _first_node(nodes, "VAELoader")
    clip = _first_node(nodes, "CLIPLoader")
    if sampler is None and unet is None:
        return None

    model_name = _base_name(_scalar(unet["inputs"].get("unet_name"))) if unet else None
    metadata: dict[str, Any] = {
        "source": "comfyui",
        "mode": _guess_comfyui_mode(model_name) if model_name else None,
        "prompt": None,
        "negative_prompt": None,
        "width": size[0],
        "height": size[1],
        "steps": None,
        "cfg": None,
        "sampler": None,
        "scheduler": None,
        "seed": None,
        "model_name": model_name,
        "vae_name": _base_name(_scalar(vae["inputs"].get("vae_name"))) if vae else None,
        "text_encoder_name": (
            _base_name(_scalar(clip["inputs"].get("clip_name"))) if clip else None
        ),
    }
    if sampler is not None:
        inputs = sampler["inputs"]
        metadata["steps"] = _scalar(inputs.get("steps"))
        metadata["cfg"] = _float_or_none(_scalar(inputs.get("cfg")))
        metadata["sampler"] = _scalar(inputs.get("sampler_name"))
        metadata["scheduler"] = _scalar(inputs.get("scheduler"))
        # KSamplerAdvanced 的随机种子字段叫 noise_seed
        metadata["seed"] = _scalar(inputs.get("seed", inputs.get("noise_seed")))
        metadata["prompt"] = _encode_text(nodes, inputs.get("positive"))
        metadata["negative_prompt"] = _encode_text(nodes, inputs.get("negative"))
    return metadata


def _first_node(
    nodes: Mapping[str, Mapping[str, Any]], *class_types: str
) -> Mapping[str, Any] | None:
    """按文档序取第一个 class_type 匹配的节点（多轮采样只取第一个）。"""

    for node in nodes.values():
        if node["class_type"] in class_types:
            return node
    return None


def _scalar(value: Any) -> Any:
    """节点输入若是链接（list 形式）则取不到字面量，返回 None。"""

    if isinstance(value, (list, dict)):
        return None
    return value


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _base_name(value: Any) -> str | None:
    """模型名可能带子目录（Windows 反斜杠），展示只取文件名。"""

    if not isinstance(value, str) or not value:
        return None
    return value.replace("\\", "/").split("/")[-1]


def _guess_comfyui_mode(model_name: str) -> str | None:
    """按 unet 文件名 best-effort 推断模式，匹配顺序：krea2 → zib → sdxl → zit。"""

    lowered = model_name.lower()
    if "krea" in lowered:
        return "krea2"
    if "zib" in lowered or "z-image-base" in lowered:
        return "zib"
    if "sdxl" in lowered:
        return "sdxl"
    if "zit" in lowered or "zimage" in lowered or "z-image-turbo" in lowered:
        return "zit"
    return None


def _encode_text(
    nodes: Mapping[str, Mapping[str, Any]], link: Any
) -> str | None:
    """沿采样器 positive/negative 链接找到 CLIPTextEncode 并取其 text。"""

    node = _link_node(nodes, link)
    if node is None:
        return None
    # negative 常经 ConditioningZeroOut 透传，需再跟一层 conditioning 输入
    if node["class_type"] == "ConditioningZeroOut":
        node = _link_node(nodes, node["inputs"].get("conditioning"))
        if node is None:
            return None
    text = node["inputs"].get("text")
    return text if isinstance(text, str) else None


def _link_node(
    nodes: Mapping[str, Mapping[str, Any]], link: Any
) -> Mapping[str, Any] | None:
    if not isinstance(link, (list, tuple)) or len(link) < 1:
        return None
    return nodes.get(str(link[0]))
