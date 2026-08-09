import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, PngImagePlugin


PNG_METADATA_KEY = "diffusion_workbench"
PNG_METADATA_SCHEMA_VERSION = 4
SUPPORTED_PNG_METADATA_SCHEMA_VERSIONS = {1, 2, 3, PNG_METADATA_SCHEMA_VERSION}


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
