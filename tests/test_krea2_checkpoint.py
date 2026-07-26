import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from demo_txt2img import validate_krea2_scaled_fp8


def quant_metadata(**overrides) -> torch.Tensor:
    config = {"format": "float8_e4m3fn", **overrides}
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, str(path))


def quant_key(weight_key: str) -> str:
    return weight_key.removesuffix("weight") + "comfy_quant"


def standard_checkpoint_tensors() -> dict[str, torch.Tensor]:
    tensors = {}
    for index in range(256):
        weight_key = f"model.diffusion_model.test_linear_{index}.weight"
        tensors[weight_key] = torch.zeros((1, 1), dtype=torch.float8_e4m3fn)
        tensors[weight_key + "_scale"] = torch.tensor(1.0, dtype=torch.float32)
        tensors[quant_key(weight_key)] = quant_metadata(full_precision_matrix_mult=True)
    for index in range(174):
        tensors[f"model.diffusion_model.test_norm_{index}"] = torch.ones(1, dtype=torch.bfloat16)
    return tensors


def structure_signature(tensors: dict[str, torch.Tensor]) -> str:
    dtype_names = {
        torch.float8_e4m3fn: "F8_E4M3",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.uint8: "U8",
    }
    entries = [
        f"{key}|{dtype_names[tensor.dtype]}|"
        f"{','.join(map(str, tensor.shape))}"
        for key, tensor in tensors.items()
        if not key.endswith(("_scale", ".comfy_quant"))
    ]
    return sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()


class Krea2CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "model.safetensors"
        self.cache_path = Path(self.temp_dir.name) / "checkpoint_formats.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_accepts_standard_scaled_fp8(self):
        tensors = standard_checkpoint_tensors()
        write_checkpoint(self.path, tensors)

        with patch("demo_txt2img.KREA2_STRUCTURE_SIGNATURE", structure_signature(tensors)):
            first_report = validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)
            second_report = validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)

        cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        record = cache["checkpoints"][str(self.path.resolve())]
        self.assertFalse(first_report["cache_hit"])
        self.assertTrue(second_report["cache_hit"])
        self.assertTrue(record["valid"])
        self.assertEqual(record["size"], self.path.stat().st_size)
        self.assertEqual(record["mtime_ns"], self.path.stat().st_mtime_ns)
        self.assertEqual(record["format"], "scaled_fp8_e4m3fn")

    def test_rescans_when_checkpoint_changes(self):
        original_tensors = standard_checkpoint_tensors()
        write_checkpoint(self.path, original_tensors)
        with patch(
            "demo_txt2img.KREA2_STRUCTURE_SIGNATURE", structure_signature(original_tensors)
        ):
            validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)

        tensors = standard_checkpoint_tensors()
        tensors["model.diffusion_model.extra_fp16"] = torch.ones(1, dtype=torch.float16)
        write_checkpoint(self.path, tensors)

        with self.assertRaisesRegex(ValueError, "F16"):
            validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)

    def test_rejects_wrong_krea2_key_shape_signature(self):
        write_checkpoint(self.path, standard_checkpoint_tensors())

        with self.assertRaisesRegex(ValueError, "key/shape"):
            validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)

    def test_rejects_non_positive_scale(self):
        tensors = standard_checkpoint_tensors()
        tensors["model.diffusion_model.test_linear_0.weight_scale"] = torch.tensor(
            0.0, dtype=torch.float32
        )
        write_checkpoint(self.path, tensors)

        with self.assertRaisesRegex(ValueError, "有限正数"):
            validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)

    def test_rejects_int8_convrot(self):
        weight_key = "model.diffusion_model.blocks.0.attn.wq.weight"
        write_checkpoint(
            self.path,
            {
                weight_key: torch.zeros((2, 2), dtype=torch.int8),
                weight_key + "_scale": torch.tensor(1.0, dtype=torch.float32),
                quant_key(weight_key): quant_metadata(
                    format="int8_tensorwise", convrot=True, convrot_groupsize=256
                ),
            },
        )

        with self.assertRaisesRegex(ValueError, "INT8"):
            validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)

    def test_rejects_mixed_fp8_fp16(self):
        weight_key = "model.diffusion_model.blocks.0.attn.wq.weight"
        write_checkpoint(
            self.path,
            {
                weight_key: torch.zeros((2, 2), dtype=torch.float8_e4m3fn),
                weight_key + "_scale": torch.tensor(1.0, dtype=torch.float32),
                quant_key(weight_key): quant_metadata(),
                "model.diffusion_model.blocks.0.attn.wk.weight": torch.zeros(
                    (2, 2), dtype=torch.float16
                ),
            },
        )

        with self.assertRaisesRegex(ValueError, "F16"):
            validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)

    def test_rejects_unscaled_fp8(self):
        write_checkpoint(
            self.path,
            {
                "model.diffusion_model.blocks.0.attn.wq.weight": torch.zeros(
                    (2, 2), dtype=torch.float8_e4m3fn
                ),
            },
        )

        with self.assertRaisesRegex(ValueError, "scale"):
            validate_krea2_scaled_fp8(str(self.path), cache_path=self.cache_path)


if __name__ == "__main__":
    unittest.main()
