import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from demo import research_krea2_backends


def quant_metadata(format_name: str, **extra) -> torch.Tensor:
    payload = {"format": format_name, **extra}
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


class Krea2BackendResearchTests(unittest.TestCase):
    def test_inspection_estimates_bf16_expansion_and_quant_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.safetensors"
            save_file(
                {
                    "model.diffusion_model.layer.weight": torch.ones(
                        (4, 8), dtype=torch.float8_e4m3fn
                    ),
                    "model.diffusion_model.layer.weight_scale": torch.tensor(
                        0.25, dtype=torch.float32
                    ),
                    "model.diffusion_model.layer.comfy_quant": quant_metadata(
                        "float8_e4m3fn"
                    ),
                    "model.diffusion_model.norm": torch.ones(8, dtype=torch.bfloat16),
                },
                str(checkpoint),
            )

            report = research_krea2_backends.inspect_checkpoint(checkpoint)

        self.assertEqual(report["quant_formats"], {"float8_e4m3fn": 1})
        self.assertEqual(report["base_dtypes"], {"BF16": 1, "F8_E4M3": 1})
        self.assertEqual(report["bf16_expanded_bytes"], (4 * 8 + 8) * 2)
        self.assertTrue(report["native_fp8_candidate"])
        self.assertFalse(report["diffusers_direct_candidate"])

    def test_int8_convrot_is_not_a_native_fp8_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.safetensors"
            save_file(
                {
                    "model.diffusion_model.layer.weight": torch.ones(
                        (4, 8), dtype=torch.int8
                    ),
                    "model.diffusion_model.layer.weight_scale": torch.ones(
                        (4, 1), dtype=torch.float32
                    ),
                    "model.diffusion_model.layer.comfy_quant": quant_metadata(
                        "int8_tensorwise", convrot=True, convrot_groupsize=256
                    ),
                },
                str(checkpoint),
            )

            report = research_krea2_backends.inspect_checkpoint(checkpoint)

        self.assertEqual(report["quant_formats"], {"int8_tensorwise": 1})
        self.assertTrue(report["requires_convrot_kernel"])
        self.assertFalse(report["native_fp8_candidate"])

    def test_cache_match_requires_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.safetensors"
            checkpoint.write_bytes(b"abc")
            stat = checkpoint.stat()
            cached = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

            self.assertTrue(research_krea2_backends.cache_matches(checkpoint, cached))
            checkpoint.write_bytes(b"changed")
            self.assertFalse(research_krea2_backends.cache_matches(checkpoint, cached))


if __name__ == "__main__":
    unittest.main()
