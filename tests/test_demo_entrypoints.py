import unittest
from pathlib import Path
from unittest.mock import patch

from demo import demo_krea2, demo_zit
import comfyui_krea2_runner
from krea2_config import KREA2_MODEL_PATH
from demo.demo_txt2img import (
    MODELS,
    load_pipeline,
    run_demo,
    simple_sigmas,
)


class DemoEntrypointTests(unittest.TestCase):
    def test_krea2_uses_requested_baseline_checkpoint(self):
        self.assertEqual(
            MODELS["krea2"]["transformer"],
            "E:\\Documents\\ComfyUI\\models\\diffusion_models\\krea2\\"
            "[GPT逼真版]krea2GPTGrandPUSSYTruth_krea2GPT.safetensors",
        )

    def test_krea2_uses_comfyui_quantized_backend(self):
        self.assertEqual(MODELS["krea2"]["backend"], "comfyui")

    @patch("demo.demo_txt2img.load_comfy_state_dict")
    def test_legacy_loader_rejects_krea2_before_reading_weights(self, load_state_dict):
        with self.assertRaisesRegex(ValueError, "ComfyUI"):
            load_pipeline("krea2")

        load_state_dict.assert_not_called()

    def test_krea2_defaults_to_576_square_eight_steps(self):
        args = demo_krea2.build_parser().parse_args(["--prompt", "test"])

        self.assertEqual((args.width, args.height), (576, 576))
        self.assertEqual(args.steps, 8)

    def test_zit_defaults_to_576_square_nine_steps(self):
        args = demo_zit.build_parser().parse_args(["--prompt", "test"])

        self.assertEqual((args.width, args.height), (576, 576))
        self.assertEqual(args.steps, 9)

    def test_simple_sigmas_match_comfyui_eight_step_schedule(self):
        self.assertEqual(
            simple_sigmas(8),
            [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        )

    def test_simple_sigmas_reject_non_positive_steps(self):
        with self.assertRaisesRegex(ValueError, "steps"):
            simple_sigmas(0)

    @patch("demo.demo_txt2img.load_pipeline")
    def test_invalid_steps_fail_before_loading_model(self, load_pipeline):
        with self.assertRaisesRegex(ValueError, "steps"):
            run_demo("krea2", "test", "", 576, 576, 0, 42)

        load_pipeline.assert_not_called()

    def test_krea2_launcher_uses_bundled_comfyui_python(self):
        args = demo_krea2.build_parser().parse_args(["--prompt", "test"])

        command = demo_krea2.build_runner_command(args)

        self.assertEqual(Path(command[0]), demo_krea2.DEFAULT_COMFY_ROOT.parent / "python" / "python.exe")
        self.assertIn(str(demo_krea2.RUNNER_PATH), command)
        self.assertNotIn("demo_txt2img.py", command)
        self.assertNotIn("--negative-prompt", command)
        self.assertNotIn("--benchmark-any-format", command)

    def test_runner_loads_the_same_checkpoint_that_was_validated(self):
        self.assertEqual(
            comfyui_krea2_runner.KREA2_MODEL_PATH,
            KREA2_MODEL_PATH,
        )

    def test_dry_run_rejects_invalid_generation_dimensions(self):
        args = comfyui_krea2_runner.build_parser().parse_args(
            [
                "--comfy-root",
                ".",
                "--prompt",
                "test",
                "--out",
                "out.png",
                "--width",
                "575",
                "--dry-run",
            ]
        )

        with self.assertRaisesRegex(ValueError, "16"):
            comfyui_krea2_runner.validate_generation_args(args)


if __name__ == "__main__":
    unittest.main()
