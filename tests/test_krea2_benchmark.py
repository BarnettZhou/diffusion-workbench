import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from demo import benchmark_all_krea2


class Krea2BenchmarkTests(unittest.TestCase):
    def test_discovers_every_safetensors_without_format_filtering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            for name in ("z-int8.safetensors", "a-fp8.safetensors", "ignore.txt"):
                (model_dir / name).touch()

            checkpoints = benchmark_all_krea2.discover_checkpoints(model_dir)

        self.assertEqual(
            [path.name for path in checkpoints],
            ["a-fp8.safetensors", "z-int8.safetensors"],
        )

    def test_runner_command_explicitly_allows_any_format_and_pins_path(self):
        args = argparse.Namespace(
            comfy_root=Path(r"C:\ComfyUI"),
            prompt="test",
            width=576,
            height=576,
            steps=8,
            seed=42,
            dry_run=False,
        )
        model_path = Path(r"E:\models\krea2\int8.safetensors")
        output_path = Path(r"E:\outputs\int8.png")

        command = benchmark_all_krea2.build_runner_command(args, model_path, output_path)

        self.assertIn("--benchmark-any-format", command)
        self.assertEqual(command[command.index("--expected-model-path") + 1], str(model_path.resolve()))
        self.assertEqual(command[command.index("--model-name") + 1], r"krea2\int8.safetensors")

    def test_parses_structured_runner_result(self):
        payload = {"load_seconds": 12.5, "generation_seconds": 6.25}
        output = "noise\nKREA2_BENCHMARK_RESULT=" + json.dumps(payload) + "\n"

        self.assertEqual(benchmark_all_krea2.parse_runner_result(output), payload)

    @patch("demo.benchmark_all_krea2.time.sleep")
    @patch("demo.benchmark_all_krea2.time.monotonic", side_effect=[0.0, 181.0, 181.1])
    @patch("demo.benchmark_all_krea2.subprocess.Popen")
    def test_timeout_terminates_only_started_process(self, popen, _monotonic, _sleep):
        process = MagicMock()
        process.pid = 1234
        process.poll.side_effect = [None, None]
        process.wait.return_value = -15
        popen.return_value = process

        with tempfile.TemporaryDirectory() as temp_dir:
            result = benchmark_all_krea2.run_checkpoint_process(
                ["python", "runner.py"],
                Path(temp_dir) / "model.log",
                timeout_seconds=180,
                min_free_ram_gib=0,
            )

        process.terminate.assert_called_once_with()

        child_env = popen.call_args.kwargs["env"]
        self.assertEqual(child_env["PYTHONUTF8"], "1")
        self.assertEqual(child_env["PYTHONIOENCODING"], "utf-8")
        process.kill.assert_not_called()
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["pid"], 1234)

    @patch("demo.benchmark_all_krea2.available_physical_memory_gib", side_effect=OSError("probe failed"))
    @patch("demo.benchmark_all_krea2.time.monotonic", side_effect=[0.0, 1.0])
    @patch("demo.benchmark_all_krea2.subprocess.Popen")
    def test_orchestrator_error_still_terminates_started_process(
        self, popen, _monotonic, _memory_probe
    ):
        process = MagicMock()
        process.poll.side_effect = [None, None]
        process.wait.return_value = -15
        popen.return_value = process

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(OSError, "probe failed"):
                benchmark_all_krea2.run_checkpoint_process(
                    ["python", "runner.py"],
                    Path(temp_dir) / "model.log",
                    timeout_seconds=180,
                    min_free_ram_gib=1,
                )

        process.terminate.assert_called_once_with()

    @patch("demo.benchmark_all_krea2.run_single_checkpoint")
    def test_cleanup_failure_aborts_batch(self, run_single):
        run_single.side_effect = benchmark_all_krea2.ProcessCleanupError("still running")

        with self.assertRaisesRegex(benchmark_all_krea2.ProcessCleanupError, "still running"):
            benchmark_all_krea2.run_all(
                argparse.Namespace(),
                [Path("one.safetensors"), Path("two.safetensors")],
                {},
                lambda _records: None,
            )

    def test_stale_format_cache_is_not_reported_as_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.safetensors"
            model_path.write_bytes(b"new")
            cache = {
                str(model_path.resolve()): {
                    "size": 999,
                    "mtime_ns": model_path.stat().st_mtime_ns,
                    "valid": True,
                    "format": "scaled_fp8_e4m3fn",
                    "dtypes": {"F8_E4M3": 256},
                }
            }

            record = benchmark_all_krea2.format_record(cache, model_path)

        self.assertIsNone(record["strict_fp8_valid"])
        self.assertEqual(record["detected_format"], "unknown")
        self.assertIn("过期", record["format_detection_error"])

    @patch("demo.benchmark_all_krea2.run_single_checkpoint")
    def test_run_all_continues_after_failure(self, run_single):
        run_single.side_effect = [
            {"model": "one", "status": "error"},
            {"model": "two", "status": "success"},
        ]
        args = argparse.Namespace()
        saved = []

        results = benchmark_all_krea2.run_all(
            args,
            [Path("one.safetensors"), Path("two.safetensors")],
            {},
            lambda records: saved.append(list(records)),
        )

        self.assertEqual([item["status"] for item in results], ["error", "success"])
        self.assertEqual(len(saved), 2)

    @patch("demo.benchmark_all_krea2.run_single_checkpoint")
    def test_run_all_records_unexpected_error_and_continues(self, run_single):
        run_single.side_effect = [
            RuntimeError("could not start"),
            {"model": "two", "status": "success"},
        ]
        saved = []

        results = benchmark_all_krea2.run_all(
            argparse.Namespace(),
            [Path("one.safetensors"), Path("two.safetensors")],
            {},
            lambda records: saved.append(list(records)),
        )

        self.assertEqual(results[0]["status"], "orchestrator_error")
        self.assertIn("could not start", results[0]["stop_reason"])
        self.assertEqual(results[1]["status"], "success")
        self.assertEqual(len(saved), 2)


if __name__ == "__main__":
    unittest.main()
