import logging
import tempfile
import unittest
from pathlib import Path

from diffusion_workbench_core.logging_config import (
    CoreLogManager,
    worker_output_level,
)


class CoreLoggingTests(unittest.TestCase):
    def test_writes_expected_levels_and_utf8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = CoreLogManager(root / "jobs.sqlite3")
            manager.logger.info("任务开始")
            manager.logger.warning("需要注意")
            manager.logger.error("生成失败")
            manager.close()

            lines = manager.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertRegex(
                lines[0],
                r"^\[INFO\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} 任务开始$",
            )
            self.assertRegex(lines[1], r"^\[WARN\].*需要注意$")
            self.assertRegex(lines[2], r"^\[ERROR\].*生成失败$")

            manager.path.rename(root / "renamed.log")

    def test_worker_output_level(self):
        cases = {
            "ordinary output": logging.INFO,
            "INFO: loading": logging.INFO,
            "WARNING: low memory": logging.WARNING,
            "[WARN] fallback": logging.WARNING,
            "ERROR: failed": logging.ERROR,
            "[ERROR] failed": logging.ERROR,
            "Traceback (most recent call last):": logging.ERROR,
        }
        for line, expected in cases.items():
            with self.subTest(line=line):
                self.assertEqual(worker_output_level(line), expected)


if __name__ == "__main__":
    unittest.main()
