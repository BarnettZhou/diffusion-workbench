import unittest

from diffusion_workbench.cli import PACKAGED_CONFIG, build_parser


class WorkbenchCliTests(unittest.TestCase):
    def test_packaged_default_config_exists(self):
        self.assertTrue(PACKAGED_CONFIG.is_file())
        self.assertEqual(build_parser().prog, "diffusion-workbench")


if __name__ == "__main__":
    unittest.main()
