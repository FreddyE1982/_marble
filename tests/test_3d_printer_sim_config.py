import importlib.util
import pathlib
import unittest

import sys

from marble.reporter import report, clear_report_group, report_group

spec = importlib.util.spec_from_file_location(
    "printer_config", pathlib.Path("3d_printer_sim/config.py")
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
load_config = module.load_config


class TestPrinterConfig(unittest.TestCase):
    def setUp(self) -> None:
        clear_report_group("3d_printer_sim")

    def tearDown(self) -> None:
        clear_report_group("3d_printer_sim")

    def test_load_config(self) -> None:
        cfg = load_config("3d_printer_sim/config.yaml")
        report("3d_printer_sim", "extruder_count", len(cfg.extruders))
        self.assertEqual(len(cfg.extruders), 2)
        self.assertEqual(cfg.build_volume.z, 250)
        logged = report_group("3d_printer_sim")
        self.assertIn("extruder_count", logged)


if __name__ == "__main__":
    unittest.main()
