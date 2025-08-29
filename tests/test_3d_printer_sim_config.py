import importlib.util
import pathlib
import unittest

import sys

spec = importlib.util.spec_from_file_location(
    "printer_config", pathlib.Path("3d_printer_sim/config.py")
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
load_config = module.load_config


class TestPrinterConfig(unittest.TestCase):
    def test_load_config(self) -> None:
        cfg = load_config("3d_printer_sim/config.yaml")
        self.assertEqual(len(cfg.extruders), 2)
        self.assertEqual(cfg.build_volume.z, 250)
        self.assertIn("PETG", cfg.filament_types)
        self.assertEqual(cfg.extruders[1].filament, "PETG")
        self.assertAlmostEqual(cfg.heater_targets["hotend"], 200)


if __name__ == "__main__":
    unittest.main()
