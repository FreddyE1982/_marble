import importlib.util
import pathlib
import tempfile
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
parse_simple_yaml = module.parse_simple_yaml


class TestPrinterConfig(unittest.TestCase):
    def test_load_config(self) -> None:
        cfg = load_config("3d_printer_sim/config.yaml")
        print("extruders:", len(cfg.extruders), "volume_z:", cfg.build_volume.z)
        self.assertEqual(len(cfg.extruders), 2)
        self.assertEqual(cfg.build_volume.z, 250)
        self.assertIn("PETG", cfg.filament_types)
        self.assertEqual(cfg.extruders[1].filament, "PETG")
        self.assertAlmostEqual(cfg.heater_targets["hotend"], 200)
        self.assertAlmostEqual(cfg.ambient_temperature, 25)
        self.assertAlmostEqual(cfg.fan_speed, 0.0)

    def test_missing_section_raises(self) -> None:
        text = (
            "build_volume:\n  x: 1\n  y: 1\n  z: 1\n"
            "bed_size:\n  x: 1\n  y: 1\n"
            "max_print_dimensions:\n  x: 1\n  y: 1\n  z: 1\n"
            "filament_types: {}\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            with self.assertRaises(ValueError):
                load_config(tmp.name)
        print("missing section raised ValueError as expected")

    def test_empty_extruders_list(self) -> None:
        text = (
            "build_volume:\n  x: 1\n  y: 1\n  z: 1\n"
            "bed_size:\n  x: 1\n  y: 1\n"
            "max_print_dimensions:\n  x: 1\n  y: 1\n  z: 1\n"
            "extruders: []\n"
            "filament_types: {}\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            with self.assertRaises(ValueError):
                load_config(tmp.name)
        print("empty extruders raised ValueError")

    def test_negative_axis_value(self) -> None:
        text = (
            "build_volume:\n  x: -1\n  y: 1\n  z: 1\n"
            "bed_size:\n  x: 1\n  y: 1\n"
            "max_print_dimensions:\n  x: 1\n  y: 1\n  z: 1\n"
            "extruders:\n  - id: 0\n    type: direct\n    hotend: e3d_v6\n    filament: PLA\n"
            "filament_types:\n  PLA:\n    hotend_temp:\n      - 190\n      - 220\n    bed_temp:\n      - 0\n      - 60\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            with self.assertRaises(ValueError):
                load_config(tmp.name)
        print("negative axis raised ValueError")

    def test_parse_simple_yaml_basic(self) -> None:
        data = parse_simple_yaml("a: 1\nb:\n  - 2\n  - 3\n")
        print("parsed:", data)
        self.assertEqual(data, {"a": 1, "b": [2, 3]})

    def test_example_configs_parse(self) -> None:
        examples = pathlib.Path("3d_printer_sim/examples")
        for path in examples.glob("*.yaml"):
            cfg = load_config(str(path))
            print(path.name, "extruders", len(cfg.extruders))
            self.assertGreater(len(cfg.extruders), 0)


if __name__ == "__main__":
    unittest.main()
