import importlib.util
import math
import pathlib
import tempfile
import unittest

import sys

# Load modules using importlib to keep tests self-contained
cfg_spec = importlib.util.spec_from_file_location(
    "printer_config", pathlib.Path("3d_printer_sim/config.py")
)
cfg_module = importlib.util.module_from_spec(cfg_spec)
assert cfg_spec.loader is not None
sys.modules[cfg_spec.name] = cfg_module
cfg_spec.loader.exec_module(cfg_module)
load_config = cfg_module.load_config

sim_spec = importlib.util.spec_from_file_location(
    "printer_sim", pathlib.Path("3d_printer_sim/simulation.py")
)
sim_module = importlib.util.module_from_spec(sim_spec)
assert sim_spec.loader is not None
sys.modules[sim_spec.name] = sim_module
sim_spec.loader.exec_module(sim_module)
PrinterSimulation = sim_module.PrinterSimulation


class TestBedTiltConfig(unittest.TestCase):
    def _base_yaml(self) -> str:
        return (
            "build_volume:\n  x: 1\n  y: 1\n  z: 1\n"
            "bed_size:\n  x: 100\n  y: 100\n"
            "max_print_dimensions:\n  x: 1\n  y: 1\n  z: 1\n"
            "extruders:\n  - id: 0\n    type: direct\n    hotend: e3d_v6\n    filament: PLA\n"
            "filament_types:\n  PLA:\n    hotend_temp:\n      - 190\n      - 220\n    bed_temp:\n      - 0\n      - 60\n"
        )

    def test_explicit_angles(self) -> None:
        text = self._base_yaml() + "bed_tilt:\n  x: 1\n  y: 2\n"
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            cfg = load_config(tmp.name)
        self.assertAlmostEqual(cfg.bed_tilt.x, 1)
        self.assertAlmostEqual(cfg.bed_tilt.y, 2)

    def test_screws_compute_tilt(self) -> None:
        text = (
            self._base_yaml()
            + "bed_screws:\n    front_left: 1\n    front_right: 1\n    back_left: 0\n    back_right: 0\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            cfg = load_config(tmp.name)
        expected = math.degrees(math.atan2(1, 100))
        self.assertAlmostEqual(cfg.bed_tilt.x, expected)
        self.assertAlmostEqual(cfg.bed_tilt.y, 0.0)


class TestBedTiltSimulation(unittest.TestCase):
    def test_rotation_and_gravity(self) -> None:
        sim = PrinterSimulation(bed_tilt_x=90, bed_tilt_y=0)
        sim.x_motor.position = 0
        sim.y_motor.position = 0
        sim.z_motor.position = 10
        sim.update(0.1)
        x, y, z = sim.visualizer.extruder.position
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, -10.0, places=5)
        self.assertAlmostEqual(z, 0.0, places=5)
        gx, gy, gz = sim.gravity
        self.assertAlmostEqual(gx, 0.0, places=5)
        self.assertAlmostEqual(gy, -9.81, places=2)
        self.assertAlmostEqual(gz, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

