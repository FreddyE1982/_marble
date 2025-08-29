import importlib.util
import math
import pathlib
import sys
import unittest


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stepper_mod = load_module("stepper", "3d_printer_sim/stepper.py")
extruder_mod = load_module("extruder", "3d_printer_sim/extruder.py")

StepperMotor = stepper_mod.StepperMotor
Extruder = extruder_mod.Extruder


class TestExtruder(unittest.TestCase):
    def test_extrusion_volume(self) -> None:
        motor = StepperMotor(max_acceleration=1000, max_jerk=1000)
        extruder = Extruder(motor, steps_per_mm=100, filament_diameter=1.75)
        extruder.set_target_velocity(100)  # steps per second
        extruder.update(1.0)
        self.assertAlmostEqual(extruder.extruded_length, 1.0)
        expected_volume = math.pi * (1.75 / 2) ** 2 * 1.0
        self.assertAlmostEqual(extruder.deposited_volume, expected_volume)


if __name__ == "__main__":
    unittest.main()

