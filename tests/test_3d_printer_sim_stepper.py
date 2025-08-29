import importlib.util
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
micro_mod = load_module("microcontroller", "3d_printer_sim/microcontroller.py")

StepperMotor = stepper_mod.StepperMotor
Microcontroller = micro_mod.Microcontroller


class TestStepperMotor(unittest.TestCase):
    def test_accel_and_jerk_limits(self) -> None:
        motor = StepperMotor(max_acceleration=10, max_jerk=5)
        motor.set_target_velocity(100)
        motor.update(1.0)
        self.assertAlmostEqual(motor.acceleration, 5)
        self.assertAlmostEqual(motor.velocity, 5)
        self.assertAlmostEqual(motor.position, 5)
        motor.update(1.0)
        self.assertAlmostEqual(motor.acceleration, 10)
        self.assertAlmostEqual(motor.velocity, 15)
        self.assertAlmostEqual(motor.position, 20)

    def test_jerk_limit_progression(self) -> None:
        motor = StepperMotor(max_acceleration=50, max_jerk=2)
        motor.set_target_velocity(100)
        motor.update(1.0)
        self.assertAlmostEqual(motor.acceleration, 2)
        motor.update(1.0)
        self.assertAlmostEqual(motor.acceleration, 4)

    def test_pin_mapping(self) -> None:
        mcu = Microcontroller()
        motor = StepperMotor(max_acceleration=10, max_jerk=5)
        mcu.map_pin(1, motor)
        self.assertIs(mcu.get_mapped_component(1), motor)


if __name__ == "__main__":
    unittest.main()

