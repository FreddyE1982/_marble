import importlib.util
import pathlib
import unittest
import sys

spec = importlib.util.spec_from_file_location(
    "microcontroller", pathlib.Path("3d_printer_sim/microcontroller.py")
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Microcontroller = module.Microcontroller


class TestMicrocontroller(unittest.TestCase):
    def test_digital_io(self) -> None:
        mc = Microcontroller()
        mc.set_digital(13, 1)
        val13 = mc.read_digital(13)
        val12 = mc.read_digital(12)
        print("digital pins:", val13, val12)
        self.assertEqual(val13, 1)
        self.assertEqual(val12, 0)

    def test_analog_io(self) -> None:
        mc = Microcontroller()
        mc.set_analog(0, 3.3)
        val0 = mc.read_analog(0)
        val1 = mc.read_analog(1)
        print("analog pins:", val0, val1)
        self.assertAlmostEqual(val0, 3.3)
        self.assertEqual(val1, 0.0)


if __name__ == "__main__":
    unittest.main()
