import importlib.util
import pathlib
import sys
import unittest

spec_mc = importlib.util.spec_from_file_location(
    "microcontroller", pathlib.Path("3d_printer_sim/microcontroller.py")
)
module_mc = importlib.util.module_from_spec(spec_mc)
assert spec_mc.loader is not None
sys.modules[spec_mc.name] = module_mc
spec_mc.loader.exec_module(module_mc)
Microcontroller = module_mc.Microcontroller

spec_sensors = importlib.util.spec_from_file_location(
    "sensors", pathlib.Path("3d_printer_sim/sensors.py")
)
module_sensors = importlib.util.module_from_spec(spec_sensors)
assert spec_sensors.loader is not None
sys.modules[spec_sensors.name] = module_sensors
spec_sensors.loader.exec_module(module_sensors)
TemperatureSensor = module_sensors.TemperatureSensor


class TestSensors(unittest.TestCase):
    def test_temperature_sensor_updates_pin(self) -> None:
        mc = Microcontroller()
        sensor = TemperatureSensor(mc=mc, pin=0, value=25.0)
        self.assertAlmostEqual(mc.read_analog(0), 25.0)
        sensor.set_temperature(200.0)
        self.assertAlmostEqual(mc.read_analog(0), 200.0)


if __name__ == "__main__":
    unittest.main()
