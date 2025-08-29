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


thermal_mod = load_module("thermal", "3d_printer_sim/thermal.py")
sensors_mod = load_module("sensors", "3d_printer_sim/sensors.py")
micro_mod = load_module("microcontroller", "3d_printer_sim/microcontroller.py")

HeatedBed = thermal_mod.HeatedBed
TemperatureSensor = sensors_mod.TemperatureSensor
Microcontroller = micro_mod.Microcontroller


class TestThermal(unittest.TestCase):
    def test_heating_and_cooling(self) -> None:
        mc = Microcontroller()
        sensor = TemperatureSensor(mc, pin=1, value=25.0)
        bed = HeatedBed(sensor, heating_rate=100, cooling_rate=50)
        bed.set_target_temperature(200)
        bed.update(1.0)
        self.assertAlmostEqual(sensor.read_temperature(), 125)
        self.assertAlmostEqual(mc.read_analog(1), 125)
        bed.update(1.0)
        self.assertAlmostEqual(sensor.read_temperature(), 200)
        bed.set_target_temperature(50)
        bed.update(1.0)
        self.assertAlmostEqual(sensor.read_temperature(), 150)


if __name__ == "__main__":
    unittest.main()

