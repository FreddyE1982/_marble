import importlib.util
import pathlib
import unittest


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    __import__("sys").modules[name] = module
    spec.loader.exec_module(module)
    return module


sim_mod = load_module("sim", "3d_printer_sim/simulation.py")
mc_mod = load_module("mc", "3d_printer_sim/microcontroller.py")
sensors_mod = load_module("sensors", "3d_printer_sim/sensors.py")

PrinterSimulation = sim_mod.PrinterSimulation
Microcontroller = mc_mod.Microcontroller
TemperatureSensor = sensors_mod.TemperatureSensor


class TestAdhesion(unittest.TestCase):
    def test_bed_temperature_influence(self) -> None:
        sim = PrinterSimulation()
        sim.z_motor.position = sim.layer_height
        sim.set_extrusion_velocity(100)

        sim.bed_temperature = 20
        sim.update(0.1)
        low_adh = sim.adhesion
        count_low = len(sim.visualizer.filament)

        sim.extruder.reset()
        sim._last_extruded = 0.0
        sim.bed_temperature = sim.optimal_bed_temp
        sim.update(0.1)

        self.assertLess(low_adh, 0.5)
        self.assertEqual(count_low, 0)
        self.assertGreater(sim.adhesion, low_adh)
        self.assertEqual(len(sim.visualizer.filament), 1)

    def test_extruder_temperature_influence(self) -> None:
        sim = PrinterSimulation()
        mc = Microcontroller()
        sensor = TemperatureSensor(mc, pin=0, value=150)
        sim.extruder.temperature_sensor = sensor

        sim.z_motor.position = sim.layer_height * 2
        sim.set_extrusion_velocity(100)

        sim.update(0.1)
        low = sim.adhesion
        count_low = len(sim.visualizer.filament)

        sim.extruder.reset()
        sim._last_extruded = 0.0
        sensor.set_temperature(240)
        sim.update(0.1)

        self.assertLess(low, 0.5)
        self.assertEqual(count_low, 0)
        self.assertGreater(sim.adhesion, low)
        self.assertEqual(len(sim.visualizer.filament), 1)


if __name__ == "__main__":
    unittest.main()

