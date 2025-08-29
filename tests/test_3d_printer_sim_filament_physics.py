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
PrinterSimulation = sim_mod.PrinterSimulation


class TestFilamentPhysics(unittest.TestCase):
    def test_cooling_with_fan(self) -> None:
        sim_no_fan = PrinterSimulation(ambient_temperature=25, fan_speed=0)
        sim_no_fan.z_motor.position = sim_no_fan.layer_height
        sim_no_fan.set_extrusion_velocity(100)
        sim_no_fan.update(0.1)
        sim_no_fan.update(1.0)
        temp_no_fan = sim_no_fan.segments[0].temperature

        sim_fan = PrinterSimulation(ambient_temperature=25, fan_speed=1)
        sim_fan.z_motor.position = sim_fan.layer_height
        sim_fan.set_extrusion_velocity(100)
        sim_fan.update(0.1)
        sim_fan.update(1.0)
        temp_fan = sim_fan.segments[0].temperature

        self.assertLess(temp_fan, temp_no_fan)

    def test_shrinkage_after_cooling(self) -> None:
        sim = PrinterSimulation()
        sim.z_motor.position = sim.layer_height
        sim.set_extrusion_velocity(100)
        sim.update(0.1)
        initial_radius = sim.segments[0].radius
        for _ in range(20):
            sim.update(1.0)
        self.assertTrue(sim.segments[0].solid)
        self.assertLess(sim.segments[0].radius, initial_radius)

    def test_squish_with_bed_tilt(self) -> None:
        sim_flat = PrinterSimulation()
        sim_flat.z_motor.position = sim_flat.layer_height - 0.05
        sim_flat.set_extrusion_velocity(100)
        sim_flat.update(0.1)
        width_flat = sim_flat.extrusion_width

        sim_tilt = PrinterSimulation(bed_tilt_x=10)
        sim_tilt.z_motor.position = sim_tilt.layer_height - 0.05
        sim_tilt.set_extrusion_velocity(100)
        sim_tilt.update(0.1)
        width_tilt = sim_tilt.extrusion_width

        self.assertGreater(width_tilt, width_flat)


if __name__ == "__main__":
    unittest.main()

