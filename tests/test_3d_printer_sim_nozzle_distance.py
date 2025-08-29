
import importlib.util
import pathlib
import unittest
import sys

spec = importlib.util.spec_from_file_location(
    "printer_sim", pathlib.Path("3d_printer_sim/simulation.py")
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
PrinterSimulation = module.PrinterSimulation


class TestNozzleDistance(unittest.TestCase):
    def test_width_and_adhesion(self) -> None:
        sim = PrinterSimulation()
        sim.z_motor.position = 0.2
        sim.set_extrusion_velocity(100)
        sim.update(0.1)
        self.assertAlmostEqual(sim.nozzle_height, 0.2)
        self.assertAlmostEqual(sim.extrusion_width, sim.nozzle_diameter)
        self.assertAlmostEqual(sim.adhesion, 1.0)
        self.assertEqual(len(sim.visualizer.filament), 1)

    def test_collision_and_detachment(self) -> None:
        sim = PrinterSimulation()
        sim.z_motor.position = -0.05
        sim.set_axis_velocities(2, 0, 0)
        sim.set_extrusion_velocity(100)
        sim.update(0.1)
        self.assertTrue(sim.collision)
        self.assertGreater(sim.surface_damage, 0)
        self.assertTrue(sim.part_detached)
        self.assertEqual(len(sim.visualizer.filament), 1)
        self.assertFalse(sim.segments[0].supported)

    def test_high_clearance_no_adhesion(self) -> None:
        sim = PrinterSimulation()
        sim.z_motor.position = 0.5
        sim.set_extrusion_velocity(100)
        sim.update(0.1)
        self.assertEqual(sim.adhesion, 0.0)
        self.assertEqual(len(sim.visualizer.filament), 1)
        self.assertFalse(sim.segments[0].supported)


if __name__ == "__main__":
    unittest.main()

