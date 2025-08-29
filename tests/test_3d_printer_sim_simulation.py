import importlib.util
import pathlib
import unittest

# Load simulation module in a standalone manner
spec = importlib.util.spec_from_file_location(
    "simulation", pathlib.Path("3d_printer_sim/simulation.py")
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys_modules = __import__("sys").modules
sys_modules[spec.name] = module
spec.loader.exec_module(module)
PrinterSimulation = module.PrinterSimulation


class TestSimulation(unittest.TestCase):
    def test_visual_sync_and_filament(self):
        sim = PrinterSimulation()
        # Keep nozzle at first-layer height to allow adhesion
        sim.z_motor.position = 0.2
        sim.set_axis_velocities(1, 2, 0)
        sim.set_extrusion_velocity(100)  # 1 mm/s with steps_per_mm=100
        sim.update(1.0)
        # Visualizer should track axis positions
        print("extruder pos and filament:", list(sim.visualizer.extruder.position), len(sim.visualizer.filament))
        self.assertEqual(list(sim.visualizer.extruder.position), [1, 2, 0.2])
        # A filament segment should have been added due to sufficient adhesion
        self.assertEqual(len(sim.visualizer.filament), 1)

    def test_axis_physics(self) -> None:
        """Axes should respect acceleration and jerk limits."""

        sim = PrinterSimulation()
        # Tight limits to make physics observable
        sim.x_motor.max_acceleration = 10
        sim.x_motor.max_jerk = 5
        sim.set_axis_velocities(100, 0, 0)

        # First second: jerk limits acceleration
        sim.update(1.0)
        print("step1 accel/vel/pos:", sim.x_motor.acceleration, sim.x_motor.velocity, sim.x_motor.position)
        self.assertAlmostEqual(sim.x_motor.acceleration, 5)
        self.assertAlmostEqual(sim.x_motor.velocity, 5)
        self.assertAlmostEqual(sim.x_motor.position, 5)

        # Second second: acceleration reaches limit
        sim.update(1.0)
        print("step2 accel/vel/pos:", sim.x_motor.acceleration, sim.x_motor.velocity, sim.x_motor.position)
        self.assertAlmostEqual(sim.x_motor.acceleration, 10)
        self.assertAlmostEqual(sim.x_motor.velocity, 15)
        self.assertAlmostEqual(sim.x_motor.position, 20)


if __name__ == "__main__":
    unittest.main()
