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
        sim.set_axis_velocities(1, 2, 3)
        sim.set_extrusion_velocity(100)  # 1 mm/s with steps_per_mm=100
        sim.update(1.0)
        # Visualizer should track axis positions
        self.assertEqual(list(sim.visualizer.extruder.position), [1, 2, 3])
        # A filament segment should have been added
        self.assertEqual(len(sim.visualizer.filament), 1)


if __name__ == "__main__":
    unittest.main()
