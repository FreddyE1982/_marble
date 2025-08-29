import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location(
    "simulation", pathlib.Path("3d_printer_sim/simulation.py")
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys_modules = __import__("sys").modules
sys_modules[spec.name] = module
spec.loader.exec_module(module)
PrinterSimulation = module.PrinterSimulation
FilamentSegment = module.FilamentSegment


class TestSupportsAndBrims(unittest.TestCase):
    def test_brim_generation(self) -> None:
        sim = PrinterSimulation()
        sim.generate_brim(radius=2, segments=4)
        print("brim segments:", len(sim.segments))
        self.assertEqual(len(sim.segments), 4)
        for seg in sim.segments:
            print("brim seg adhesion/removal:", seg.adhesion_strength, seg.removal_force)
            self.assertEqual(seg.kind, "brim")
            self.assertGreater(seg.adhesion_strength, 1.0)
            self.assertLess(seg.removal_force, 1.0)

    def test_support_creation(self) -> None:
        sim = PrinterSimulation()
        sim.enable_auto_support(True)
        sim.z_motor.position = 10.3
        sim.set_extrusion_velocity(100)
        sim.update(1.0)
        kinds = [seg.kind for seg in sim.segments]
        print("segment kinds:", kinds)
        self.assertIn("support", kinds)
        self.assertIn("normal", kinds)
        support_segments = [seg for seg in sim.segments if seg.kind == "support"]
        for seg in support_segments:
            print("support adhesion/removal:", seg.adhesion_strength, seg.removal_force)
        self.assertTrue(all(abs(seg.adhesion_strength - 0.5) < 1e-6 for seg in support_segments))
        self.assertTrue(all(abs(seg.removal_force - 0.5) < 1e-6 for seg in support_segments))


if __name__ == "__main__":
    unittest.main()
