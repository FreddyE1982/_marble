import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location(
    "printer_visualization", pathlib.Path("3d_printer_sim/visualization.py")
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys_modules = __import__("sys").modules
sys_modules[spec.name] = module
spec.loader.exec_module(module)
PrinterVisualizer = module.PrinterVisualizer


class TestVisualization(unittest.TestCase):
    def test_scene_components(self):
        viz = PrinterVisualizer()
        # bed and extruder should be present in the scene
        self.assertIn(viz.bed, viz.scene.children)
        self.assertIn(viz.extruder, viz.scene.children)

    def test_position_update(self):
        viz = PrinterVisualizer()
        viz.update_extruder_position(10, 20, 30)
        self.assertEqual(list(viz.extruder.position), [10, 20, 30])

    def test_filament_segment_added(self):
        viz = PrinterVisualizer()
        before = len(viz.scene.children)
        viz.add_filament_segment(0, 0, 0)
        self.assertGreater(len(viz.scene.children), before)


if __name__ == "__main__":
    unittest.main()

