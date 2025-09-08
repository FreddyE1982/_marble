import importlib
import unittest

from marble import plugin_cost_profiler as cp


class PluginCostProfilerTests(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(cp)

    def test_record_and_get_cost(self) -> None:
        cp.record("test", 10.0)
        self.assertEqual(cp.get_cost("test"), 10.0)
        cp.record("test", 6.0)
        self.assertEqual(cp.get_cost("test"), 8.0)

    def test_default_value(self) -> None:
        self.assertEqual(cp.get_cost("missing", 1.23), 1.23)


if __name__ == "__main__":
    unittest.main()
