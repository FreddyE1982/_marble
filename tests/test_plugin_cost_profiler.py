import importlib
import math
import unittest

from marble import plugin_cost_profiler as cp


class PluginCostProfilerTests(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(cp)
        cp.enable()

    def test_record_and_get_cost(self) -> None:
        cp.record("test", 10.0)
        cost1 = cp.get_cost("test")
        print("first cost:", cost1)
        self.assertEqual(cost1, 10.0)
        cp.record("test", 6.0)
        cost2 = cp.get_cost("test")
        print("ema cost:", cost2)
        self.assertEqual(cost2, 8.0)

    def test_default_value(self) -> None:
        default = cp.get_cost("missing", 1.23)
        print("default cost:", default)
        self.assertEqual(default, 1.23)

    def test_missing_returns_nan(self) -> None:
        val = cp.get_cost("unknown")
        print("missing cost:", val)
        self.assertTrue(math.isnan(val))


if __name__ == "__main__":
    unittest.main()
