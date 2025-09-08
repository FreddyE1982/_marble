import importlib
import sys
import types
import unittest

from marble.decision_controller import get_plugin_cost
from marble import plugin_cost_profiler as cp


class GetPluginCostProfilerTests(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(cp)
        sys.modules.pop("marble.plugins.fakeplugin", None)

    def test_profiler_priority_and_zero_init(self) -> None:
        mod = types.ModuleType("marble.plugins.fakeplugin")
        mod.PLUGIN_COST = 7.5
        sys.modules["marble.plugins.fakeplugin"] = mod

        cp.record("fakeplugin", 2.5)
        cost = get_plugin_cost("fakeplugin")
        print("cost from profiler:", cost)
        self.assertEqual(cost, 2.5)

        importlib.reload(cp)
        sys.modules["marble.plugins.fakeplugin"] = mod
        cost = get_plugin_cost("fakeplugin")
        stored = cp.get_cost("fakeplugin")
        print("fallback cost and stored cost:", cost, stored)
        self.assertEqual(cost, 7.5)
        self.assertEqual(stored, 0.0)


if __name__ == "__main__":
    unittest.main()
