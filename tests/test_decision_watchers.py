import sys
import types
import unittest

from marble.decision_controller import (
    DecisionController,
    L1_PENALTY,
    get_plugin_cost,
)
from marble.reporter import REPORTER, clear_report_group


class TestDecisionWatchers(unittest.TestCase):
    def setUp(self) -> None:
        clear_report_group("metrics")

    def test_watchers_collect_values(self) -> None:
        REPORTER.item["latency", "metrics"] = 5.0
        dc = DecisionController(
            watch_metrics=["metrics/latency"],
            watch_variables=["marble.decision_controller.L1_PENALTY"],
        )
        values = dc._gather_watchables()
        print("watcher collected:", values)
        self.assertEqual(values["metrics/latency"], 5.0)
        self.assertEqual(
            values["marble.decision_controller.L1_PENALTY"], L1_PENALTY
        )

    def test_plugin_cost_autodetect(self) -> None:
        mod = types.ModuleType("marble.plugins.fakeplugin")
        mod.PLUGIN_COST = 7.5
        sys.modules["marble.plugins.fakeplugin"] = mod
        cost = get_plugin_cost("fakeplugin")
        print("auto detected cost:", cost)
        self.assertEqual(cost, 7.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

