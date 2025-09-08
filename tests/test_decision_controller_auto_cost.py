import importlib
import unittest

from marble import plugin_cost_profiler as cp
import marble.decision_controller as dc


class DecisionControllerAutoCostTests(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(cp)
        importlib.reload(dc)
        dc.AUTO_COST_PROFILE = True
        dc.BUDGET_LIMIT = 0.5

    def test_profiled_cost_used(self) -> None:
        dc.LAST_STATE_CHANGE.clear()
        x_t = {"A": "on"}
        h_t = {"A": {}}
        history: list[dict] = []
        # Instantiating the controller enables the profiler when the flag is set
        dc.DecisionController(top_k=1)
        sel1 = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("first selection", sel1)
        self.assertEqual(sel1, {"A": "on"})
        # Simulate plugin execution and its measured cost
        cp.record("A", 1.0)
        history.append({"A": {"action": "on", "cost": 1.0}})
        sel2 = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("second selection", sel2)
        self.assertEqual(sel2, {})


if __name__ == "__main__":
    unittest.main()

