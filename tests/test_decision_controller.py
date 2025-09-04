import unittest
import time
import marble.decision_controller as dc


class TestDecisionController(unittest.TestCase):
    def test_incompatibility_and_capacity(self):
        dc.BUDGET_LIMIT = 10.0
        h_t = {"A": {"cost": 2}, "B": {"cost": 1}, "C": {"cost": 4}}
        x_t = {"A": "on", "B": "on", "C": "on"}
        history = [{"B": "on"}]
        selected = dc.decide_actions(h_t, x_t, history)
        print("selected after constraints:", selected)
        self.assertEqual(selected, {"A": "on"})

    def test_budget_limit(self):
        dc.BUDGET_LIMIT = 3.0
        h_t = {"A": {"cost": 2}, "B": {"cost": 1}, "C": {"cost": 4}}
        x_t = {"A": "on", "B": "on", "C": "on"}
        history = []
        selected = dc.decide_actions(h_t, x_t, history)
        print("selected under budget:", selected)
        self.assertEqual(selected, {"B": "on", "A": "on"})

    def test_per_plugin_running_cost(self):
        dc.BUDGET_LIMIT = 5.0
        h_t = {"A": {"cost": 3}, "B": {"cost": 3}}
        x_t = {"A": "on", "B": "on"}
        history = [{"A": "on"}]
        selected = dc.decide_actions(h_t, x_t, history)
        print("selected with per-plugin budget:", selected)
        self.assertEqual(selected, {"B": "on"})

    def test_tau_penalty(self):
        dc.BUDGET_LIMIT = 5.0
        dc.TAU_THRESHOLD = 5.0
        dc.LAST_STATE_CHANGE.clear()
        now = time.time()
        dc.record_plugin_state_change("A", now)
        h_t = {"A": {"cost": 1}, "B": {"cost": 1}}
        x_t = {"A": "on", "B": "on"}
        history = []
        selected = dc.decide_actions(h_t, x_t, history)
        print("selected with tau penalty:", selected)
        self.assertEqual(selected, {"B": "on"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
