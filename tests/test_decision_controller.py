import unittest
import time
import torch
import marble.decision_controller as dc
from marble.plugins import PLUGIN_ID_REGISTRY


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

    def test_decision_controller_cadence(self):
        dc.STEP_COUNTER = 0
        names = list(PLUGIN_ID_REGISTRY.keys())[:2]
        controller = dc.DecisionController(cadence=2, top_k=1)
        h_t = {names[0]: {"cost": 1}, names[1]: {"cost": 1}}
        ctx = torch.zeros(1, 1, 16)
        sel1 = controller.decide(h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1})
        print("first cadence selection:", sel1)
        self.assertEqual(sel1, {})
        sel2 = controller.decide(h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1})
        print("second cadence selection:", sel2)
        self.assertTrue(set(sel2).issubset(set(names)))

    def test_decide_with_contributions(self):
        dc.STEP_COUNTER = 0
        names = list(PLUGIN_ID_REGISTRY.keys())[:2]
        controller = dc.DecisionController(top_k=2, use_bayesian=False)
        dc.BUDGET_LIMIT = 5.0
        h_t = {names[0]: {"cost": 5.0}, names[1]: {"cost": 4.0}}
        ctx = torch.zeros(1, 1, 16)
        activation = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        outcomes = torch.tensor([2.0, -1.0, 1.0])
        sel = controller.decide(
            h_t,
            ctx,
            metrics={"latency": 1, "throughput": 1, "cost": 1},
            activation=activation,
            outcomes=outcomes,
        )
        print("selection with contributions:", sel)
        self.assertEqual(sel, {names[0]: "on"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
