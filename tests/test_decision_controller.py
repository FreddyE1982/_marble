import unittest
import time
import torch
import marble.decision_controller as dc
from marble.plugins import PLUGIN_ID_REGISTRY


class TestDecisionController(unittest.TestCase):
    def test_incompatibility_and_capacity(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.BUDGET_LIMIT = 10.0
        h_t = {"A": {"cost": 2}, "B": {"cost": 1}, "C": {"cost": 4}}
        x_t = {"A": "on", "B": "on", "C": "on"}
        history = [{"B": "on"}]
        selected = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("selected after constraints:", selected)
        self.assertEqual(selected, {"A": "on"})

    def test_budget_limit(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.BUDGET_LIMIT = 3.0
        h_t = {"A": {"cost": 2}, "B": {"cost": 1}, "C": {"cost": 4}}
        x_t = {"A": "on", "B": "on", "C": "on"}
        history = []
        selected = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("selected under budget:", selected)
        self.assertEqual(selected, {"B": "on", "A": "on"})

    def test_per_plugin_running_cost(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.BUDGET_LIMIT = 5.0
        h_t = {"A": {"cost": 3}, "B": {"cost": 3}}
        x_t = {"A": "on", "B": "on"}
        history = [{"A": "on"}]
        selected = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("selected with per-plugin budget:", selected)
        self.assertEqual(selected, {"B": "on"})

    def test_cost_change_respected(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.BUDGET_LIMIT = 3.5
        h_t1 = {"A": {"cost": 3}}
        x_t1 = {"A": "on"}
        history: list[dict] = []
        recorder: dict[str, float] = {}
        sel1 = dc.decide_actions(
            h_t1, x_t1, history, all_plugins=h_t1.keys(), cost_recorder=recorder
        )
        history.append(
            {n: {"action": a, "cost": recorder.get(n, 0.0)} for n, a in sel1.items()}
        )
        h_t2 = {"A": {"cost": 1}}
        x_t2 = {"A": "on"}
        sel2 = dc.decide_actions(h_t2, x_t2, history, all_plugins=h_t2.keys())
        print("second selection under budget after cost change:", sel2)
        self.assertEqual(sel2, {})

    def test_dynamic_cost_vector(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.BUDGET_LIMIT = 10.0
        name = list(PLUGIN_ID_REGISTRY.keys())[0]
        idx = PLUGIN_ID_REGISTRY[name]
        controller = dc.DecisionController(top_k=1)
        ctx = torch.zeros(1, 1, 16)
        h1 = {name: {"cost": 4.0}}
        controller.decide(h1, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1})
        g_budget = controller.agent.constraints[0]
        c1 = float(g_budget(torch.tensor([idx]))[0])
        h2 = {name: {"cost": 1.0}}
        controller.decide(h2, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1})
        c2 = float(g_budget(torch.tensor([idx]))[0])
        print("budget penalties:", c1, c2)
        self.assertAlmostEqual(c1, 4.0 / dc.BUDGET_LIMIT)
        self.assertAlmostEqual(c2, 1.0 / dc.BUDGET_LIMIT)
        dc.PLUGIN_GRAPH.reset()

    def test_tau_penalty(self):
        dc.BUDGET_LIMIT = 5.0
        dc.TAU_THRESHOLD = 5.0
        dc.LAST_STATE_CHANGE.clear()
        now = time.time()
        dc.record_plugin_state_change("A", now)
        h_t = {"A": {"cost": 1}, "B": {"cost": 1}}
        x_t = {"A": "on", "B": "on"}
        history = []
        selected = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("selected with tau penalty:", selected)
        self.assertEqual(selected, {"B": "on"})

    def test_decision_controller_cadence(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
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

    def test_dwell_bonus(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.BUDGET_LIMIT = 3.0
        dc.DWELL_BONUS = 1.0
        dc.DWELL_COUNT.clear()
        h_t = {"A": {"cost": 2}, "B": {"cost": 2}}
        x_t = {"A": "on", "B": "on"}
        history = []
        sel1 = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        sel2 = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("first selection:", sel1)
        print("second selection with dwell bonus:", sel2)
        self.assertEqual(sel1, {"A": "on"})
        self.assertEqual(sel2, {"A": "on"})

    def test_linear_constraints_accept(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.LINEAR_CONSTRAINTS_A = [[1, 1]]
        dc.LINEAR_CONSTRAINTS_B = [2]
        dc.BUDGET_LIMIT = 5.0
        h_t = {"B": {"cost": 1}, "C": {"cost": 1}}
        x_t = {"B": "on", "C": "on"}
        history = []
        selected = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("selection satisfying linear constraint:", selected)
        self.assertEqual(selected, {"B": "on", "C": "on"})

    def test_linear_constraints_reject(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.LINEAR_CONSTRAINTS_A = [[1, 1]]
        dc.LINEAR_CONSTRAINTS_B = [1]
        dc.BUDGET_LIMIT = 5.0
        h_t = {"B": {"cost": 1}, "C": {"cost": 1}}
        x_t = {"B": "on", "C": "on"}
        history = []
        selected = dc.decide_actions(h_t, x_t, history, all_plugins=h_t.keys())
        print("selection under tight linear constraint:", selected)
        self.assertEqual(selected, {"B": "on"})

    def test_multiple_selection_learning(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.LINEAR_CONSTRAINTS_A = []
        dc.LINEAR_CONSTRAINTS_B = []
        dc.BUDGET_LIMIT = 10.0
        names = list(PLUGIN_ID_REGISTRY.keys())[:3]
        controller = dc.DecisionController(top_k=2)
        torch.manual_seed(0)
        h_t = {n: {"cost": 0.0} for n in names}
        ctx = torch.zeros(1, 1, 16)
        captured = {}
        orig_step = controller.agent.step

        def fake_step(states, actions, returns):
            captured["actions"] = actions.clone()
            return orig_step(states, actions, returns)

        controller.agent.step = fake_step
        controller.decide(
            h_t,
            ctx,
            metrics={"latency": 1, "throughput": 1, "cost": 1},
        )
        self.assertEqual(len(captured.get("actions", [])), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
