import unittest
import torch
import marble.decision_controller as dc


class TestContributionRegressor(unittest.TestCase):
    def test_contribution_influences_selection(self):
        torch.manual_seed(0)
        activation = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        outcomes = torch.tensor([2.0, -1.0, 1.0])
        contribs = dc.estimate_plugin_contributions(activation, outcomes, ["A", "B"], l1_penalty=0.01)
        h_t = {"A": {"cost": 5.0}, "B": {"cost": 4.0}}
        x_t = {"A": "on", "B": "on"}
        dc.BUDGET_LIMIT = 5.0
        selected = dc.decide_actions(h_t, x_t, [], contrib_scores=contribs)
        print("contribs:", contribs)
        print("selected with contribs:", selected)
        self.assertEqual(selected, {"A": "on"})

    def test_excess_contribution_does_not_increase_budget(self):
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        dc.BUDGET_LIMIT = 1.0
        h_t1 = {"A": {"cost": 1.0}}
        x_t1 = {"A": "on"}
        contribs = {"A": 2.0}
        rec: dict[str, float] = {}
        sel1 = dc.decide_actions(h_t1, x_t1, [], contrib_scores=contribs, cost_recorder=rec)
        history = [{n: {"action": a, "cost": rec.get(n, 0.0)} for n, a in sel1.items()}]
        h_t2 = {"B": {"cost": 2.0}}
        x_t2 = {"B": "on"}
        sel2 = dc.decide_actions(h_t2, x_t2, history, all_plugins=h_t2.keys())
        print("cost recorded for A:", rec.get("A"))
        print("second selection after excessive contribution:", sel2)
        self.assertEqual(rec.get("A"), 0.0)
        self.assertEqual(sel2, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
