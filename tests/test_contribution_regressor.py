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


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
