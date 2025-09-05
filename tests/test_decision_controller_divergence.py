import unittest
import torch
import marble.decision_controller as dc
from marble.plugins import PLUGIN_ID_REGISTRY


class TestDecisionControllerDivergence(unittest.TestCase):
    def setUp(self) -> None:
        self.names = list(PLUGIN_ID_REGISTRY.keys())[:1]
        self.controller = dc.DecisionController(top_k=1)
        self.ctx = torch.zeros(1, 1, 16)

    def test_nan_metrics_flag(self) -> None:
        h_t = {self.names[0]: {"cost": 1.0}}
        metrics = {"latency": float("nan"), "throughput": 0.0, "cost": 0.0}
        self.controller.decide(h_t, self.ctx, metrics=metrics)
        self.assertTrue(self.controller.divergence)
        self.assertEqual(self.controller._reward_log[-1], -self.controller.reward_shaper.M_div)

    def test_inf_action_flag(self) -> None:
        h_t = {self.names[0]: {"cost": float("inf")}}
        metrics = {"latency": 0.0, "throughput": 0.0, "cost": 0.0}
        self.controller.decide(h_t, self.ctx, metrics=metrics)
        self.assertTrue(self.controller.divergence)
        self.assertEqual(self.controller._reward_log[-1], -self.controller.reward_shaper.M_div)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
