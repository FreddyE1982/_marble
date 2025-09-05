import unittest
import torch
import marble.decision_controller as dc
from marble.plugins import PLUGIN_ID_REGISTRY


class TestDecisionControllerPhase(unittest.TestCase):
    def _build(self, bias):
        ctrl = dc.DecisionController(
            top_k=1,
            sampler_mode="gumbel-top-k",
            phase_count=2,
        )
        ctrl.phase_proj.weight.data.zero_()
        ctrl.phase_proj.bias.data = torch.tensor(bias, dtype=torch.float32)
        ctrl.phase_bias.weight.data = torch.tensor(
            [[20.0, 0.0], [0.0, 20.0]], dtype=torch.float32
        )
        return ctrl

    def test_phase_bias_shifts_selection(self):
        names = list(PLUGIN_ID_REGISTRY.keys())[:2]
        h_t = {names[0]: {"cost": 1}, names[1]: {"cost": 1}}
        ctx = torch.zeros(1, 1, 16)

        ctrl0 = self._build([10.0, 0.0])
        sel0 = ctrl0.decide(
            h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1}
        )
        print("phase 0 selection:", sel0)

        ctrl1 = self._build([0.0, 10.0])
        sel1 = ctrl1.decide(
            h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1}
        )
        print("phase 1 selection:", sel1)

        self.assertEqual(set(sel0.keys()), {names[0]})
        self.assertEqual(set(sel1.keys()), {names[1]})


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
