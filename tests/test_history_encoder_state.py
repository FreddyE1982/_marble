import unittest
import torch

from marble.decision_controller import DecisionController
from marble.plugins import PLUGIN_ID_REGISTRY


class TestHistoryEncoderState(unittest.TestCase):
    def test_state_updates_over_decisions(self):
        names = list(PLUGIN_ID_REGISTRY.keys())[:2]
        controller = DecisionController()
        hints = {n: {"cost": 1} for n in names}
        ctx = torch.zeros(1, 1, controller.encoder.ctx_rnn.hidden_size)
        controller.decide(hints, ctx)
        h1 = controller._h_t[0, 0].detach().clone()
        controller.decide(hints, ctx)
        h2 = controller._h_t[0, 0].detach().clone()
        print("h1 norm:", float(h1.norm()), "h2 norm:", float(h2.norm()))
        print("metric window after no metrics:", list(controller._metric_window))
        print("last metrics after no metrics:", controller.last_metrics)
        print("reward log after no metrics:", controller._reward_log)
        self.assertFalse(torch.allclose(h1, h2))
        self.assertEqual(len(controller._metric_window), 0)
        self.assertEqual(controller.last_metrics, {})
        self.assertEqual(len(controller._reward_log), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
