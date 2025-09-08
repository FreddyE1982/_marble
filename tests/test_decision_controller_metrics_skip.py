import unittest
import torch
import marble.decision_controller as dc
from marble.decision_controller import DecisionController


class TestDecisionControllerMetricsSkip(unittest.TestCase):
    def test_action_reward_alignment(self) -> None:
        dc.PLUGIN_GRAPH.reset()
        dc.LAST_STATE_CHANGE.clear()
        dc.TAU_THRESHOLD = 0.0
        name = list(dc.PLUGIN_ID_REGISTRY.keys())[0]
        h_t = {name: {"cost": 1.0}}
        ctx = torch.zeros(1, 1, 16)
        ctrl = DecisionController(top_k=1)

        sel1 = ctrl.decide(h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1})
        print("sel1", sel1)
        prev_action_vec = ctrl._prev_action_vec.clone()
        prev_reward = ctrl._prev_reward

        sel2 = ctrl.decide(h_t, ctx, metrics=None)
        print("sel2", sel2)
        self.assertTrue(torch.equal(ctrl._prev_action_vec, prev_action_vec))
        self.assertEqual(ctrl._prev_reward, prev_reward)

        sel3 = ctrl.decide(h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1})
        print("sel3", sel3)
        self.assertEqual(len(ctrl._activation_log), 2)
        self.assertEqual(len(ctrl._reward_log), 2)
        self.assertTrue(torch.equal(ctrl._prev_action_vec, ctrl._activation_log[-1]))
        self.assertEqual(ctrl._prev_reward, ctrl._reward_log[-1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
