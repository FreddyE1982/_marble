import unittest
import torch
import marble.decision_controller as dc
from marble.plugins import PLUGIN_ID_REGISTRY


class TestDecisionControllerContribution(unittest.TestCase):
    def test_controller_uses_contributions(self):
        torch.manual_seed(0)
        controller = dc.DecisionController(cadence=1, top_k=2)
        names = list(PLUGIN_ID_REGISTRY.keys())[:2]
        id0 = PLUGIN_ID_REGISTRY[names[0]]
        id1 = PLUGIN_ID_REGISTRY[names[1]]
        n = len(PLUGIN_ID_REGISTRY)
        act1 = torch.zeros(n); act1[id0] = 1
        act2 = torch.zeros(n); act2[id1] = 1
        act3 = torch.zeros(n); act3[id0] = 1; act3[id1] = 1
        controller._activation_log = [act1, act2, act3]
        controller._reward_log = [2.0, -1.0, 1.0]
        dc.BUDGET_LIMIT = 5.0
        h_t = {names[0]: {'cost': 5.0}, names[1]: {'cost': 4.0}}
        ctx = torch.zeros(1, 1, controller.encoder.embedding.embedding_dim)
        sel = controller.decide(h_t, ctx, metrics={'latency': 1, 'throughput': 1, 'cost': 1})
        print('selected with auto contributions:', sel)
        self.assertEqual(sel, {names[0]: 'on'})

    def test_pairwise_scores(self):
        torch.manual_seed(0)
        controller = dc.DecisionController(cadence=1)
        activation = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        outcomes = torch.tensor([0.0, 1.0, 1.0, 5.0])
        contribs = controller.compute_contributions(activation, outcomes, ["A", "B"])
        pair = contribs["pairwise"][("A", "B")]
        print('pairwise contribution:', pair)
        self.assertAlmostEqual(pair, 3.0, delta=0.1)


if __name__ == '__main__':  # pragma: no cover
    unittest.main(verbosity=2)
