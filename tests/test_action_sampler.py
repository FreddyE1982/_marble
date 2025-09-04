import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from marble.action_sampler import select_plugins, sample_actions


class TestActionSampler(unittest.TestCase):
    def test_bernoulli_sampling(self) -> None:
        plugin_ids = torch.tensor([5, 6, 7])
        e_t = torch.tensor([[20.0, 0.0], [-20.0, 0.0], [-20.0, 0.0]])
        e_a = torch.tensor([1.0, 0.0])
        selected = select_plugins(plugin_ids, e_t, e_a, mode="bernoulli")
        print("bernoulli selected ids:", selected)
        self.assertEqual(selected, [5])

    def test_gumbel_topk_sampling(self) -> None:
        torch.manual_seed(0)
        plugin_ids = torch.tensor([10, 11, 12, 13])
        e_t = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ])
        e_a = torch.ones(4)
        selected = select_plugins(plugin_ids, e_t, e_a, mode="gumbel-top-k", top_k=2)
        print("gumbel selected ids:", selected)
        self.assertEqual(selected, [11, 13])

    def test_relaxed_sampling_gradients(self) -> None:
        torch.manual_seed(0)
        logits = torch.zeros(3, requires_grad=True)
        costs = torch.ones(3)
        mask = sample_actions(
            logits,
            mode="bernoulli-relaxed",
            temperature=0.5,
            costs=costs,
            budget=2.0,
            incompatibility={},
        )
        print("relaxed mask:", mask)
        loss = mask.sum()
        loss.backward()
        print("gradients:", logits.grad)
        self.assertTrue(mask.requires_grad)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.any(logits.grad != 0))

    def test_projection_constraints(self) -> None:
        torch.manual_seed(0)
        logits = torch.ones(3)
        costs = torch.ones(3)
        incompat = {0: {1}, 1: {0}}
        mask = sample_actions(
            logits,
            mode="bernoulli-relaxed",
            temperature=0.5,
            costs=costs,
            budget=2.0,
            incompatibility=incompat,
        ).detach()
        print("projected mask:", mask)
        self.assertLessEqual(float((mask * costs).sum().item()), 2.0)
        self.assertLessEqual(float(mask[0] + mask[1]), 1.0)
        self.assertTrue(set(mask.tolist()) <= {0.0, 1.0})


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
