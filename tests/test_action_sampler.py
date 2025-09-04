import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from marble.action_sampler import select_plugins


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
