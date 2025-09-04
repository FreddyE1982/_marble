import unittest
import torch

from marble.reward_shaper import RewardShaper


class TestRewardShaper(unittest.TestCase):
    def test_reward_and_window(self):
        rs = RewardShaper(window_size=3)
        data = [
            (3.0, 1.0, 3.0),
            (2.0, 2.0, 2.0),
            (1.0, 3.0, 1.0),
            (0.0, 4.0, 0.0),
        ]
        rewards = []
        for lat, thr, cost in data:
            r, betas = rs.update(lat, thr, cost)
            print("step", len(rewards), "reward", r, "betas", betas)
            rewards.append(r)
        # After initial fill, latency and cost decreasing -> negative betas
        self.assertLess(betas["latency"], 0.0)
        self.assertLess(betas["cost"], 0.0)
        self.assertGreater(betas["throughput"], 0.0)
        self.assertGreater(rewards[-1], 0.0)
        # Sliding window size should not exceed 3
        self.assertEqual(rs._lat.maxlen, 3)
        self.assertEqual(len(rs._lat), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
