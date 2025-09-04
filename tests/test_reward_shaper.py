import unittest
import torch

from marble.reward_shaper import RewardShaper
from marble.decision_controller import INCOMPATIBILITY_SETS


class TestRewardShaper(unittest.TestCase):
    def test_components_and_penalties(self):
        rs = RewardShaper(window_size=3)
        window = [
            {"latency": 3.0, "throughput": 1.0, "cost": 3.0, "wall_time": 0.0},
            {"latency": 2.0, "throughput": 2.0, "cost": 2.0, "wall_time": 1.0},
            {"latency": 1.0, "throughput": 3.0, "cost": 1.0, "wall_time": 2.0},
        ]
        action_mask = {"A": 1, "C": 1}
        delta_mask = {"A": 1, "C": 1}
        h_t = {"A": {"cost": 2.0}, "C": {"cost": 1.0}}
        reward, comps = rs.update(
            window, action_mask, delta_mask, h_t, INCOMPATIBILITY_SETS
        )
        self.assertLess(comps["latency_slope"], 0.0)
        self.assertGreater(comps["throughput_slope"], 0.0)
        self.assertLess(comps["cost_slope"], 0.0)
        self.assertEqual(comps["toggle_penalty"], 2.0)
        self.assertEqual(comps["compatibility_penalty"], 1.0)
        self.assertEqual(comps["compute_cost_penalty"], 3.0)
        self.assertIsInstance(reward, float)

    def test_divergence_indicator(self):
        rs = RewardShaper(window_size=3, M_div=0.1)
        window = [
            {"latency": 1.0, "throughput": 3.0, "cost": 1.0, "wall_time": 0.0},
            {"latency": 2.0, "throughput": 2.0, "cost": 2.0, "wall_time": 1.0},
            {"latency": 3.0, "throughput": 1.0, "cost": 3.0, "wall_time": 2.0},
        ]
        r, comps = rs.update(window, {}, {}, {})
        self.assertEqual(comps["divergence"], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
