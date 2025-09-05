import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import marble.decision_controller as dc
from marble.decision_controller import DecisionController


class TestDecisionControllerDwell(unittest.TestCase):
    def test_dwell_suppresses_quick_switch(self) -> None:
        torch.manual_seed(0)
        dc.PLUGIN_GRAPH.recommend_next_plugin = lambda _: set()
        ctrl = DecisionController(dwell_threshold=2)
        ctx = torch.zeros(1, 1, 16)
        first = ctrl.decide({"auto_target_scaler": {"action": "on"}}, ctx)
        second = ctrl.decide(
            {
                "auto_target_scaler": {"action": "off"},
                "autoneuron": {"action": "on"},
            },
            ctx,
        )
        print("first", first, "second", second)
        self.assertIn("auto_target_scaler", first)
        self.assertEqual(second, {})


if __name__ == "__main__":
    unittest.main()

