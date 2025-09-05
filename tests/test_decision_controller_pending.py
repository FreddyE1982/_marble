import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import marble.decision_controller as dc
from marble.decision_controller import DecisionController


class TestDecisionControllerPending(unittest.TestCase):
    def test_pending_graph_returns_empty_selection(self) -> None:
        torch.manual_seed(0)
        dc.PLUGIN_GRAPH.reset()
        # Create a dependency cycle so that no plugin becomes ready
        dc.PLUGIN_GRAPH.add_dependency("auto_target_scaler", "autoneuron")
        dc.PLUGIN_GRAPH.add_dependency("autoneuron", "auto_target_scaler")
        ctrl = DecisionController()
        ctx = torch.zeros(1, 1, 16)
        selection = ctrl.decide(
            {
                "auto_target_scaler": {"action": "on"},
                "autoneuron": {"action": "on"},
            },
            ctx,
        )
        print("selection", selection)
        self.assertEqual(selection, {})


if __name__ == "__main__":
    unittest.main()

