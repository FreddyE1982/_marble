import unittest
import torch
import marble.decision_controller as dc
import marble.plugin_cost_profiler as pcp
from marble.plugins import PLUGIN_ID_REGISTRY


class DecisionControllerAutoBudgetTests(unittest.TestCase):
    def test_auto_budget_inference_and_recalc(self):
        pcp.enable()
        dc.STEP_COUNTER = 0
        dc.PLUGIN_GRAPH.reset()
        name = list(PLUGIN_ID_REGISTRY.keys())[0]
        ctrl = dc.DecisionController(budget=None, warmup_steps=2, safety_factor=1.2, recalc_interval=2)
        ctx = torch.zeros(1, 1, 16)
        h = {name: {"cost": 1.0}}
        for _ in range(2):
            ctrl.decide(h, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1})
            pcp.record(name, h[name]["cost"])
        print("budget after warmup:", dc.BUDGET_LIMIT)
        self.assertAlmostEqual(dc.BUDGET_LIMIT, 1.0 * 1.2, delta=0.05)
        h[name]["cost"] = 2.0
        for _ in range(2):
            ctrl.decide(h, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1})
            pcp.record(name, h[name]["cost"])
        print("budget after recalibration:", dc.BUDGET_LIMIT)
        self.assertAlmostEqual(dc.BUDGET_LIMIT, 2.0 * 1.2, delta=0.05)
        dc.PLUGIN_GRAPH.reset()
