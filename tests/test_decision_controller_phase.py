import unittest
import torch
import marble.decision_controller as dc
from marble.plugins import PLUGIN_ID_REGISTRY
from marble.action_sampler import compute_logits, sample_actions
from marble.reward_shaper import RewardShaper


class TestDecisionControllerPhase(unittest.TestCase):
    def _build(self, bias):
        ctrl = dc.DecisionController(
            top_k=1,
            sampler_mode="gumbel-top-k",
            phase_count=2,
        )
        ctrl.phase_proj.weight.data.zero_()
        ctrl.phase_proj.bias.data = torch.tensor(bias, dtype=torch.float32)
        ctrl.phase_bias.weight.data = torch.tensor(
            [[20.0, 0.0], [0.0, 20.0]], dtype=torch.float32
        )
        return ctrl

    def test_phase_bias_shifts_selection(self):
        names = list(PLUGIN_ID_REGISTRY.keys())[:2]
        h_t = {names[0]: {"cost": 1}, names[1]: {"cost": 1}}
        ctx = torch.zeros(1, 1, 16)

        ctrl0 = self._build([10.0, 0.0])
        sel0 = ctrl0.decide(
            h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1}
        )
        print("phase 0 selection:", sel0)

        ctrl1 = self._build([0.0, 10.0])
        sel1 = ctrl1.decide(
            h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1}
        )
        print("phase 1 selection:", sel1)

        self.assertEqual(set(sel0.keys()), {names[0]})
        self.assertEqual(set(sel1.keys()), {names[1]})

    def test_phase_count_one_matches_baseline(self):
        torch.manual_seed(0)
        dc.PLUGIN_GRAPH.reset()
        names = list(PLUGIN_ID_REGISTRY.keys())[:2]
        h_t = {names[0]: {"cost": 1}, names[1]: {"cost": 1}}
        ctx = torch.zeros(1, 1, 16)

        ctrl = dc.DecisionController(
            top_k=1, sampler_mode="gumbel-top-k", phase_count=1
        )
        self.assertIsNone(ctrl.phase_proj)

        plugin_ids = torch.tensor(
            [PLUGIN_ID_REGISTRY[n] for n in names], dtype=torch.long
        )
        past_ids = [0]
        ctx_rep = ctx.expand(len(plugin_ids), -1, -1)
        feats = ctrl.encoder(plugin_ids, ctx_rep, past_ids)
        o_t = feats.mean(dim=0)
        r_prev = torch.tensor([ctrl._prev_reward], dtype=o_t.dtype)
        ctrl._h_t = ctrl.history_encoder(
            o_t, ctrl._prev_action_vec.to(o_t), r_prev, ctrl._h_t
        )
        e_a_t = ctrl._h_t[0].squeeze(0)
        baseline_logits = compute_logits(feats, e_a_t)

        ctrl.cost_vec = torch.zeros(len(PLUGIN_ID_REGISTRY), dtype=torch.float32)
        for name, info in h_t.items():
            idx = PLUGIN_ID_REGISTRY.get(name)
            if idx is not None:
                ctrl.cost_vec[idx] = float(info.get("cost", 0.0))

        torch.manual_seed(0)
        mask = sample_actions(
            baseline_logits,
            mode="gumbel-top-k",
            top_k=1,
            costs=ctrl.cost_vec[plugin_ids],
            budget=dc.BUDGET_LIMIT,
            incompatibility={},
        )
        idx = (mask > 0.5).nonzero(as_tuple=False).squeeze(1)
        expected = {
            names[i]: h_t[names[i]].get("action", "on") for i in idx.tolist()
        }

        torch.manual_seed(0)
        sel = ctrl.decide(
            h_t, ctx, metrics={"latency": 1, "throughput": 1, "cost": 1}
        )
        print("baseline selection:", expected)
        print("controller selection:", sel)
        self.assertEqual(sel, expected)

        rs = RewardShaper()
        action_mask = {n: 1 if n in expected else 0 for n in names}
        delta_mask = action_mask
        base_reward, _ = rs.update(
            [{"latency": 1, "throughput": 1, "cost": 1}],
            action_mask,
            delta_mask,
            h_t,
            dc.INCOMPATIBILITY_SETS,
            force_divergence=False,
        )
        print("baseline reward:", base_reward)
        print("controller reward:", ctrl._prev_reward)
        self.assertEqual(ctrl._prev_reward, base_reward)

        for attr, base in zip(["w1", "w2", "w3", "w4", "w5", "w6"], ctrl._reward_base):
            self.assertEqual(getattr(ctrl.reward_shaper, attr), base)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
