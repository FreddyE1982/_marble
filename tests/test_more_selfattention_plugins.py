import unittest


class MoreSelfAttentionPluginTests(unittest.TestCase):
    def make_brain(self):
        from marble.marblemain import Brain
        b = Brain(1, size=2)
        idxs = list(b.available_indices())
        first = b.add_neuron(idxs[0], tensor=[0.0])
        if len(idxs) > 1:
            second = b.add_neuron(idxs[1], tensor=[0.0], connect_to=idxs[0])
            # remove auto synapse from second->first and recreate bidirectional link
            for s in list(getattr(second, "outgoing", [])):
                if s.target is first:
                    b.remove_synapse(s)
            b.connect(idxs[0], idxs[1], direction="bi")
        return b

    def test_plugins_register_learnables(self):
        from marble.marblemain import Wanderer, SelfAttention, attach_selfattention, REPORTER
        REPORTER.clear_group("selfattention")
        plugins = {
            "step_fader": ["slope"],
            "loss_variance_temp": ["window", "factor"],
            "synapse_renorm": ["target_norm"],
            "neuron_swap": ["swap_prob"],
            "time_gate_temp": ["interval", "gated_temp"],
            "age_prune": ["max_age", "cool_temp"],
            "activation_boost_lr": ["threshold", "boost"],
            "weight_decay": ["decay"],
            "synapse_noise": ["noise_std"],
            "loss_center_lr": ["target_loss", "scale"],
        }
        for name, params in plugins.items():
            with self.subTest(name=name):
                b = self.make_brain()
                w = Wanderer(b, seed=1)
                sa = SelfAttention(type_name=name)
                attach_selfattention(w, sa)
                w.walk(max_steps=1, lr=0.0)
                learnables = getattr(w, "_learnables", {})
                print("selfattention plugin", name, "learnables", list(learnables.keys()))
                for p in params:
                    self.assertIn(p, learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
