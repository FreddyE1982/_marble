import unittest


class UltraSelfAttentionPluginSuiteTests(unittest.TestCase):
    def make_brain(self):
        from marble.marblemain import Brain

        b = Brain(1, size=2)
        idxs = list(b.available_indices())
        b.add_neuron(idxs[0], tensor=[0.0])
        if len(idxs) > 1:
            b.add_neuron(idxs[1], tensor=[0.0])
            b.connect(idxs[0], idxs[1], direction="bi")
        return b

    def test_ultra_selfattention_plugins_register_learnables(self):
        from marble.marblemain import Wanderer, SelfAttention, attach_selfattention, REPORTER

        REPORTER.clear_group("selfattention")
        plugins = {
            "phase_shift": ["phase_shift"],
            "signal_booster": ["boost_gain"],
            "residue_norm": ["residue_bias"],
            "tunnel_vision": ["tunnel_focus"],
            "temporal_echo": ["echo_decay"],
        }
        for name, params in plugins.items():
            with self.subTest(name=name):
                b = self.make_brain()
                w = Wanderer(b, seed=1)
                sa = SelfAttention(type_name=name)
                attach_selfattention(w, sa)
                w.walk(max_steps=1, lr=0.0)
                learnables = getattr(w, "_learnables", {})
                print("ultra selfattention plugin", name, "learnables", list(learnables.keys()))
                for p in params:
                    self.assertIn(p, learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

