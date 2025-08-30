import unittest


class UltraSynapsePluginTests(unittest.TestCase):
    def _build_brain_with_plugin(self, plugin_name: str):
        from marble.marblemain import Brain

        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.add_neuron((1.0, 0.0), tensor=[0.0], weight=1.0, bias=0.0)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="uni", type_name=plugin_name)
        return b

    def _walk_and_collect(self, brain):
        from marble.marblemain import Wanderer

        w = Wanderer(brain, seed=1)
        start = brain.get_neuron((0.0, 0.0))
        w.walk(max_steps=1, lr=1e-2, start=start)
        return w

    def test_plugins_register_and_expose_params(self):
        plugin_params = {
            "superposition_surge": ["surge_alpha", "surge_beta"],
            "entropy_lens": ["lens_focus"],
            "dimensional_rift": ["rift_depth"],
            "causal_loop": ["loop_strength"],
            "membrane_oscillator": ["osc_freq", "osc_damp"],
        }
        for name, params in plugin_params.items():
            brain = self._build_brain_with_plugin(name)
            w = self._walk_and_collect(brain)
            print("ultra synapse plugin", name, "learnables", list(w._learnables.keys()))
            for p in params:
                self.assertIn(p, w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
