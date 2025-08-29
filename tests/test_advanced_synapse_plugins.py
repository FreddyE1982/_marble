import unittest


class AdvancedSynapsePluginSuiteTests(unittest.TestCase):
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

    def test_advanced_synapse_plugins_register_learnables(self):
        plugin_params = {
            "echo_chamber": ["echo_decay", "echo_depth"],
            "quantum_flip": ["flip_prob"],
            "nonlocal_tunnel": ["tunnel_strength"],
            "fractal_enhance": ["fract_depth", "fract_scale"],
            "phase_noise": ["noise_freq", "noise_amp"],
        }
        for name, params in plugin_params.items():
            brain = self._build_brain_with_plugin(name)
            w = self._walk_and_collect(brain)
            print("advanced synapse plugin", name, "learnables", list(w._learnables.keys()))
            for p in params:
                self.assertIn(p, w._learnables, f"{name} missing {p}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

