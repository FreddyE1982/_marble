import unittest


class SuperAdvancedNeuronPluginTests(unittest.TestCase):
    def test_plugins_register_and_expose_params(self) -> None:
        from marble.marblemain import Brain, Wanderer, _NEURON_TYPES

        plugins = {
            "chaotic_sine": ["chaos_r", "chaos_iters", "chaos_bias"],
            "entropic_mixer": ["ent_alpha", "ent_beta", "ent_bias"],
            "mirror_tanh": ["mirror_scale", "mirror_bias"],
            "spiral_time": ["spiral_freq", "spiral_phase", "spiral_bias"],
            "lattice_resonance": ["lattice_mod", "lattice_scale", "lattice_bias"],
        }

        for name, params in plugins.items():
            self.assertIn(name, _NEURON_TYPES)
            brain = Brain(1, size=1)
            w = Wanderer(brain)
            n = brain.add_neuron(brain.available_indices()[0], tensor=[0.0], type_name=name)
            n._plugin_state["wanderer"] = w
            plug = _NEURON_TYPES[name]
            plug.forward(n, input_value=[0.0])
            print("plugin", name, "learnables", list(w._learnables.keys()))
            for p in params:
                self.assertIn(p, w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

