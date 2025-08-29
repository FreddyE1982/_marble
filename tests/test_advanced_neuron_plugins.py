import unittest


class AdvancedNeuronPluginTests(unittest.TestCase):
    def test_plugins_register_and_expose_params(self) -> None:
        from marble.marblemain import Brain, Wanderer, _NEURON_TYPES

        plugins = {
            "sinewave": ["sine_amp", "sine_freq", "sine_phase", "sine_bias"],
            "gaussian": ["gauss_mean", "gauss_sigma", "gauss_scale", "gauss_bias"],
            "polynomial": ["poly_a", "poly_b", "poly_c"],
            "exponential": ["exp_rate", "exp_scale", "exp_bias"],
            "rbf": ["rbf_center", "rbf_gamma", "rbf_scale", "rbf_bias"],
        }

        for name, params in plugins.items():
            self.assertIn(name, _NEURON_TYPES)
            brain = Brain(1, size=1)
            w = Wanderer(brain)
            n = brain.add_neuron(brain.available_indices()[0], tensor=[0.0], type_name=name)
            n._plugin_state["wanderer"] = w
            plug = _NEURON_TYPES[name]
            plug.forward(n, input_value=[0.0])
            for p in params:
                self.assertIn(p, w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

