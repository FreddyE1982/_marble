import unittest


class UltraNeuronPluginTests(unittest.TestCase):
    def test_plugins_register_and_expose_params(self) -> None:
        from marble.marblemain import Brain, Wanderer, _NEURON_TYPES

        plugins = {
            "quantum_mirror": ["mirror_coeff", "mirror_bias"],
            "temporal_fission": ["fiss_amp", "fiss_phase"],
            "phantom_harmonics": ["phantom_scale", "phantom_shift", "phantom_bias"],
            "neutrino_field": ["field_strength", "field_decay"],
            "entropy_burst": ["burst_intensity", "burst_bias"],
        }

        for name, params in plugins.items():
            self.assertIn(name, _NEURON_TYPES)
            brain = Brain(1, size=1)
            w = Wanderer(brain)
            n = brain.add_neuron(brain.available_indices()[0], tensor=[0.0], type_name=name)
            n._plugin_state["wanderer"] = w
            plug = _NEURON_TYPES[name]
            plug.forward(n, input_value=[0.0])
            print("ultra neuron plugin", name, "learnables", list(w._learnables.keys()))
            for p in params:
                self.assertIn(p, w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
