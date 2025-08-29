import unittest


class TestNewNeuronPlugins(unittest.TestCase):
    def test_registration_and_params(self) -> None:
        from marble.marblemain import Brain, Wanderer, _NEURON_TYPES

        plugins = {
            "quantum_tunnel": ["qt_barrier"],
            "fractal_logistic": ["fractal_r", "fractal_iters"],
            "hyperbolic_blend": ["hyper_alpha"],
            "oscillating_decay": ["osc_freq", "osc_decay"],
            "echo_mix": ["echo_memory"],
        }

        for name, params in plugins.items():
            self.assertIn(name, _NEURON_TYPES)
            brain = Brain(1, size=1)
            w = Wanderer(brain)
            n = brain.add_neuron(
                brain.available_indices()[0], tensor=[0.0], type_name=name
            )
            n._plugin_state["wanderer"] = w
            plug = _NEURON_TYPES[name]
            plug.forward(n, input_value=[0.0])
            print("plugin", name, "learnables", list(w._learnables.keys()))
            for p in params:
                self.assertIn(p, w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

