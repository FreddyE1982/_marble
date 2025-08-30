import unittest


class UltraBrainTrainPluginTests(unittest.TestCase):
    def _brain_and_wanderer(self):
        from marble.marblemain import Brain, Wanderer

        b = Brain(2, size=(4, 4))
        it = iter(b.available_indices())
        i1 = next(it)
        i2 = next(it)
        b.add_neuron(i1, tensor=0.0)
        b.add_neuron(i2, tensor=0.0)
        b.connect(i1, i2, direction="uni")
        w = Wanderer(b)
        return b, w, b.neurons.get(i1)

    def test_ultra_brain_train_plugins_register_learnables(self) -> None:
        from marble.marblemain import _BRAIN_TRAIN_TYPES

        plugins = {
            "quantum_jitter": ["jitter_factor"],
            "fractal_steps": ["fractal_dim"],
            "echo_repeater": ["echo_strength"],
            "gravity_anchor": ["gravity_bias"],
            "time_reverse": ["reverse_bias"],
        }

        for name, params in plugins.items():
            with self.subTest(name=name):
                self.assertIn(name, _BRAIN_TRAIN_TYPES)
                b, w, start = self._brain_and_wanderer()
                b.train(w, num_walks=1, max_steps=1, lr=1e-3, type_name=name, start_selector=lambda brain: start)
                learnables = getattr(w, "_learnables", {})
                print("ultra brain_train plugin", name, "learnables", list(learnables.keys()))
                for p in params:
                    self.assertIn(p, learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

