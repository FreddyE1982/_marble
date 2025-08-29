import unittest


class AdvancedBrainTrainPluginTests(unittest.TestCase):
    def _brain_and_wanderer(self):
        from marble.marblemain import Brain, Wanderer

        b = Brain(2, size=(4, 4))
        it = iter(b.available_indices())
        i1 = next(it)
        i2 = next(it)
        n1 = b.add_neuron(i1, tensor=0.0)
        n2 = b.add_neuron(i2, tensor=0.0)
        b.connect(i1, i2, direction="uni")
        w = Wanderer(b)
        return b, w, n1

    def test_plugins_register_and_expose_params(self) -> None:
        from marble.marblemain import _BRAIN_TRAIN_TYPES

        plugins = {
            "meta_optimizer": ["meta_lr"],
            "entropy_gate": ["entropy_gate"],
            "gradient_memory": ["memory_decay"],
            "temporal_mix": ["mix_ratio"],
            "sync_shift": ["phase_shift"],
        }

        for name, params in plugins.items():
            self.assertIn(name, _BRAIN_TRAIN_TYPES)
            b, w, start = self._brain_and_wanderer()
            b.train(w, num_walks=1, max_steps=1, lr=1e-3, type_name=name, start_selector=lambda brain: start)
            print("brain_train plugin", name, "learnables", list(w._learnables.keys()))
            for p in params:
                self.assertIn(p, w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

