import unittest
import torch

from marble.marblemain import Brain, Wanderer


class AdvancedWandererPluginSuiteTests(unittest.TestCase):
    def make_brain(self):
        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        return b

    def test_advanced_wanderer_plugins_register_learnables(self):
        plugins = {
            "chaoswalk": "chaos_lambda",
            "quantumsuper": "collapse_prob",
            "gravitywell": "gravity_strength",
            "memorydecay": "memory_half_life",
            "fractalseed": "fractal_depth",
        }
        for name, param in plugins.items():
            with self.subTest(name=name):
                b = self.make_brain()
                w = Wanderer(b, type_name=name, seed=1)
                w.walk(max_steps=1, lr=0.0)
                print("advanced wanderer plugin", name, "learnables", list(w._learnables.keys()))
                self.assertIn(param, w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

