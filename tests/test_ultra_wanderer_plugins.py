import unittest

from marble.marblemain import Brain, Wanderer


class UltraWandererPluginSuiteTests(unittest.TestCase):
    def make_brain(self):
        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")
        return b

    def test_ultra_wanderer_plugins_register_learnables(self):
        plugins = {
            "dimensionalshift": "shift_bias",
            "timewarp": "warp_strength",
            "echochaser": "echo_gain",
            "shadowclone": "clone_bias",
            "curvaturedrive": "curve_factor",
        }
        for name, param in plugins.items():
            with self.subTest(name=name):
                b = self.make_brain()
                w = Wanderer(b, type_name=name, seed=1)
                w.walk(max_steps=1, lr=0.0)
                print("ultra wanderer plugin", name, "learnables", list(w._learnables.keys()))
                self.assertIn(param, w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

