import unittest


class UltraNeuroplasticityPluginSuiteTests(unittest.TestCase):
    def make_brain(self):
        from marble.marblemain import Brain

        b = Brain(2, size=(3, 3))
        idxs = list(b.available_indices())
        b.add_neuron(idxs[0], tensor=[0.0])
        b.add_neuron(idxs[1], tensor=[0.0])
        b.connect(idxs[0], idxs[1], direction="uni")
        return b

    def test_ultra_neuroplasticity_plugins_register_learnables(self):
        from marble.marblemain import Wanderer

        plugins = {
            "weight_shifter": ["shift_amount"],
            "bias_pulse": ["pulse_intensity"],
            "threshold_fader": ["fade_rate"],
            "synapse_bounce": ["bounce_scale"],
            "signal_echo": ["echo_strength"],
        }
        for name, params in plugins.items():
            with self.subTest(name=name):
                b = self.make_brain()
                w = Wanderer(b, neuroplasticity_type=name)
                w.walk(max_steps=1, lr=0.0)
                learnables = getattr(w, "_learnables", {})
                print("ultra neuroplasticity plugin", name, "learnables", list(learnables.keys()))
                for p in params:
                    self.assertIn(p, learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

