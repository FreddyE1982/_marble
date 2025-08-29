import unittest


class TestAutoNeuron(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer, register_neuron_type

        class FailingPlugin:
            def forward(self, neuron, input_value=None):
                raise RuntimeError("boom")

        self.Brain = Brain
        self.Wanderer = Wanderer
        self.register_neuron_type = register_neuron_type
        self.register_neuron_type("fail", FailingPlugin())

    def test_fallback_on_error(self):
        b = self.Brain(1, size=(1,))
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = self.Wanderer(b, seed=0)
        w.ensure_learnable_param("autoneuron_bias_fail", 10.0)
        w.ensure_learnable_param("autoneuron_bias_base", -10.0)
        res = w.walk(max_steps=1, start=n, lr=0.0)
        self.assertIn("loss", res)
        self.assertEqual(n.type_name, "autoneuron")


if __name__ == "__main__":
    unittest.main(verbosity=2)

