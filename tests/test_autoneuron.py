import unittest
import math


class TestAutoNeuron(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer, register_neuron_type

        class FailingPlugin:
            def __init__(self):
                self.calls = 0

            def forward(self, neuron, input_value=None):
                self.calls += 1
                raise RuntimeError("boom")

        self.Brain = Brain
        self.Wanderer = Wanderer
        self.register_neuron_type = register_neuron_type
        self.fail_impl = FailingPlugin()
        self.register_neuron_type("fail", self.fail_impl)

    def test_fallback_on_error(self):
        b = self.Brain(1, size=(1,))
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = self.Wanderer(b, seed=0)
        w.ensure_learnable_param("autoneuron_bias_fail", 10.0)
        w.ensure_learnable_param("autoneuron_bias_base", -10.0)
        res = w.walk(max_steps=1, start=n, lr=0.0)
        print("walk result:", res, "fail calls:", self.fail_impl.calls)
        self.assertIn("loss", res)
        self.assertGreater(self.fail_impl.calls, 0)
        self.assertEqual(n.type_name, "autoneuron")

    def test_exclude_prevents_usage(self):
        from marble.marblemain import register_neuron_type
        from marble.plugins.autoneuron import AutoNeuronPlugin

        class FailingPlugin:
            def __init__(self):
                self.calls = 0

            def forward(self, neuron, input_value=None):
                self.calls += 1
                raise RuntimeError("boom")

        fail_impl = FailingPlugin()
        register_neuron_type("fail2", fail_impl)
        register_neuron_type("autoneuron_nofail", AutoNeuronPlugin(disabled_types=["fail2"]))

        b = self.Brain(1, size=(1,))
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[1.0], type_name="autoneuron_nofail")
        w = self.Wanderer(b, seed=0)
        w.ensure_learnable_param("autoneuron_bias_fail2", 10.0)
        w.ensure_learnable_param("autoneuron_bias_base", -10.0)
        res = w.walk(max_steps=1, start=n, lr=0.0)
        print("walk result:", res, "fail calls:", fail_impl.calls)
        self.assertIn("loss", res)
        self.assertEqual(fail_impl.calls, 0)

    def test_attention_score_handles_large_metrics(self):
        from marble.plugins.autoneuron import AutoNeuronPlugin

        class Dummy:
            _last_walk_mean_loss = 1.0
            _walk_step_count = 10 ** 9
            brain = type("B", (), {"neurons": list(range(10 ** 6)), "synapses": []})()

        plug = AutoNeuronPlugin()
        score = plug._attention_score(Dummy())
        self.assertTrue(math.isfinite(float(score)))
        self.assertLess(float(score), 1e3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

