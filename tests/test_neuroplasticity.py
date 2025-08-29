import unittest


class TestNeuroplasticity(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, UniversalTensorCodec, make_datapair, REPORTER
        self.Brain = Brain
        self.Codec = UniversalTensorCodec
        self.make_dp = make_datapair
        self.reporter = REPORTER
        self.reporter.clear_group("neuroplasticity")

    def test_base_plugin_grows_graph_when_no_outgoing(self):
        b = self.Brain(2, size=(6, 6))
        codec = self.Codec()

        # Ensure brain has at least one neuron with no outgoing
        if not b.neurons:
            idxs = b.available_indices()
            start_idx = idxs[0] if idxs else (0, 0)
            n0 = b.add_neuron(start_idx, tensor=[0.0])
        else:
            n0 = next(iter(b.neurons.values()))

        from marble.marblemain import run_training_with_datapairs
        before_neuron_count = len(b.neurons)

        dp = self.make_dp({"x": 1}, 0.0)
        _ = run_training_with_datapairs(b, [dp], codec, steps_per_pair=1, lr=1e-2, loss="nn.MSELoss")

        after_neuron_count = len(b.neurons)
        print("neuroplasticity grow count:", before_neuron_count, "->", after_neuron_count)

        # Neuron count must increase
        self.assertGreater(after_neuron_count, before_neuron_count)
        # Reporter emitted events
        init_log = self.reporter.item("init", "neuroplasticity", "events")
        grow_log = self.reporter.item("grow", "neuroplasticity", "events")
        self.assertIsNotNone(init_log)
        self.assertIsNotNone(grow_log)


if __name__ == "__main__":
    unittest.main(verbosity=2)

