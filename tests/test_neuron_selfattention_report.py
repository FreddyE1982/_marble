import unittest


class TestNeuronSelfAttentionReport(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, SelfAttention
        self.Brain = Brain
        self.SelfAttention = SelfAttention

    def test_neuron_report(self):
        b = self.Brain(1, size=(3,))
        n = b.add_neuron((0,), weight=1.5, type_name="sigmoid")
        sa = self.SelfAttention()
        info = sa.get_neuron_report(n)
        print("neuron report:", info)
        self.assertEqual(info["weight"], 1.5)
        self.assertEqual(info["type_name"], "sigmoid")
        self.assertEqual(info["position"], (0,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
