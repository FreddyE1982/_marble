import unittest

class TestBrainStatus(unittest.TestCase):
    def setUp(self):
        from marble.reporter import clear_report_group
        clear_report_group("brain")
        clear_report_group("training")
        from marble.marblemain import Brain, UniversalTensorCodec, make_datapair, run_training_with_datapairs
        self.Brain = Brain
        self.Codec = UniversalTensorCodec
        self.make_dp = make_datapair
        self.train = run_training_with_datapairs

    def test_brain_status_counts(self):
        b = self.Brain(2)
        b.add_neuron((1, 1))
        b.add_neuron((0, 0), connect_to=(1, 1), direction="uni")
        b.remove_neuron(b.get_neuron((1, 1)))
        status = b.status()
        print("brain status:", status)
        self.assertIn("neurons_added", status)
        self.assertGreaterEqual(status["neurons_added"], 2)
        self.assertGreaterEqual(status["synapses_added"], 1)

    def test_training_status(self):
        b = self.Brain(2)
        codec = self.Codec()
        pairs = [self.make_dp(0.0, 0.0)]
        res = self.train(b, pairs, codec, steps_per_pair=1, lr=1e-3)
        print("training status:", res.get("status"))
        self.assertIn("status", res)
        self.assertIn("plugins_active", res["status"])
        # ensure new counters are exposed through training status
        self.assertIn("neurons_added", res["status"])
        self.assertGreaterEqual(res["status"]["neurons_added"], 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
