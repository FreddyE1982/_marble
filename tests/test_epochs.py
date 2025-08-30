import unittest


class TestEpochs(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, UniversalTensorCodec, make_datapair, REPORTER
        self.Brain = Brain
        self.Codec = UniversalTensorCodec
        self.make_dp = make_datapair
        self.reporter = REPORTER

    def test_epochs_over_pairs(self):
        b = self.Brain(2, size=(6, 6))
        codec = self.Codec()
        pairs = [
            self.make_dp({"label": 0}, 0.0),
            self.make_dp({"label": 1}, 1.0),
        ]
        from marble.marblemain import run_wanderer_epochs_with_datapairs

        pre_brain_id = id(b)
        pre_graph_id = id(b.neurons)

        result = run_wanderer_epochs_with_datapairs(
            b,
            pairs,
            codec,
            num_epochs=2,
            steps_per_pair=2,
            lr=1e-2,
            loss="nn.MSELoss",
        )
        self.assertEqual(id(b), pre_brain_id)
        self.assertEqual(id(b.neurons), pre_graph_id)
        print("epochs final_loss:", result["final_loss"])
        self.assertIn("epochs", result)
        self.assertEqual(len(result["epochs"]), 2)
        # Reporter entries exist
        ep0 = self.reporter.item("epoch_0", "training", "epochs")
        ep1 = self.reporter.item("epoch_1", "training", "epochs")
        print("epoch logs:", ep0, ep1)
        self.assertIsNotNone(ep0)
        self.assertIsNotNone(ep1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

