import unittest


class TestTrainingWithDataPairs(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, UniversalTensorCodec, make_datapair, REPORTER
        self.Brain = Brain
        self.Codec = UniversalTensorCodec
        self.make_dp = make_datapair
        self.reporter = REPORTER

    def test_runs_over_pairs(self):
        # Simple 2D brain with default mandelbrot occupancy
        b = self.Brain(2, size=(6, 6))
        codec = self.Codec()

        # Three simple datapairs: left context is a label/int, right is numeric target
        pairs = [
            self.make_dp({"label": 0}, 0.0),
            self.make_dp({"label": 1}, 1.0),
            self.make_dp({"label": 2}, 0.5),
        ]

        from marble.marblemain import run_training_with_datapairs

        result = run_training_with_datapairs(
            b,
            pairs,
            codec,
            steps_per_pair=3,
            lr=5e-3,
            loss="nn.MSELoss",
        )

        print("datapair training final_loss:", result["final_loss"], "count:", result["count"])
        self.assertEqual(result["count"], 3)
        self.assertIn("history", result)
        self.assertEqual(len(result["history"]), 3)

        # Reporter captures summary
        summary = self.reporter.item("datapair_summary", "training", "datapair")
        print("datapair training summary log:", summary)
        self.assertIsNotNone(summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)

