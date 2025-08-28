import unittest


class TestWandererHelperAndSynapse(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, run_wanderer_training, Wanderer
        self.Brain = Brain
        self.run_wanderer_training = run_wanderer_training
        self.Wanderer = Wanderer

    def test_synapse_weight_scaling(self):
        from marble.marblemain import Synapse

        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n2 = b.add_neuron((1.0, 0.0), tensor=[0.0], weight=1.0, bias=0.0)
        s = b.connect((0.0, 0.0), (1.0, 0.0), direction="uni")
        # Ensure synapse has weight and scaling applies
        self.assertTrue(hasattr(s, "weight"))
        s.weight = 2.0
        s.transmit([3.0], direction="forward")
        out = n2.forward()
        out = out.tolist() if hasattr(out, "tolist") else out
        print("synapse weighted transmit output:", out)
        self.assertEqual([round(v, 6) for v in out], [6.0])

    def test_run_wanderer_training_history(self):
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        result = self.run_wanderer_training(b, num_walks=3, max_steps=3, lr=1e-2, seed=42, loss="nn.MSELoss")
        print("run_wanderer_training final_loss:", result.get("final_loss"))
        print("run_wanderer_training steps in first walk:", len(result["history"][0]["step_metrics"]))
        self.assertIn("history", result)
        self.assertEqual(len(result["history"]), 3)
        # Per-step metrics present
        self.assertIn("step_metrics", result["history"][0])
        self.assertGreaterEqual(len(result["history"][0]["step_metrics"]), 1)
        step0 = result["history"][0]["step_metrics"][0]
        self.assertIn("loss", step0)
        self.assertIn("delta", step0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
