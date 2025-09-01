import unittest
import torch


class TestNeuronLossDiffTracking(unittest.TestCase):
    def test_loss_diff_recording(self):
        torch.manual_seed(0)
        from marble.marblemain import Brain, Wanderer

        brain = Brain(1)
        n1 = brain.add_neuron((0,), tensor=1.0)
        n2 = brain.add_neuron((1,), tensor=0.0, weight=2.0)
        brain.connect(n1.position, n2.position)

        w = Wanderer(brain, seed=0)
        w.walk(max_steps=2, start=n1, lr=0.01)

        print("n1 loss diffs:", list(n1.loss_diffs), "mean:", n1.mean_loss_diff)
        print("n2 loss diffs:", list(n2.loss_diffs), "mean:", n2.mean_loss_diff)
        self.assertEqual(list(n1.loss_diffs), [0.0])
        self.assertAlmostEqual(n1.mean_loss_diff, 0.0, places=6)
        self.assertEqual(list(n2.loss_diffs), [3.0])
        self.assertAlmostEqual(n2.mean_loss_diff, 3.0, places=6)

    def test_loss_diff_window(self):
        from marble.marblemain import Neuron

        n = Neuron(0.0, loss_diff_window=2)
        n.record_loss_diff(1.0)
        n.record_loss_diff(2.0)
        n.record_loss_diff(3.0)
        print("history after diffs:", list(n.loss_diffs), "mean:", n.mean_loss_diff)
        self.assertEqual(list(n.loss_diffs), [2.0, 3.0])
        self.assertAlmostEqual(n.mean_loss_diff, 2.5, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
