import unittest
import types
import torch

from marble.marblemain import Brain, Wanderer


class TestNeuronPruningMechanism(unittest.TestCase):
    def test_prune_after_two_hits(self) -> None:
        b = Brain(1, mode="sparse", sparse_bounds=((0.0, None),))
        n1 = b.add_neuron((0.0,), tensor=[0.0])
        n2 = b.add_neuron((1.0,), tensor=[0.0], connect_to=(0.0,))
        # remove auto connection and reconnect both directions
        for s in list(getattr(n2, "outgoing", [])):
            if s.target is n1:
                b.remove_synapse(s)
        b.connect((0.0,), (1.0,), direction="uni")
        b.connect((1.0,), (0.0,), direction="uni")

        def no_record(self, diff: float) -> None:
            pass

        n1.record_loss_diff = types.MethodType(no_record, n1)
        n1.mean_loss_diff = 10.0

        w = Wanderer(b, seed=123)
        w._compute_loss = lambda outputs, override_loss=None: torch.tensor(0.0)
        stats = w.walk(max_steps=4, lr=1e-2, start=n1)
        print("pruning walk stats:", stats)
        self.assertEqual(b.neurons_pruned, 1)
        self.assertNotIn(n1, b.neurons.values())


if __name__ == "__main__":
    unittest.main(verbosity=2)
