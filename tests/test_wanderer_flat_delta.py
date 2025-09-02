import unittest
import torch

from marble.marblemain import Brain, Wanderer


class TestWandererFlatDelta(unittest.TestCase):
    def test_flat_delta_marks_neuron(self) -> None:
        b = Brain(1, mode="sparse", sparse_bounds=((0.0, None),))
        n1 = b.add_neuron((0.0,), tensor=[0.0])
        n2 = b.add_neuron((1.0,), tensor=[0.0])
        b.connect((0.0,), (0.0,), direction="uni")
        w = Wanderer(b, seed=123, neuro_config={"max_flat_steps": 1})
        w._compute_loss = lambda outputs, override_loss=None: torch.tensor(0.0)
        w.walk(max_steps=3, lr=1e-2, start=n1)
        marks = getattr(b, "_prune_marks", {})
        print("flat_delta_marks", marks.get(n1, 0))
        self.assertIn(n1, marks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
