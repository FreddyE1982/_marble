import unittest
from marble.progressbar import ProgressBar

class TestProgressBarLabels(unittest.TestCase):
    def test_neuron_label_replaces_brain(self):
        p = ProgressBar()
        p.start(1)
        p.update(
            cur_size=1,
            cap=2,
            cur_loss=0,
            mean_loss=0,
            loss_speed=0,
            mean_loss_speed=0,
            status={},
            synapses=0,
            mean_speed=0,
        )
        postfix = getattr(p._bar, "postfix", "")
        self.assertNotIn("brain=", postfix)
        self.assertIn("neurons=1/2", postfix)
        p.end(cur_ep=1, tot_ep=1, cur_walk=1, tot_walks=1, loss=0.0, steps=1)

if __name__ == "__main__":
    unittest.main()
