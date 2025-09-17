import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from marble.marblemain import Brain
from marble.wanderer import Wanderer


class TestTensorBoardProgressBar(unittest.TestCase):
    def test_tensorboard_walk_skips_progressbar(self) -> None:
        brain = Brain(1, size=(1,), tensorboard=True)
        start = brain.add_neuron((0,), tensor=[0.0])
        wanderer = Wanderer(brain, seed=0)

        buf = io.StringIO()
        with patch("marble.wanderer.ProgressBar") as mock_progress:
            mock_progress.side_effect = AssertionError("Progress bar should not start when tensorboard is enabled")
            with redirect_stdout(buf):
                stats = wanderer.walk(max_steps=1, start=start, lr=1e-2)

        output = buf.getvalue()
        self.assertFalse(brain.enable_progressbar)
        self.assertIsNotNone(brain.tensorboard_logdir)
        self.assertGreaterEqual(stats.get("steps", 0), 0)
        self.assertTrue(
            "TensorBoard inline display active" in output
            or "%tensorboard --logdir" in output
        )
        self.assertTrue(getattr(brain, "_tensorboard_announced", False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
