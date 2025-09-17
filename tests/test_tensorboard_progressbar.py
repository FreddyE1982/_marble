import glob
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from marble.marblemain import Brain
from marble.wanderer import Wanderer
from marble.reporter import REPORTER, _TensorBoardAdapter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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

    def test_tensorboard_graph_contains_neurons_and_synapses(self) -> None:
        adapter_backup = REPORTER._tensorboard
        with tempfile.TemporaryDirectory() as tmpdir:
            new_adapter = _TensorBoardAdapter(True, tmpdir, 1)
            REPORTER._tensorboard = new_adapter
            try:
                brain = Brain(1, size=(2,), tensorboard=True)
                brain.add_neuron((0,), tensor=[0.0])
                brain.add_neuron((1,), tensor=[0.0], connect_to=(0,))
                REPORTER.flush_tensorboard()
                new_adapter.close()

                event_files = glob.glob(os.path.join(tmpdir, "events.out.tfevents.*"))
                self.assertTrue(event_files, "Expected TensorBoard event files to be created")

                accumulator = EventAccumulator(tmpdir)
                accumulator.Reload()
                graph_def = accumulator.Graph()
                self.assertIsNotNone(graph_def)

                node_names = {node.name for node in graph_def.node}
                self.assertIn("brain/grid", node_names)
                self.assertIn("neuron/[0]", node_names)
                self.assertIn("neuron/[1]", node_names)
                self.assertTrue(any(name.startswith("synapse/") for name in node_names))

                adjacency = {node.name: set(node.input) for node in graph_def.node}
                synapse_name = next(name for name in node_names if name.startswith("synapse/"))
                sources = adjacency.get(synapse_name, set())
                targets = {name for name, inputs in adjacency.items() if synapse_name in inputs}
                connections = {(src, dst) for src in sources for dst in targets}
                self.assertTrue(
                    ("neuron/[0]", "neuron/[1]") in connections
                    or ("neuron/[1]", "neuron/[0]") in connections,
                    "Expected synapse to connect the two neurons",
                )
            finally:
                try:
                    new_adapter.close()
                except Exception:
                    pass
                REPORTER._tensorboard = adapter_backup


if __name__ == "__main__":
    unittest.main(verbosity=2)
