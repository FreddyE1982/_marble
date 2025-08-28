import os
import tempfile
import unittest

from marble.marblemain import Brain
from marble.wanderer import Wanderer
from marble.reporter import clear_report_group


class TestBrainSnapshot(unittest.TestCase):
    def test_snapshot_save_and_load(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=3, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        b.add_neuron((0,), tensor=[1.0])
        b.add_neuron((1,), tensor=[0.0])
        b.connect((0,), (1,))
        w = Wanderer(b)
        w.walk(max_steps=1)
        files = [f for f in os.listdir(tmp) if f.endswith(".marble")]
        print("snapshot files count:", len(files))
        self.assertTrue(files)
        loaded = Brain.load_snapshot(os.path.join(tmp, files[0]))
        print("loaded brain neurons:", len(loaded.neurons))
        self.assertEqual(len(loaded.neurons), len(b.neurons))


if __name__ == "__main__":
    unittest.main(verbosity=2)
