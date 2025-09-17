import os
import tempfile
import unittest

from marble.marblemain import Brain, UniversalTensorCodec
from marble.reporter import clear_report_group


class TestBrainSnapshot(unittest.TestCase):
    def test_snapshot_save_and_load(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=3, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        codec = UniversalTensorCodec()
        tokens = codec.encode("foo bar foo bar")
        b.codec = codec
        b.add_neuron((1,), tensor=[0.0])
        b.add_neuron((0,), tensor=[1.0], connect_to=(1,), direction="uni")
        snap_path = b.save_snapshot()
        print("snapshot path:", snap_path)
        loaded = Brain.load_snapshot(snap_path)
        print("loaded brain neurons:", len(loaded.neurons))
        self.assertEqual(len(loaded.neurons), len(b.neurons))
        self.assertTrue(hasattr(loaded, "codec"))
        decoded = loaded.codec.decode(tokens)
        self.assertEqual(decoded, "foo bar foo bar")

    def test_snapshot_dynamic_brain_without_size(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=None, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        b.add_neuron((0,), tensor=[0.0])
        b.add_neuron((1,), tensor=[1.0], connect_to=(0,), direction="uni")
        snap_path = b.save_snapshot()
        self.assertTrue(os.path.exists(snap_path))
        self.assertIn(os.path.basename(snap_path), os.listdir(tmp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
