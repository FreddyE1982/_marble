import os
import pickle
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
        with open(snap_path, "rb") as saved:
            header = saved.read(2)
        self.assertEqual(header, b"\x1f\x8b")
        print("snapshot path:", snap_path)
        loaded = Brain.load_snapshot(snap_path)
        print("loaded brain neurons:", len(loaded.neurons))
        self.assertEqual(len(loaded.neurons), len(b.neurons))
        self.assertTrue(hasattr(loaded, "codec"))
        decoded = loaded.codec.decode(tokens)
        self.assertEqual(decoded, "foo bar foo bar")

    def test_load_legacy_uncompressed_snapshot(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        legacy_path = os.path.join(tmp, "legacy_snapshot.marble")
        legacy_data = {
            "version": 1,
            "n": 1,
            "mode": "grid",
            "size": [3],
            "bounds": [[0.0, 1.0]],
            "formula": None,
            "max_iters": 50,
            "escape_radius": 2.0,
            "sparse_bounds": [],
            "neurons": [
                {
                    "position": [0],
                    "weight": 0.5,
                    "bias": 0.1,
                    "age": 1,
                    "type_name": "default",
                    "tensor": [0.25],
                }
            ],
            "synapses": [],
        }
        with open(legacy_path, "wb") as f:
            pickle.dump(legacy_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        loaded = Brain.load_snapshot(legacy_path)
        self.assertEqual(len(loaded.neurons), 1)
        neuron = next(iter(loaded.neurons.values()))
        tensor_value = neuron.tensor
        if hasattr(tensor_value, "detach") and hasattr(tensor_value, "tolist"):
            tensor_payload = tensor_value.detach().to("cpu").tolist()
        else:
            tensor_payload = list(tensor_value)
        self.assertEqual(tensor_payload, [0.25])
        self.assertEqual(neuron.type_name, "default")

    def test_snapshot_dynamic_brain_without_size(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=None, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        b.add_neuron((0,), tensor=[0.0])
        b.add_neuron((1,), tensor=[1.0], connect_to=(0,), direction="uni")
        snap_path = b.save_snapshot()
        self.assertTrue(os.path.exists(snap_path))
        self.assertIn(os.path.basename(snap_path), os.listdir(tmp))

    def test_snapshot_unique_filenames_when_saving_quickly(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(
            1,
            size=3,
            store_snapshots=True,
            snapshot_path=tmp,
            snapshot_freq=1,
            snapshot_keep=50,
        )
        paths = [b.save_snapshot() for _ in range(3)]
        basenames = [os.path.basename(p) for p in paths]
        self.assertEqual(len(basenames), len(set(basenames)))
        self.assertGreaterEqual(len(os.listdir(tmp)), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
