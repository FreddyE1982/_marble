import gzip
import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

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
        with gzip.open(snap_path, "rb") as payload_file:
            payload = pickle.load(payload_file)
        print("snapshot keys:", sorted(payload.keys()))
        self.assertIn("codec_state", payload)
        self.assertNotIn("codec_vocab", payload)
        self.assertGreater(len(payload.get("neurons", [])), 0)
        first_neuron = payload["neurons"][0]
        self.assertIn("tensor_blob", first_neuron)
        blob = first_neuron["tensor_blob"]
        print("tensor_blob metadata:", blob)
        self.assertIsInstance(blob.get("data"), bytes)
        self.assertEqual(blob.get("shape"), [1])
        print("snapshot path:", snap_path)
        loaded = Brain.load_snapshot(snap_path)
        print("loaded brain neurons:", len(loaded.neurons))
        self.assertEqual(len(loaded.neurons), len(b.neurons))
        self.assertTrue(hasattr(loaded, "codec"))
        decoded = loaded.codec.decode(tokens)
        self.assertEqual(decoded, "foo bar foo bar")

    def test_load_snapshot_with_legacy_codec_vocab(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "legacy_codec_vocab.marble")
        codec = UniversalTensorCodec()
        legacy_data = {
            "version": 1,
            "n": 1,
            "mode": "grid",
            "size": [1],
            "bounds": [[0.0, 1.0]],
            "formula": None,
            "max_iters": 50,
            "escape_radius": 2.0,
            "sparse_bounds": [],
            "neurons": [
                {
                    "position": [0],
                    "weight": 1.0,
                    "bias": 0.0,
                    "age": 0,
                    "type_name": "default",
                    "tensor": [0.0],
                }
            ],
            "synapses": [],
            "codec_vocab": codec.dump_vocab(),
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(legacy_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        loaded = Brain.load_snapshot(path)
        print("loaded legacy codec via vocab", hasattr(loaded, "codec"))
        self.assertTrue(hasattr(loaded, "codec"))
        fresh_tokens = loaded.codec.encode("legacy payload")
        decoded = loaded.codec.decode(fresh_tokens)
        self.assertEqual(decoded, "legacy payload")

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

    def test_tensor_blob_round_trip_multi_dimensional(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(2, size=None, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        base_tensor = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        b.add_neuron((0, 0), tensor=base_tensor)
        snap_path = b.save_snapshot()
        with gzip.open(snap_path, "rb") as payload_file:
            payload = pickle.load(payload_file)
        self.assertEqual(len(payload.get("neurons", [])), 1)
        neuron_payload = payload["neurons"][0]
        self.assertIn("tensor_blob", neuron_payload)
        blob = neuron_payload["tensor_blob"]
        print("blob schema:", {k: type(v).__name__ for k, v in blob.items()})
        expected_shape = list(base_tensor.shape)
        self.assertEqual(blob.get("shape"), expected_shape)
        self.assertIsInstance(blob.get("data"), bytes)
        self.assertEqual(len(blob.get("data", b"")), base_tensor.numel() * base_tensor.element_size())
        adjusted_payload = payload
        for neuron in adjusted_payload.get("neurons", []):
            neuron.pop("tensor", None)
        blob_only_path = os.path.join(tmp, "blob_only_snapshot.marble")
        with gzip.open(blob_only_path, "wb") as f:
            pickle.dump(adjusted_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        restored = Brain.load_snapshot(blob_only_path)
        restored_neuron = restored.get_neuron((0, 0))
        self.assertIsNotNone(restored_neuron)
        assert restored_neuron is not None
        restored_tensor = restored_neuron.tensor
        expected_cpu = base_tensor.detach().cpu()
        if hasattr(restored_tensor, "detach") and hasattr(restored_tensor, "to"):
            round_tripped = restored_tensor.detach().to("cpu")
            self.assertEqual(tuple(round_tripped.shape), tuple(expected_cpu.shape))
            self.assertTrue(torch.allclose(round_tripped, expected_cpu))
        else:
            round_tripped_np = np.asarray(restored_tensor)
            self.assertEqual(round_tripped_np.shape, expected_cpu.numpy().shape)
            self.assertTrue(np.allclose(round_tripped_np, expected_cpu.numpy()))

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
