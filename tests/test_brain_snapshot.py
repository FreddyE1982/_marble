import gzip
import os
import pickle
import tempfile
import unittest
from array import array

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
        b.add_neuron((1,), tensor=[0.0], type_name="alpha", weight=2.0, bias=-0.25, age=5)
        b.add_neuron(
            (0,),
            tensor=[1.0],
            connect_to=(1,),
            direction="uni",
            type_name="beta",
            weight=1.5,
            bias=0.75,
            age=3,
        )
        synapse = b.synapses[-1]
        synapse.weight = 0.5
        synapse.age = 4
        synapse.type_name = "gamma"
        synapse.direction = "bi"
        snap_path = b.save_snapshot()
        with open(snap_path, "rb") as saved:
            header = saved.read(2)
        self.assertEqual(header, b"\x1f\x8b")
        with gzip.open(snap_path, "rb") as payload_file:
            payload = pickle.load(payload_file)
        print("snapshot keys:", sorted(payload.keys()))
        print("snapshot layout:", payload.get("layout"))
        self.assertIn("codec_state", payload)
        self.assertNotIn("codec_vocab", payload)
        self.assertIn("string_table", payload)
        table = payload["string_table"]
        self.assertIsInstance(table, list)
        self.assertTrue(all(isinstance(entry, str) for entry in table))
        self.assertEqual(payload.get("layout"), "columnar")
        neurons_block = payload["neurons"]
        self.assertIsInstance(neurons_block, dict)
        self.assertIn("positions", neurons_block)
        self.assertIsInstance(neurons_block["positions"], array)
        self.assertEqual(neurons_block["positions"].typecode, "i")
        self.assertEqual(neurons_block["position_dims"], 1)
        self.assertEqual(neurons_block["position_dtype"], "int")
        self.assertIn("weights", neurons_block)
        self.assertIn("biases", neurons_block)
        self.assertIn("ages", neurons_block)
        self.assertIn("type_ids", neurons_block)
        self.assertIsInstance(neurons_block["weights"], array)
        self.assertIsInstance(neurons_block["biases"], array)
        self.assertIsInstance(neurons_block["ages"], array)
        self.assertIsInstance(neurons_block["type_ids"], array)
        self.assertIsInstance(neurons_block["tensor_refs"], array)
        self.assertEqual(neurons_block["tensor_refs"].typecode, "i")
        self.assertEqual(neurons_block["count"], 2)
        self.assertEqual(neurons_block["weights"].tolist(), [2.0, 1.5])
        self.assertEqual(neurons_block["biases"].tolist(), [-0.25, 0.75])
        self.assertEqual(neurons_block["ages"].tolist(), [5, 3])
        tensor_refs = neurons_block["tensor_refs"].tolist()
        tensor_values = neurons_block["tensor_values"]
        self.assertIsInstance(tensor_values, list)
        self.assertTrue(tensor_values)
        self.assertTrue(all(0 <= ref < len(tensor_values) for ref in tensor_refs))
        syn_block = payload["synapses"]
        self.assertIsInstance(syn_block, dict)
        self.assertIn("source_indices", syn_block)
        self.assertIsInstance(syn_block["source_indices"], array)
        self.assertEqual(syn_block["source_indices"].typecode, "i")
        self.assertIsInstance(syn_block["target_indices"], array)
        self.assertEqual(syn_block["target_indices"].typecode, "i")
        self.assertIn("weights", syn_block)
        self.assertIn("ages", syn_block)
        self.assertIn("type_ids", syn_block)
        self.assertIn("direction_ids", syn_block)
        self.assertIsInstance(syn_block["weights"], array)
        self.assertEqual(syn_block["weights"].typecode, "f")
        self.assertIsInstance(syn_block["ages"], array)
        self.assertEqual(syn_block["ages"].typecode, "i")
        self.assertIsInstance(syn_block["type_ids"], array)
        self.assertEqual(syn_block["type_ids"].typecode, "i")
        self.assertIsInstance(syn_block["direction_ids"], array)
        self.assertEqual(syn_block["direction_ids"].typecode, "i")
        self.assertEqual(syn_block["count"], 1)
        self.assertEqual(syn_block["weights"].tolist(), [0.5])
        self.assertEqual(syn_block["ages"].tolist(), [4])
        idx_pairs = set(
            zip(
                syn_block["source_indices"].tolist(),
                syn_block["target_indices"].tolist(),
            )
        )
        self.assertIn((1, 0), idx_pairs)
        for dir_idx in syn_block["direction_ids"].tolist():
            self.assertGreaterEqual(dir_idx, 0)
            self.assertEqual(table[dir_idx], "bi")
        neuron_type_indexes = neurons_block["type_ids"].tolist()
        self.assertTrue(all(isinstance(idx, int) for idx in neuron_type_indexes))
        resolved_types = [table[idx] if idx >= 0 else None for idx in neuron_type_indexes]
        self.assertIn("alpha", resolved_types)
        self.assertIn("beta", resolved_types)
        print("snapshot path:", snap_path)
        loaded = Brain.load_snapshot(snap_path)
        print("loaded brain neurons:", len(loaded.neurons))
        self.assertEqual(len(loaded.neurons), len(b.neurons))
        self.assertEqual(len(loaded.synapses), len(b.synapses))
        self.assertTrue(hasattr(loaded, "codec"))
        decoded = loaded.codec.decode(tokens)
        self.assertEqual(decoded, "foo bar foo bar")

    def test_snapshot_omits_default_fields(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=3, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        b.add_neuron((0,), tensor=[0.0])
        b.add_neuron((1,), tensor=[1.0], connect_to=(0,), direction="uni")
        snap_path = b.save_snapshot()
        with gzip.open(snap_path, "rb") as payload_file:
            payload = pickle.load(payload_file)
        neurons_block = payload["neurons"]
        self.assertNotIn("weights", neurons_block)
        self.assertNotIn("biases", neurons_block)
        self.assertNotIn("ages", neurons_block)
        self.assertNotIn("type_ids", neurons_block)
        syn_block = payload["synapses"]
        self.assertNotIn("weights", syn_block)
        self.assertNotIn("ages", syn_block)
        self.assertNotIn("type_ids", syn_block)
        self.assertNotIn("direction_ids", syn_block)
        loaded = Brain.load_snapshot(snap_path)
        self.assertEqual(len(loaded.neurons), 2)
        first = loaded.get_neuron((0,))
        second = loaded.get_neuron((1,))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.weight, 1.0)
        self.assertEqual(second.bias, 0.0)
        self.assertEqual(first.age, 0)
        self.assertEqual(second.age, 0)
        self.assertEqual(first.type_name, None)
        self.assertEqual(second.type_name, None)
        self.assertEqual(len(loaded.synapses), 1)
        syn = loaded.synapses[0]
        self.assertEqual(syn.weight, 1.0)
        self.assertEqual(syn.age, 0)
        self.assertEqual(syn.type_name, None)
        self.assertEqual(syn.direction, "uni")

    def test_load_snapshot_with_index_synapses(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "index_snapshot.marble")
        snapshot = {
            "version": 1,
            "n": 1,
            "mode": "grid",
            "size": [2],
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
                },
                {
                    "position": [1],
                    "weight": 1.0,
                    "bias": 0.0,
                    "age": 0,
                    "type_name": "default",
                    "tensor": [0.5],
                },
            ],
            "synapses": [
                {
                    "source_idx": 0,
                    "target_idx": 1,
                    "direction": "uni",
                    "age": 0,
                    "type_name": "default",
                    "weight": 0.75,
                }
            ],
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        loaded = Brain.load_snapshot(path)
        self.assertEqual(len(loaded.neurons), 2)
        self.assertEqual(len(loaded.synapses), 1)
        syn = loaded.synapses[0]
        self.assertEqual(syn.weight, 0.75)
        self.assertEqual(syn.direction, "uni")
        src_pos = getattr(syn.source, "position")
        dst_pos = getattr(syn.target, "position")
        self.assertEqual(tuple(src_pos), (0,))
        self.assertEqual(tuple(dst_pos), (1,))
        roundtrip_path = os.path.join(tmp, "roundtrip_snapshot.marble")
        saved_roundtrip = loaded.save_snapshot(roundtrip_path)
        with gzip.open(saved_roundtrip, "rb") as saved_payload:
            new_payload = pickle.load(saved_payload)
        print("roundtrip snapshot layout:", new_payload.get("layout"))
        self.assertEqual(new_payload.get("layout"), "columnar")
        reloaded = Brain.load_snapshot(saved_roundtrip)
        self.assertEqual(len(reloaded.neurons), len(loaded.neurons))
        self.assertEqual(len(reloaded.synapses), len(loaded.synapses))

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
            "synapses": [
                {
                    "source": [0],
                    "target": [0],
                    "direction": "uni",
                    "age": 2,
                    "type_name": "default",
                    "weight": 1.25,
                }
            ],
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
        self.assertEqual(len(loaded.synapses), 1)
        legacy_syn = loaded.synapses[0]
        self.assertEqual(legacy_syn.age, 2)
        self.assertEqual(legacy_syn.weight, 1.25)

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
