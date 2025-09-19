import math
import os
import tempfile
import time
import unittest
from array import array

import torch

from marble.marblemain import Brain, UniversalTensorCodec
from marble.reporter import clear_report_group
from marble.snapshot_stream import SnapshotStreamWriter, iterate_snapshot_frames, read_latest_state


class TestBrainSnapshot(unittest.TestCase):
    def _latest_payload(self, path: str) -> dict:
        payload = read_latest_state(path)
        self.assertIsNotNone(payload)
        self.assertIsInstance(payload, dict)
        return payload  # type: ignore[return-value]

    def _normalize_fills(self, raw):
        if raw is None:
            return []
        if isinstance(raw, dict):
            values = raw.get("values", [])
            lengths = raw.get("lengths", [])
            values_list = self._to_list(values)
            lengths_list = [int(v) for v in self._to_list(lengths)]
            normalized = []
            for idx, value in enumerate(values_list):
                length = lengths_list[idx] if idx < len(lengths_list) else 0
                normalized.append((float(value), int(length)))
            extra = raw.get("pairs")
            if extra is not None:
                for entry in self._to_list(extra):
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        normalized.append((float(entry[0]), int(entry[1])))
            return normalized
        normalized = []
        for entry in self._to_list(raw):
            if isinstance(entry, dict):
                value = float(entry.get("value", 0.0))
                length = int(entry.get("length", 0))
                normalized.append((value, length))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                normalized.append((float(entry[0]), int(entry[1])))
        return normalized

    def _to_list(self, value):
        if isinstance(value, list):
            return value
        if isinstance(value, array):
            return value.tolist()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        try:
            return list(value)
        except TypeError:
            return []

    def _decode_ragged(self, raw):
        if raw is None:
            return []
        if isinstance(raw, dict) and "lengths" in raw and "values" in raw:
            lengths = [int(v) for v in self._to_list(raw.get("lengths", []))]
            values = [float(v) for v in self._to_list(raw.get("values", []))]
            count_hint = int(raw.get("count", len(lengths))) if hasattr(raw, "get") else len(lengths)
            result = []
            cursor = 0
            for length in lengths:
                safe_length = max(0, int(length))
                end = cursor + safe_length
                segment = values[cursor:end]
                if len(segment) < safe_length:
                    segment = segment + [0.0] * (safe_length - len(segment))
                result.append(segment[:safe_length])
                cursor = end
            while len(result) < max(count_hint, 0):
                result.append([])
            return result
        if isinstance(raw, list):
            candidates = raw
        else:
            candidates = self._to_list(raw)
        result = []
        for entry in candidates:
            result.append([float(v) for v in self._to_list(entry)])
        return result

    def _tensor_refs(self, neurons_block):
        if "count" in neurons_block:
            count = int(neurons_block.get("count", 0))
        else:
            refs_field = neurons_block.get("tensor_refs")
            derived = len(self._to_list(refs_field)) if refs_field is not None else 0
            if derived <= 0:
                encoding = neurons_block.get("position_encoding")
                if encoding == "linear":
                    derived = len(self._to_list(neurons_block.get("linear_indices", [])))
                else:
                    dims = int(neurons_block.get("position_dims", 0) or 0)
                    if dims > 0:
                        positions = self._to_list(neurons_block.get("positions", []))
                        if positions:
                            derived = len(positions) // max(dims, 1)
            count = derived
        refs_field = neurons_block.get("tensor_refs")
        if refs_field is None:
            return [-1] * count
        refs_list = [int(v) for v in self._to_list(refs_field)]
        if len(refs_list) < count:
            refs_list.extend([-1] * (count - len(refs_list)))
        return refs_list[:count]

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
        frames = list(iterate_snapshot_frames(snap_path))
        self.assertGreaterEqual(len(frames), 1)
        payload = frames[-1].state
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
        self.assertEqual(neurons_block.get("position_encoding"), "linear")
        self.assertIn("linear_indices", neurons_block)
        self.assertIsInstance(neurons_block["linear_indices"], array)
        self.assertEqual(neurons_block["linear_indices"].typecode, "Q")
        self.assertEqual(neurons_block.get("position_dims", 1), 1)
        self.assertEqual(neurons_block.get("position_dtype", "int"), "int")
        self.assertNotIn("positions", neurons_block)
        self.assertIn("weights", neurons_block)
        self.assertIn("biases", neurons_block)
        self.assertIn("ages", neurons_block)
        self.assertIn("type_ids", neurons_block)
        self.assertIsInstance(neurons_block["weights"], array)
        self.assertIsInstance(neurons_block["biases"], array)
        self.assertIsInstance(neurons_block["ages"], array)
        self.assertIsInstance(neurons_block["type_ids"], array)
        tensor_fill_refs = neurons_block.get("tensor_fill_refs")
        self.assertIsNotNone(tensor_fill_refs)
        self.assertIsInstance(tensor_fill_refs, array)
        self.assertIn(
            tensor_fill_refs.typecode,
            ("b", "B", "h", "H", "i", "I", "q", "Q"),
        )
        linear_indices_list = self._to_list(neurons_block.get("linear_indices", []))
        derived_count = neurons_block.get("count", len(linear_indices_list))
        if not derived_count:
            derived_count = len(linear_indices_list)
        self.assertEqual(derived_count, 2)
        self.assertEqual(neurons_block["weights"].tolist(), [2.0, 1.5])
        self.assertEqual(neurons_block["biases"].tolist(), [-0.25, 0.75])
        self.assertEqual(neurons_block["ages"].tolist(), [5, 3])
        tensor_refs = self._tensor_refs(neurons_block)
        fill_refs = tensor_fill_refs.tolist()
        self.assertEqual(sorted(linear_indices_list), [0, 1])
        tensor_pool_raw = payload.get("tensor_pool") or []
        if tensor_pool_raw:
            self.assertIsInstance(tensor_pool_raw, dict)
            self.assertIn("values", tensor_pool_raw)
            self.assertIn("lengths", tensor_pool_raw)
        tensor_pool_fills = payload.get("tensor_pool_fills")
        self.assertIsInstance(tensor_pool_fills, dict)
        normalized_fills = self._normalize_fills(tensor_pool_fills)
        self.assertIn((0.0, 1), normalized_fills)
        self.assertIn((1.0, 1), normalized_fills)
        self.assertTrue(all(ref == -1 for ref in tensor_refs))
        self.assertTrue(all(ref >= 0 for ref in fill_refs))
        syn_block = payload["synapses"]
        self.assertIsInstance(syn_block, dict)
        self.assertIn("source_indices", syn_block)
        self.assertIsInstance(syn_block["source_indices"], array)
        self.assertEqual(syn_block["source_indices"].typecode, "B")
        self.assertIsInstance(syn_block["target_indices"], array)
        self.assertEqual(syn_block["target_indices"].typecode, "B")
        self.assertIn("weights", syn_block)
        self.assertIn("ages", syn_block)
        self.assertIn("type_ids", syn_block)
        self.assertIn("direction_ids", syn_block)
        self.assertIsInstance(syn_block["weights"], array)
        self.assertEqual(syn_block["weights"].typecode, "f")
        self.assertIsInstance(syn_block["ages"], array)
        self.assertEqual(syn_block["ages"].typecode, "B")
        self.assertIsInstance(syn_block["type_ids"], array)
        self.assertEqual(syn_block["type_ids"].typecode, "B")
        self.assertIsInstance(syn_block["direction_ids"], array)
        self.assertEqual(syn_block["direction_ids"].typecode, "B")
        syn_count = syn_block.get("count", len(self._to_list(syn_block.get("source_indices", []))))
        self.assertEqual(syn_count, 1)
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
        loaded_positions = sorted(tuple(neuron.position) for neuron in loaded.neurons.values())
        original_positions = sorted(tuple(neuron.position) for neuron in b.neurons.values())
        self.assertEqual(loaded_positions, original_positions)
        self.assertTrue(hasattr(loaded, "codec"))
        decoded = loaded.codec.decode(tokens)
        self.assertEqual(decoded, "foo bar foo bar")

    def test_snapshot_tensor_pool_and_inline(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        brain = Brain(1, size=4, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        repeated_tensor = [0.125, -0.5]
        inline_tensor = [0.75, -0.125, 0.5]
        brain.add_neuron((0,), tensor=repeated_tensor, type_name="dup")
        brain.add_neuron((1,), tensor=repeated_tensor, connect_to=(0,), direction="uni", type_name="dup")
        brain.add_neuron((2,), tensor=inline_tensor, connect_to=(1,), direction="uni", type_name="unique")
        snap_path = brain.save_snapshot()
        payload = self._latest_payload(snap_path)
        neurons_block = payload["neurons"]
        tensor_refs = self._tensor_refs(neurons_block)
        tensor_fill_refs = neurons_block.get("tensor_fill_refs", array("b"))
        tensor_fill_vals = tensor_fill_refs.tolist() if hasattr(tensor_fill_refs, "tolist") else list(tensor_fill_refs)
        self.assertTrue(all(ref == -1 for ref in tensor_fill_vals))
        self.assertEqual(neurons_block.get("tensor_ref_mode"), "segmented")
        tensor_pool = self._decode_ragged(payload.get("tensor_pool"))
        self.assertEqual(len(tensor_pool), 1)
        pool_values = tensor_pool[0]
        self.assertEqual(len(pool_values), len(repeated_tensor))
        for actual, expected in zip(pool_values, repeated_tensor):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))
        tensor_values = self._decode_ragged(neurons_block.get("tensor_values"))
        self.assertEqual(len(tensor_values), 1)
        for actual, expected in zip(tensor_values[0], inline_tensor):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))
        self.assertEqual(tensor_refs[0], tensor_refs[1])
        self.assertEqual(tensor_refs[0], 0)
        self.assertEqual(tensor_refs[2], 1)
        reloaded = Brain.load_snapshot(snap_path)
        reloaded_tensors = [
            list(neuron.tensor.detach().to("cpu").view(-1).tolist())
            if hasattr(neuron.tensor, "detach")
            else list(neuron.tensor)
            for neuron in sorted(reloaded.neurons.values(), key=lambda n: n.position)
        ]
        for actual, expected in zip(reloaded_tensors[0], repeated_tensor):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))
        for actual, expected in zip(reloaded_tensors[1], repeated_tensor):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))
        for actual, expected in zip(reloaded_tensors[2], inline_tensor):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))

    def test_snapshot_omits_default_fields(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=3, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        b.add_neuron((0,), tensor=[0.0])
        b.add_neuron((1,), tensor=[1.0], connect_to=(0,), direction="uni")
        snap_path = b.save_snapshot()
        payload = self._latest_payload(snap_path)
        self.assertTrue(payload.get("tensor_pool") in (None, {}))
        self.assertIn("tensor_pool_fills", payload)
        neurons_block = payload["neurons"]
        self.assertEqual(neurons_block.get("position_encoding"), "linear")
        self.assertIn("linear_indices", neurons_block)
        self.assertNotIn("positions", neurons_block)
        self.assertEqual(sorted(neurons_block["linear_indices"].tolist()), [0, 1])
        self.assertIn("tensor_fill_refs", neurons_block)
        fill_refs = neurons_block["tensor_fill_refs"].tolist()
        self.assertTrue(all(ref >= 0 for ref in fill_refs))
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

    def test_snapshot_tensor_pool_deduplicates_neuron_tensors(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=3, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        shared_tensor = [0.5, 0.25]
        b.add_neuron((0,), tensor=shared_tensor)
        b.add_neuron((1,), tensor=shared_tensor, connect_to=(0,), direction="uni")
        b.add_neuron((2,), tensor=[1.25], connect_to=(1,), direction="uni")
        snap_path = b.save_snapshot()
        payload = self._latest_payload(snap_path)
        tensor_pool = self._decode_ragged(payload.get("tensor_pool"))
        self.assertEqual(len(tensor_pool), 1)
        pool_values = tensor_pool[0]
        for actual, expected in zip(pool_values, shared_tensor):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))
        tensor_refs = self._tensor_refs(payload["neurons"])
        fill_refs = payload["neurons"].get("tensor_fill_refs")
        self.assertIsNotNone(fill_refs)
        fill_refs_list = fill_refs.tolist()
        self.assertEqual(tensor_refs[0], tensor_refs[1])
        self.assertEqual(fill_refs_list[0], -1)
        self.assertEqual(fill_refs_list[1], -1)
        self.assertEqual(tensor_refs[2], -1)
        self.assertGreaterEqual(fill_refs_list[2], 0)
        reloaded = Brain.load_snapshot(snap_path)
        self.assertEqual(len(reloaded.neurons), 3)
        tensors = []
        for neuron in reloaded.neurons.values():
            value = neuron.tensor
            if hasattr(value, "detach") and hasattr(value, "tolist"):
                tensors.append(value.detach().to("cpu").tolist())
            else:
                tensors.append(list(value))
        self.assertIn([0.5, 0.25], tensors)
        self.assertIn([1.25], tensors)

    def test_snapshot_records_fill_metadata_for_uniform_tensors(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=4, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        huge_constant = [3.75] * 4096
        normal_tensor = [0.1, 0.2, 0.3]
        b.add_neuron((0,), tensor=huge_constant)
        b.add_neuron((1,), tensor=normal_tensor, connect_to=(0,), direction="uni")
        snap_path = b.save_snapshot()
        payload = self._latest_payload(snap_path)
        tensor_pool = self._decode_ragged(payload.get("tensor_pool"))
        tensor_pool_fills = payload.get("tensor_pool_fills")
        neurons_block = payload["neurons"]
        tensor_values = self._decode_ragged(neurons_block.get("tensor_values"))
        self.assertFalse(tensor_pool)
        self.assertIsInstance(tensor_pool_fills, dict)
        self.assertEqual(len(tensor_values), 1)
        self.assertEqual(len(tensor_values[0]), len(normal_tensor))
        for actual, expected in zip(tensor_values[0], normal_tensor):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))
        normalized_fills = self._normalize_fills(tensor_pool_fills)
        self.assertTrue(normalized_fills, "expected fill metadata")
        fill_entry = None
        for value, length in normalized_fills:
            if int(length) == 4096:
                fill_entry = (value, length)
                break
        self.assertIsNotNone(fill_entry)
        fill_value, fill_length = fill_entry
        self.assertAlmostEqual(float(fill_value), 3.75)
        self.assertEqual(int(fill_length), 4096)
        tensor_refs = self._tensor_refs(neurons_block)
        fill_refs = neurons_block["tensor_fill_refs"].tolist()
        self.assertEqual(neurons_block.get("tensor_ref_mode"), "segmented")
        self.assertEqual(tensor_refs[0], -1)
        self.assertEqual(fill_refs[0], 0)
        self.assertEqual(fill_refs[1], -1)
        self.assertEqual(tensor_refs[1], 0)
        reloaded = Brain.load_snapshot(snap_path)
        self.assertEqual(len(reloaded.neurons), 2)
        tensors = []
        for neuron in reloaded.neurons.values():
            value = neuron.tensor
            if hasattr(value, "detach") and hasattr(value, "tolist"):
                tensors.append(value.detach().to("cpu").tolist())
            else:
                tensors.append(list(value))
        self.assertTrue(any(tensor == huge_constant for tensor in tensors))
        self.assertTrue(
            any(
                len(tensor) == len(normal_tensor)
                and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(tensor, normal_tensor))
                for tensor in tensors
            )
        )

    def test_snapshot_uses_signed_byte_arrays_when_needed(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=5, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        b.add_neuron((0,), tensor=[0.0], type_name="kind_a", age=2)
        b.add_neuron((1,), tensor=[1.0], connect_to=(0,), direction="bi", age=3)
        first_syn = b.synapses[-1]
        first_syn.type_name = "syn_kind"
        first_syn.age = 5
        b.add_neuron((2,), tensor=[2.0], type_name="kind_c", connect_to=(1,), age=7)
        second_syn = b.synapses[-1]
        second_syn.age = 1
        snap_path = b.save_snapshot()
        payload = self._latest_payload(snap_path)
        neurons_block = payload["neurons"]
        syn_block = payload["synapses"]
        print(
            "narrow typecodes:",
            neurons_block["type_ids"].typecode if "type_ids" in neurons_block else None,
            syn_block["type_ids"].typecode if "type_ids" in syn_block else None,
        )
        self.assertIn("type_ids", neurons_block)
        self.assertEqual(neurons_block["type_ids"].typecode, "b")
        tensor_refs_field = neurons_block.get("tensor_refs")
        if tensor_refs_field is not None:
            self.assertIn(
                tensor_refs_field.typecode,
                ("b", "B", "h", "H", "i", "I", "q", "Q"),
            )
        self.assertIn("tensor_fill_refs", neurons_block)
        self.assertIn(
            neurons_block["tensor_fill_refs"].typecode,
            ("b", "B", "h", "H", "i", "I", "q", "Q"),
        )
        self.assertIn("source_indices", syn_block)
        self.assertEqual(syn_block["source_indices"].typecode, "B")
        self.assertIn("target_indices", syn_block)
        self.assertEqual(syn_block["target_indices"].typecode, "B")
        self.assertIn("type_ids", syn_block)
        self.assertEqual(syn_block["type_ids"].typecode, "b")
        reloaded = Brain.load_snapshot(snap_path)
        self.assertEqual(len(reloaded.neurons), len(b.neurons))
        self.assertEqual(len(reloaded.synapses), len(b.synapses))
        reloaded_types = [n.type_name for n in reloaded.neurons.values()]
        self.assertIn("kind_a", reloaded_types)
        self.assertIn("kind_c", reloaded_types)

    def test_snapshot_promotes_integer_arrays_to_uint16(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        neuron_count = 300
        b = Brain(1, size=neuron_count + 1, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        prev_pos = None
        for idx in range(neuron_count):
            pos = (idx,)
            tensor_payload = [float(idx)]
            if prev_pos is None:
                b.add_neuron(pos, tensor=tensor_payload, type_name="bulk", age=idx % 3)
            else:
                b.add_neuron(pos, tensor=tensor_payload, connect_to=prev_pos, age=idx % 5)
            prev_pos = pos
        snap_path = b.save_snapshot()
        payload = self._latest_payload(snap_path)
        neurons_block = payload["neurons"]
        syn_block = payload["synapses"]
        tensor_refs_field = neurons_block.get("tensor_refs")
        tensor_ref_typecode = None if tensor_refs_field is None else tensor_refs_field.typecode
        print(
            "large snapshot typecodes:",
            tensor_ref_typecode,
            syn_block["source_indices"].typecode,
            syn_block["target_indices"].typecode,
        )
        if tensor_refs_field is not None:
            self.assertEqual(tensor_refs_field.typecode, "b")
        else:
            self.assertTrue(all(ref == -1 for ref in self._tensor_refs(neurons_block)))
        self.assertIn("tensor_fill_refs", neurons_block)
        self.assertEqual(neurons_block["tensor_fill_refs"].typecode, "H")
        self.assertEqual(syn_block["source_indices"].typecode, "H")
        self.assertEqual(syn_block["target_indices"].typecode, "H")
        reloaded = Brain.load_snapshot(snap_path)
        self.assertEqual(len(reloaded.neurons), neuron_count)
        self.assertEqual(len(reloaded.synapses), neuron_count - 1)

    def test_load_legacy_snapshot_without_tensor_pool(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "legacy_pool_snapshot.marble")
        legacy_payload = {
            "version": 1,
            "n": 1,
            "mode": "grid",
            "size": [3],
            "bounds": [[0.0, 1.0]],
            "formula": None,
            "max_iters": 50,
            "escape_radius": 2.0,
            "sparse_bounds": [],
            "layout": "columnar",
            "neurons": {
                "count": 2,
                "position_dims": 1,
                "position_dtype": "int",
                "positions": array("i", [0, 1]),
                "tensor_refs": array("i", [0, 0]),
                "tensor_values": [[0.125], [0.125]],
            },
            "synapses": {
                "count": 0,
                "source_indices": array("i"),
                "target_indices": array("i"),
            },
        }
        writer = SnapshotStreamWriter(path)
        try:
            writer.append_state(legacy_payload, reason="legacy-no-pool")
        finally:
            writer.close()
        loaded = Brain.load_snapshot(path)
        self.assertEqual(len(loaded.neurons), 2)
        for neuron in loaded.neurons.values():
            value = neuron.tensor
            if hasattr(value, "detach") and hasattr(value, "tolist"):
                payload_vals = value.detach().to("cpu").tolist()
            else:
                payload_vals = list(value)
            self.assertEqual(payload_vals, [0.125])

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
        writer = SnapshotStreamWriter(path)
        try:
            writer.append_state(snapshot, reason="legacy-index")
        finally:
            writer.close()
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
        new_payload = self._latest_payload(saved_roundtrip)
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
        writer = SnapshotStreamWriter(path)
        try:
            writer.append_state(legacy_data, reason="legacy-codec")
        finally:
            writer.close()
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
        writer = SnapshotStreamWriter(legacy_path)
        try:
            writer.append_state(legacy_data, reason="legacy-uncompressed")
        finally:
            writer.close()
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
        self.assertEqual(len(set(paths)), 1)
        stream_path = paths[0]
        self.assertTrue(os.path.exists(stream_path))
        frames = list(iterate_snapshot_frames(stream_path))
        self.assertGreaterEqual(len(frames), 3)
        self.assertEqual(len(os.listdir(tmp)), 1)

    def test_snapshot_cuda_fallback_does_not_duplicate(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=3, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        existing_path = os.path.join(tmp, "snapshot_cuda_seed.marble")
        with open(existing_path, "wb") as fh:
            fh.write(b"cuda snapshot placeholder")
        b._last_snapshot_meta = {
            "path": existing_path,
            "time": time.time(),
            "device": "cuda",
        }
        b._snapshot_fallback_window = 10.0
        b._allow_cpu_snapshot_when_cuda = True
        b._force_snapshot_device = "cpu"
        before_files = set(os.listdir(tmp))
        returned = b.save_snapshot()
        after_files = set(os.listdir(tmp))
        self.assertEqual(returned, existing_path)
        self.assertEqual(before_files, after_files)

    def test_snapshot_rejects_cpu_override_when_cuda_available(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=3, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        b._force_snapshot_device = "cpu"
        if torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                b.save_snapshot()
        else:
            b._allow_cpu_snapshot_when_cuda = True
            path = b.save_snapshot()
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
