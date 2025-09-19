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
        if isinstance(value, dict):
            encoding = value.get("enc") or value.get("encoding") or value.get("mode")
            if isinstance(encoding, str):
                enc_lower = encoding.lower()
                if enc_lower.startswith("sparse"):
                    return []
                if enc_lower in {"range", "arange"}:
                    try:
                        count_val = int(value.get("count", 0))
                    except Exception:
                        count_val = 0
                    start_raw = value.get("start", 0)
                    step_raw = value.get("step", 1)
                    try:
                        start_val = float(start_raw)
                        step_val = float(step_raw)
                        generated = [start_val + step_val * idx for idx in range(max(0, count_val))]
                        if all(abs(val - round(val)) < 1e-9 for val in generated):
                            return [int(round(val)) for val in generated]
                        return generated
                    except Exception:
                        return []
                if enc_lower in {"const", "constant"}:
                    try:
                        count_val = int(value.get("count", 0))
                    except Exception:
                        count_val = 0
                    fill_raw = value.get("value", value.get("default", 0))
                    try:
                        fill_val = float(fill_raw)
                    except Exception:
                        fill_val = 0.0
                    if abs(fill_val - round(fill_val)) < 1e-9:
                        fill_val = int(round(fill_val))
                    return [fill_val] * max(0, count_val)
                if enc_lower.startswith("delta"):
                    try:
                        count_val = int(value.get("count", 0))
                    except Exception:
                        count_val = 0
                    deltas = self._to_list(value.get("deltas", []))
                    start_raw = value.get("start", 0)
                    seq: list = [start_raw]
                    current = start_raw
                    for delta_entry in deltas:
                        current = current + delta_entry
                        seq.append(current)
                    if count_val <= 0:
                        count_val = len(seq)
                    if len(seq) < count_val:
                        fill_val = seq[-1] if seq else start_raw
                        seq.extend([fill_val] * (count_val - len(seq)))
                    if len(seq) > count_val:
                        seq = seq[:count_val]
                    return seq
                if enc_lower == "rle":
                    try:
                        count_val = int(value.get("count", 0))
                    except Exception:
                        count_val = 0
                    values_list = self._to_list(value.get("values", []))
                    counts_list = [int(v) for v in self._to_list(value.get("counts", []))]
                    run_total = min(len(values_list), len(counts_list))
                    fallback = values_list[0] if values_list else 0
                    seq: list = []
                    for idx in range(run_total):
                        repeat = max(0, counts_list[idx])
                        if repeat <= 0:
                            continue
                        run_value = values_list[idx] if idx < len(values_list) else fallback
                        seq.extend([run_value] * repeat)
                    if count_val > 0 and len(seq) < count_val:
                        seq.extend([fallback] * (count_val - len(seq)))
                    return seq
                if enc_lower == "bits":
                    try:
                        count_val = int(value.get("count", 0))
                    except Exception:
                        count_val = 0
                    payload = value.get("data", value.get("payload", value.get("bits", b"")))
                    if isinstance(payload, memoryview):
                        payload_bytes = payload.tobytes()
                    elif isinstance(payload, (bytes, bytearray)):
                        payload_bytes = bytes(payload)
                    else:
                        try:
                            payload_bytes = bytes(payload) if payload is not None else b""
                        except Exception:
                            payload_bytes = b""
                    if count_val <= 0:
                        count_val = len(payload_bytes) * 8
                    result_bits = []
                    for idx in range(max(0, count_val)):
                        byte_index = idx // 8
                        bit_index = idx % 8
                        bit_value = 0
                        if byte_index < len(payload_bytes):
                            bit_value = (payload_bytes[byte_index] >> bit_index) & 1
                        result_bits.append(int(bit_value))
                    return result_bits
                if enc_lower == "bitpack":
                    try:
                        count_val = int(value.get("count", 0))
                    except Exception:
                        count_val = 0
                    bits_per = value.get("bits_per", value.get("bits", 0))
                    try:
                        bits_per_int = int(bits_per)
                    except Exception:
                        bits_per_int = 0
                    payload = value.get("data", value.get("payload", b""))
                    if isinstance(payload, memoryview):
                        payload_bytes = payload.tobytes()
                    elif isinstance(payload, (bytes, bytearray)):
                        payload_bytes = bytes(payload)
                    else:
                        try:
                            payload_bytes = bytes(payload) if payload is not None else b""
                        except Exception:
                            payload_bytes = b""
                    if bits_per_int <= 0:
                        return [0] * max(0, count_val)
                    mask = (1 << bits_per_int) - 1
                    total_bits = len(payload_bytes) * 8
                    target = max(0, count_val)
                    decoded = []
                    for idx in range(target):
                        bit_pos = idx * bits_per_int
                        if bit_pos + bits_per_int > total_bits:
                            decoded.append(0)
                            continue
                        byte_pos = bit_pos // 8
                        bit_offset = bit_pos % 8
                        value_chunk = payload_bytes[byte_pos] >> bit_offset
                        bits_used = 8 - bit_offset
                        if bits_used < bits_per_int and byte_pos + 1 < len(payload_bytes):
                            value_chunk |= payload_bytes[byte_pos + 1] << bits_used
                        decoded.append(int(value_chunk & mask))
                    return decoded
                if enc_lower == "palette":
                    palette_values = self._to_list(value.get("values", []))
                    indices_list = self._to_list(value.get("indices", []))
                    if not palette_values:
                        return []
                    try:
                        count_val = int(value.get("count", 0))
                    except Exception:
                        count_val = 0
                    target = max(len(indices_list), max(0, count_val))
                    if target <= 0:
                        return []
                    resolved = []
                    fallback = palette_values[0]
                    for idx in range(target):
                        if idx < len(indices_list):
                            source = indices_list[idx]
                        else:
                            source = 0
                        try:
                            palette_idx = int(source)
                        except Exception:
                            palette_idx = 0
                        if 0 <= palette_idx < len(palette_values):
                            resolved.append(palette_values[palette_idx])
                        else:
                            resolved.append(fallback)
                    return resolved
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        try:
            return list(value)
        except TypeError:
            return []

    def _decode_numeric_field(self, raw, count, default):
        if raw is None:
            return [default] * count if count else []
        if isinstance(raw, dict):
            encoding = raw.get("enc") or raw.get("encoding") or raw.get("mode")
            if isinstance(encoding, str):
                enc_lower = encoding.lower()
                if enc_lower in {"range", "arange"}:
                    try:
                        count_val = int(raw.get("count", count or 0))
                    except Exception:
                        count_val = count or 0
                    start_raw = raw.get("start", default)
                    step_raw = raw.get("step", 1)
                    seq = []
                    for idx in range(max(0, count_val)):
                        value = start_raw + step_raw * idx
                        try:
                            seq.append(float(value))
                        except Exception:
                            seq.append(default)
                    if count and len(seq) < count:
                        seq.extend([default] * (count - len(seq)))
                    if isinstance(default, int):
                        seq = [int(round(val)) for val in seq]
                    return seq
                if enc_lower in {"const", "constant"}:
                    try:
                        count_val = int(raw.get("count", count or 0))
                    except Exception:
                        count_val = count or 0
                    fill_value = raw.get("value", raw.get("default", default))
                    try:
                        fill_numeric = float(fill_value)
                    except Exception:
                        fill_numeric = default
                    seq = [fill_numeric] * max(0, count_val)
                    if count and len(seq) < count:
                        seq.extend([default] * (count - len(seq)))
                    if isinstance(default, int):
                        seq = [int(round(val)) for val in seq]
                    return seq
                if enc_lower.startswith("delta"):
                    try:
                        count_val = int(raw.get("count", count or 0))
                    except Exception:
                        count_val = count or 0
                    deltas = self._to_list(raw.get("deltas", []))
                    start_value = raw.get("start", default)
                    if isinstance(default, int):
                        current = int(start_value)
                        deltas_cast = [int(v) for v in deltas]
                    else:
                        current = float(start_value)
                        deltas_cast = [float(v) for v in deltas]
                    generated = [current]
                    for delta_entry in deltas_cast:
                        current = current + delta_entry
                        generated.append(current)
                    if count_val <= 0:
                        count_val = len(generated)
                    if len(generated) < count_val:
                        fill_val = generated[-1] if generated else start_value
                        generated.extend([fill_val] * (count_val - len(generated)))
                    if len(generated) > count_val:
                        generated = generated[:count_val]
                    if count and len(generated) < count:
                        fill_val = generated[-1] if generated else start_value
                        generated.extend([fill_val] * (count - len(generated)))
                    if count and len(generated) > count:
                        generated = generated[:count]
                    if isinstance(default, int):
                        return [int(v) for v in generated]
                    return [float(v) for v in generated]
                if enc_lower == "rle":
                    try:
                        count_val = int(raw.get("count", count or 0))
                    except Exception:
                        count_val = count or 0
                    values_list = self._to_list(raw.get("values", []))
                    counts_list = [int(v) for v in self._to_list(raw.get("counts", []))]
                    run_total = min(len(values_list), len(counts_list))
                    target_len = count or count_val
                    if isinstance(default, int):
                        fallback = int(default)
                        seq: list[int] = []
                        for idx in range(run_total):
                            repeat = max(0, counts_list[idx])
                            if repeat <= 0:
                                continue
                            value = values_list[idx] if idx < len(values_list) else fallback
                            try:
                                numeric = int(value)
                            except Exception:
                                numeric = fallback
                            seq.extend([numeric] * repeat)
                        if target_len and len(seq) < target_len:
                            seq.extend([fallback] * (target_len - len(seq)))
                        if target_len and len(seq) > target_len:
                            seq = seq[:target_len]
                        return seq
                    fallback_float = float(default)
                    seq_f: list[float] = []
                    for idx in range(run_total):
                        repeat = max(0, counts_list[idx])
                        if repeat <= 0:
                            continue
                        value = values_list[idx] if idx < len(values_list) else fallback_float
                        try:
                            numeric = float(value)
                        except Exception:
                            numeric = fallback_float
                        seq_f.extend([numeric] * repeat)
                    if target_len and len(seq_f) < target_len:
                        seq_f.extend([fallback_float] * (target_len - len(seq_f)))
                    if target_len and len(seq_f) > target_len:
                        seq_f = seq_f[:target_len]
                    return seq_f
                if enc_lower == "bits":
                    try:
                        count_val = int(raw.get("count", count or 0))
                    except Exception:
                        count_val = count or 0
                    payload = raw.get("data", raw.get("payload", raw.get("bits", b"")))
                    if isinstance(payload, memoryview):
                        payload_bytes = payload.tobytes()
                    elif isinstance(payload, (bytes, bytearray)):
                        payload_bytes = bytes(payload)
                    else:
                        try:
                            payload_bytes = bytes(payload) if payload is not None else b""
                        except Exception:
                            payload_bytes = b""
                    if count_val <= 0:
                        count_val = len(payload_bytes) * 8
                    target_len = count if count and count > count_val else count_val
                    result_bits = []
                    for idx in range(max(target_len or 0, 0)):
                        byte_index = idx // 8
                        bit_index = idx % 8
                        bit_value = 0
                        if byte_index < len(payload_bytes):
                            bit_value = (payload_bytes[byte_index] >> bit_index) & 1
                        result_bits.append(int(bit_value))
                    if target_len and len(result_bits) < target_len:
                        result_bits.extend([int(default)] * (target_len - len(result_bits)))
                    if isinstance(default, float) and not isinstance(default, int):
                        return [float(v) for v in result_bits[:target_len or len(result_bits)]]
                    return result_bits[:target_len or len(result_bits)]
                if enc_lower == "bitpack":
                    try:
                        count_val = int(raw.get("count", count or 0))
                    except Exception:
                        count_val = count or 0
                    bits_per = raw.get("bits_per", raw.get("bits", 0))
                    try:
                        bits_per_int = int(bits_per)
                    except Exception:
                        bits_per_int = 0
                    payload = raw.get("data", raw.get("payload", b""))
                    if isinstance(payload, memoryview):
                        payload_bytes = payload.tobytes()
                    elif isinstance(payload, (bytes, bytearray)):
                        payload_bytes = bytes(payload)
                    else:
                        try:
                            payload_bytes = bytes(payload) if payload is not None else b""
                        except Exception:
                            payload_bytes = b""
                    if bits_per_int <= 0:
                        target_len = max(count or 0, max(0, count_val))
                        return [default] * target_len
                    mask = (1 << bits_per_int) - 1
                    total_bits = len(payload_bytes) * 8
                    target_len = max(count or 0, max(0, count_val))
                    decoded = []
                    for idx in range(target_len):
                        bit_pos = idx * bits_per_int
                        if bit_pos + bits_per_int > total_bits:
                            decoded.append(default)
                            continue
                        byte_pos = bit_pos // 8
                        bit_offset = bit_pos % 8
                        value_chunk = payload_bytes[byte_pos] >> bit_offset
                        bits_used = 8 - bit_offset
                        if bits_used < bits_per_int and byte_pos + 1 < len(payload_bytes):
                            value_chunk |= payload_bytes[byte_pos + 1] << bits_used
                        decoded.append(int(value_chunk & mask))
                    if isinstance(default, float) and not isinstance(default, int):
                        return [float(v) for v in decoded]
                    return [int(v) for v in decoded]
                if enc_lower == "pattern":
                    pattern_raw = raw.get("pattern")
                    pattern_seq = self._decode_numeric_field(pattern_raw, 0, default)
                    try:
                        repeats_val = int(raw.get("repeats", raw.get("repeat", 0)))
                    except Exception:
                        repeats_val = 0
                    repeats_val = max(0, repeats_val)
                    sequence: list = []
                    for _ in range(repeats_val):
                        sequence.extend(pattern_seq)
                    if "tail" in raw:
                        tail_seq = self._decode_numeric_field(raw.get("tail"), 0, default)
                        sequence.extend(tail_seq)
                    try:
                        count_val = int(raw.get("count", count or len(sequence)))
                    except Exception:
                        count_val = count or len(sequence)
                    desired = max(count or 0, max(0, count_val))
                    if desired and len(sequence) < desired and pattern_seq:
                        pattern_len = len(pattern_seq)
                        if pattern_len > 0:
                            while len(sequence) < desired:
                                remaining = desired - len(sequence)
                                if remaining >= pattern_len:
                                    sequence.extend(pattern_seq)
                                else:
                                    sequence.extend(pattern_seq[:remaining])
                                    break
                    if desired and len(sequence) > desired:
                        sequence = sequence[:desired]
                    if not desired:
                        desired = len(sequence)
                    if isinstance(default, int):
                        return [int(round(float(v))) if isinstance(v, (int, float)) else int(default) for v in sequence[:desired]]
                    return [float(v) for v in sequence[:desired]]
                if enc_lower == "palette":
                    try:
                        count_val = int(raw.get("count", count or 0))
                    except Exception:
                        count_val = count or 0
                    palette_values = self._to_list(raw.get("values", []))
                    indices_list = self._to_list(raw.get("indices", []))
                    if not palette_values:
                        target_len = max(count or 0, max(0, count_val))
                        fallback = default
                        return [fallback] * target_len
                    target_len = max(count or 0, max(0, count_val), len(indices_list))
                    if target_len <= 0:
                        return []
                    fallback = palette_values[0]
                    decoded = []
                    for idx in range(target_len):
                        if idx < len(indices_list):
                            source = indices_list[idx]
                        else:
                            source = 0
                        try:
                            palette_idx = int(source)
                        except Exception:
                            palette_idx = 0
                        if 0 <= palette_idx < len(palette_values):
                            decoded.append(palette_values[palette_idx])
                        else:
                            decoded.append(fallback)
                    if isinstance(default, int):
                        return [int(round(float(v))) if isinstance(v, (int, float)) else int(default) for v in decoded]
                    return [float(v) for v in decoded]
                if enc_lower.startswith("sparse"):
                    total = raw.get("count", count or 0)
                    try:
                        total_int = int(total)
                    except Exception:
                        total_int = count or 0
                    if total_int < 0:
                        total_int = 0
                    indices = [int(v) for v in self._to_list(raw.get("indices", []))]
                    values = self._to_list(raw.get("values", []))
                    max_index = max(indices) + 1 if indices else 0
                    target_len = max(total_int, count or 0, max_index)
                    decoded = [default] * target_len
                    for pos, idx in enumerate(indices):
                        if 0 <= idx < len(decoded):
                            value = values[pos] if pos < len(values) else default
                            decoded[idx] = value
                    if count and len(decoded) < count:
                        decoded.extend([default] * (count - len(decoded)))
                    if count and len(decoded) > count:
                        decoded = decoded[:count]
                    return decoded
        result = self._to_list(raw)
        if count and len(result) < count:
            result = result + [default] * (count - len(result))
        if count and len(result) > count:
            result = result[:count]
        return result

    def _string_table_entries(self, raw):
        if raw is None:
            return []
        if isinstance(raw, dict):
            lengths = [int(v) for v in self._to_list(raw.get("lengths", []))]
            payload = raw.get("data", raw.get("payload", b""))
            if isinstance(payload, memoryview):
                payload_bytes = payload.tobytes()
            elif isinstance(payload, (bytes, bytearray)):
                payload_bytes = bytes(payload)
            else:
                try:
                    payload_bytes = bytes(payload)
                except Exception:
                    payload_bytes = b""
            encoding = raw.get("enc", raw.get("encoding", "utf-8"))
            entries = []
            offset = 0
            for length in lengths:
                safe_length = max(0, int(length))
                chunk = payload_bytes[offset : offset + safe_length]
                offset += safe_length
                try:
                    entries.append(chunk.decode(encoding))
                except Exception:
                    entries.append(chunk.decode("utf-8", errors="replace"))
            return entries
        return [str(entry) for entry in self._to_list(raw)]

    def _decode_ragged(self, raw):
        if raw is None:
            return []
        if isinstance(raw, dict):
            encoding = raw.get("enc") or raw.get("encoding") or raw.get("mode")
            if isinstance(encoding, str) and encoding.lower() == "repeat":
                try:
                    count_val = int(raw.get("count", 0))
                except Exception:
                    count_val = 0
                base = self._to_list(raw.get("value", []))
                return [base[:] for _ in range(max(0, count_val))]
        if isinstance(raw, dict) and "lengths" in raw and "values" in raw:
            lengths = [int(v) for v in self._to_list(raw.get("lengths", []))]
            total = sum(max(0, int(length)) for length in lengths)
            values = [
                float(v)
                for v in self._decode_numeric_field(raw.get("values"), total, 0.0)
            ]
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

    def _field_typecode(self, raw):
        if isinstance(raw, array):
            return raw.typecode
        if isinstance(raw, dict):
            indices = raw.get("indices")
            if isinstance(indices, array):
                return indices.typecode
            values = raw.get("values")
            if isinstance(values, array):
                return values.typecode
        return None

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
        refs_list = [int(v) for v in self._decode_numeric_field(refs_field, count, -1)]
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
        table_entries = self._string_table_entries(payload["string_table"])
        self.assertTrue(all(isinstance(entry, str) for entry in table_entries))
        self.assertEqual(payload.get("layout", "columnar"), "columnar")
        neurons_block = payload["neurons"]
        self.assertIsInstance(neurons_block, dict)
        self.assertEqual(neurons_block.get("position_encoding"), "linear")
        self.assertIn("linear_indices", neurons_block)
        linear_indices_field = neurons_block["linear_indices"]
        self.assertTrue(isinstance(linear_indices_field, (array, dict)))
        if isinstance(linear_indices_field, dict):
            self.assertEqual(linear_indices_field.get("enc"), "range")
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
        self.assertTrue(isinstance(tensor_fill_refs, (array, dict)))
        if isinstance(tensor_fill_refs, array):
            self.assertIn(
                tensor_fill_refs.typecode,
                ("b", "B", "h", "H", "i", "I", "q", "Q"),
            )
        linear_indices_list = self._to_list(linear_indices_field)
        derived_count = neurons_block.get("count", len(linear_indices_list))
        if not derived_count:
            derived_count = len(linear_indices_list)
        self.assertEqual(derived_count, 2)
        self.assertEqual(neurons_block["weights"].tolist(), [2.0, 1.5])
        self.assertEqual(neurons_block["biases"].tolist(), [-0.25, 0.75])
        self.assertEqual(neurons_block["ages"].tolist(), [5, 3])
        tensor_refs = self._tensor_refs(neurons_block)
        fill_refs = self._to_list(tensor_fill_refs)
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
            self.assertEqual(table_entries[dir_idx], "bi")
        neuron_type_indexes = neurons_block["type_ids"].tolist()
        self.assertTrue(all(isinstance(idx, int) for idx in neuron_type_indexes))
        resolved_types = [
            table_entries[idx] if idx >= 0 else None for idx in neuron_type_indexes
        ]
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
        neuron_count = len(brain.neurons)
        tensor_fill_field = neurons_block.get("tensor_fill_refs")
        tensor_fill_vals = [
            int(v)
            for v in self._decode_numeric_field(tensor_fill_field, neuron_count, -1)
        ]
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
        fill_refs_field = neurons_block.get("tensor_fill_refs")
        fill_refs = [
            int(v)
            for v in self._decode_numeric_field(
                fill_refs_field,
                len(b.neurons),
                -1,
            )
        ]
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
        fill_refs_field = payload["neurons"].get("tensor_fill_refs")
        self.assertIsNotNone(fill_refs_field)
        fill_refs_list = [
            int(v)
            for v in self._decode_numeric_field(
                fill_refs_field,
                len(b.neurons),
                -1,
            )
        ]
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
        fill_refs_field = neurons_block.get("tensor_fill_refs")
        fill_refs = [
            int(v)
            for v in self._decode_numeric_field(
                fill_refs_field,
                len(b.neurons),
                -1,
            )
        ]
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

    def test_snapshot_compresses_size_and_linear_indices_sequences(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        brain = Brain(4, size=(5, 5, 5, 5), store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        previous = None
        for idx in range(6):
            coords = [0, 0, idx // 5, idx % 5]
            if previous is None:
                brain.add_neuron(coords, tensor=[float(idx)])
            else:
                brain.add_neuron(coords, tensor=[float(idx)], connect_to=previous)
            previous = coords
        snapshot_path = brain.save_snapshot()
        payload = self._latest_payload(snapshot_path)
        size_field = payload.get("size")
        self.assertIsInstance(size_field, dict)
        self.assertEqual(size_field.get("enc"), "const")
        self.assertEqual(int(size_field.get("count", 0)), 4)
        size_values = self._to_list(size_field)
        self.assertEqual(size_values, [5, 5, 5, 5])
        neurons_block = payload["neurons"]
        linear_field = neurons_block.get("linear_indices")
        self.assertIsInstance(linear_field, dict)
        self.assertEqual(linear_field.get("enc"), "range")
        self.assertEqual(int(linear_field.get("count", 0)), 6)
        linear_values = self._to_list(linear_field)
        self.assertEqual(linear_values, list(range(6)))

    def test_snapshot_uses_delta_encoding_for_irregular_ages(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        brain = Brain(1, size=16, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        previous = None
        base_age = 10**6
        for idx in range(8):
            pos = (idx,)
            age_value = base_age + idx * (idx + 1) // 2
            kwargs = {"tensor": [float(idx)], "age": age_value}
            if previous is None:
                brain.add_neuron(pos, **kwargs)
            else:
                brain.add_neuron(pos, connect_to=previous, direction="uni", **kwargs)
            previous = pos
        snapshot_path = brain.save_snapshot()
        payload = self._latest_payload(snapshot_path)
        neurons_block = payload["neurons"]
        ages_field = neurons_block.get("ages")
        self.assertIsInstance(ages_field, dict)
        self.assertEqual(ages_field.get("enc"), "delta")
        self.assertEqual(int(ages_field.get("count", 0)), len(brain.neurons))
        deltas_payload = ages_field.get("deltas")
        self.assertIsNotNone(deltas_payload)
        self.assertGreater(len(self._to_list(deltas_payload)), 0)
        reloaded = Brain.load_snapshot(snapshot_path)
        reloaded_ages = [
            int(neuron.age)
            for neuron in sorted(reloaded.neurons.values(), key=lambda n: n.position)
        ]
        expected_ages = [base_age + idx * (idx + 1) // 2 for idx in range(8)]
        self.assertEqual(reloaded_ages, expected_ages)

    def test_snapshot_sparse_indices_use_sequence_encoding(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        brain = Brain(1, size=24, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        previous = None
        for idx in range(20):
            pos = (idx,)
            kwargs = {"tensor": [float(idx)]}
            if 5 <= idx < 10:
                kwargs["weight"] = 2.0 + idx
            if previous is None:
                brain.add_neuron(pos, **kwargs)
            else:
                brain.add_neuron(pos, connect_to=previous, direction="uni", **kwargs)
            previous = pos
        snapshot_path = brain.save_snapshot()
        payload = self._latest_payload(snapshot_path)
        neurons_block = payload["neurons"]
        weights_field = neurons_block.get("weights")
        self.assertIsInstance(weights_field, dict)
        self.assertEqual(weights_field.get("enc"), "sparse")
        indices_field = weights_field.get("indices")
        self.assertIsInstance(indices_field, dict)
        self.assertEqual(indices_field.get("enc"), "range")
        self.assertEqual(int(indices_field.get("count", 0)), 5)
        self.assertEqual(self._to_list(indices_field), list(range(5, 10)))
        values_field = weights_field.get("values")
        self.assertIsInstance(values_field, dict)
        self.assertEqual(values_field.get("enc"), "range")
        reloaded = Brain.load_snapshot(snapshot_path)
        reloaded_weights = [
            float(neuron.weight)
            for neuron in sorted(reloaded.neurons.values(), key=lambda n: n.position)
        ]
        for idx, weight in enumerate(reloaded_weights):
            if 5 <= idx < 10:
                self.assertTrue(
                    math.isclose(weight, 2.0 + idx, rel_tol=1e-6, abs_tol=1e-6)
                )
            else:
                self.assertTrue(math.isclose(weight, 1.0, rel_tol=1e-6, abs_tol=1e-6))

    def test_snapshot_tensor_fill_refs_use_bit_encoding(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        brain = Brain(1, size=12, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        previous = None
        for idx in range(12):
            pos = (idx,)
            tensor_value = 0.25 if idx < 6 else -0.75
            kwargs = {"tensor": [tensor_value]}
            if previous is None:
                brain.add_neuron(pos, **kwargs)
            else:
                brain.add_neuron(pos, connect_to=previous, direction="uni", **kwargs)
            previous = pos
        snapshot_path = brain.save_snapshot()
        payload = self._latest_payload(snapshot_path)
        neurons_block = payload["neurons"]
        fill_field = neurons_block.get("tensor_fill_refs")
        self.assertIsInstance(fill_field, dict)
        self.assertEqual(fill_field.get("enc"), "bits")
        self.assertEqual(int(fill_field.get("count", 0)), len(brain.neurons))
        data_payload = fill_field.get("data", fill_field.get("payload"))
        self.assertIsNotNone(data_payload)
        bit_values = self._to_list(fill_field)
        self.assertEqual(bit_values[:6], [0] * 6)
        self.assertEqual(bit_values[6:], [1] * 6)
        reloaded = Brain.load_snapshot(snapshot_path)
        reloaded_tensors = [
            list(
                neuron.tensor.detach().to("cpu").view(-1).tolist()
                if hasattr(neuron.tensor, "detach")
                else neuron.tensor
            )
            for neuron in sorted(reloaded.neurons.values(), key=lambda n: n.position)
        ]
        for idx, values in enumerate(reloaded_tensors):
            expected = 0.25 if idx < 6 else -0.75
            self.assertTrue(
                all(math.isclose(v, expected, rel_tol=1e-6, abs_tol=1e-6) for v in values)
            )

    def test_snapshot_uses_repeat_encoding_for_sparse_bounds(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        repeated_bounds = tuple((0.0, 1.0) for _ in range(5))
        brain = Brain(
            5,
            mode="sparse",
            sparse_bounds=repeated_bounds,
            store_snapshots=True,
            snapshot_path=tmp,
            snapshot_freq=1,
        )
        origin = (0.0, 0.0, 0.0, 0.0, 0.0)
        brain.add_neuron(origin, tensor=[0.0])
        brain.add_neuron((0.5, 0.0, 0.0, 0.0, 0.0), tensor=[1.0], connect_to=origin)
        snapshot_path = brain.save_snapshot()
        payload = self._latest_payload(snapshot_path)
        sparse_bounds_field = payload.get("sparse_bounds")
        self.assertIsInstance(sparse_bounds_field, dict)
        self.assertEqual(sparse_bounds_field.get("enc"), "repeat")
        self.assertEqual(int(sparse_bounds_field.get("count", 0)), 5)
        decoded_sparse_bounds = self._decode_ragged(sparse_bounds_field)
        self.assertEqual(decoded_sparse_bounds, [list(row) for row in repeated_bounds])

    def test_snapshot_uses_sparse_numeric_blocks(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=6, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        prev = None
        for idx in range(6):
            pos = (idx,)
            tensor_payload = [0.0]
            kwargs = {}
            if idx == 2:
                tensor_payload = [0.1, 0.2]
                kwargs["weight"] = 2.5
            if idx == 3:
                kwargs["bias"] = 0.125
            if idx == 4:
                kwargs["type_name"] = "special"
            if idx == 5:
                kwargs["age"] = 11
            if prev is None:
                b.add_neuron(pos, tensor=tensor_payload, **kwargs)
            else:
                b.add_neuron(pos, tensor=tensor_payload, connect_to=prev, direction="uni", **kwargs)
            prev = pos
        b.synapses[0].weight = 0.6
        b.synapses[0].age = 9
        b.synapses[0].type_name = "syn_special"
        if len(b.synapses) > 1:
            b.synapses[1].direction = "bi"
        snap_path = b.save_snapshot()
        payload = self._latest_payload(snap_path)
        neurons_block = payload["neurons"]
        neuron_count = len(b.neurons)
        weights_field = neurons_block.get("weights")
        biases_field = neurons_block.get("biases")
        ages_field = neurons_block.get("ages")
        type_ids_field = neurons_block.get("type_ids")
        tensor_refs_field = neurons_block.get("tensor_refs")
        self.assertIsInstance(weights_field, dict)
        self.assertEqual(weights_field.get("enc"), "sparse")
        decoded_weights = [
            float(v)
            for v in self._decode_numeric_field(weights_field, neuron_count, 1.0)
        ]
        self.assertTrue(
            all(
                math.isclose(v, 1.0, rel_tol=1e-6, abs_tol=1e-6)
                for i, v in enumerate(decoded_weights)
                if i != 2
            )
        )
        self.assertAlmostEqual(decoded_weights[2], 2.5)
        self.assertIsInstance(biases_field, dict)
        decoded_biases = [
            float(v)
            for v in self._decode_numeric_field(biases_field, neuron_count, 0.0)
        ]
        self.assertAlmostEqual(decoded_biases[3], 0.125)
        self.assertIsInstance(ages_field, dict)
        decoded_ages = [
            int(v)
            for v in self._decode_numeric_field(ages_field, neuron_count, 0)
        ]
        self.assertEqual(decoded_ages[5], 11)
        self.assertIsInstance(type_ids_field, dict)
        decoded_types = [
            int(v)
            for v in self._decode_numeric_field(type_ids_field, neuron_count, -1)
        ]
        self.assertGreaterEqual(decoded_types[4], 0)
        decoded_refs = [
            int(v)
            for v in self._decode_numeric_field(tensor_refs_field, neuron_count, -1)
        ]
        self.assertGreaterEqual(decoded_refs[2], 0)
        for idx, value in enumerate(decoded_refs):
            if idx != 2:
                self.assertEqual(value, -1)
        syn_block = payload["synapses"]
        syn_count = len(b.synapses)
        syn_weights_field = syn_block.get("weights")
        syn_ages_field = syn_block.get("ages")
        syn_types_field = syn_block.get("type_ids")
        direction_field = syn_block.get("direction_ids")
        self.assertIsInstance(syn_weights_field, dict)
        decoded_syn_weights = [
            float(v)
            for v in self._decode_numeric_field(syn_weights_field, syn_count, 1.0)
        ]
        self.assertAlmostEqual(decoded_syn_weights[0], 0.6)
        self.assertIsInstance(syn_ages_field, dict)
        decoded_syn_ages = [
            int(v)
            for v in self._decode_numeric_field(syn_ages_field, syn_count, 0)
        ]
        self.assertEqual(decoded_syn_ages[0], 9)
        self.assertIsInstance(syn_types_field, dict)
        decoded_syn_types = [
            int(v)
            for v in self._decode_numeric_field(syn_types_field, syn_count, -1)
        ]
        self.assertGreaterEqual(decoded_syn_types[0], 0)
        if direction_field is not None:
            decoded_directions = [
                int(v)
                for v in self._decode_numeric_field(direction_field, syn_count, -1)
            ]
            if len(decoded_directions) > 1:
                self.assertGreaterEqual(decoded_directions[1], 0)
        reloaded = Brain.load_snapshot(snap_path)
        self.assertAlmostEqual(reloaded.get_neuron((2,)).weight, 2.5)
        self.assertAlmostEqual(reloaded.get_neuron((3,)).bias, 0.125)
        self.assertEqual(reloaded.get_neuron((5,)).age, 11)
        self.assertEqual(reloaded.get_neuron((4,)).type_name, "special")
        self.assertAlmostEqual(reloaded.synapses[0].weight, 0.6)
        self.assertEqual(reloaded.synapses[0].age, 9)
        self.assertEqual(reloaded.synapses[0].type_name, "syn_special")

    def test_snapshot_uses_rle_for_tensor_refs(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        run_lengths = [32, 16, 16]
        total = sum(run_lengths)
        brain = Brain(
            1,
            size=total + 1,
            store_snapshots=True,
            snapshot_path=tmp,
            snapshot_freq=1,
        )
        prev = None
        tensor_payloads = [
            [0.25, -0.25],
            [1.5, -1.5],
            [3.5, -3.5],
        ]
        idx_counter = 0
        for run_idx, run_length in enumerate(run_lengths):
            payload = tensor_payloads[run_idx]
            for _ in range(run_length):
                pos = (idx_counter,)
                if prev is None:
                    brain.add_neuron(pos, tensor=payload)
                else:
                    brain.add_neuron(pos, tensor=payload, connect_to=prev)
                prev = pos
                idx_counter += 1
        snap_path = brain.save_snapshot()
        payload = self._latest_payload(snap_path)
        neurons_block = payload["neurons"]
        tensor_refs_field = neurons_block.get("tensor_refs")
        self.assertIsInstance(tensor_refs_field, dict)
        self.assertEqual(tensor_refs_field.get("enc"), "rle")
        decoded_refs = [
            int(v)
            for v in self._decode_numeric_field(tensor_refs_field, len(brain.neurons), -1)
        ]
        start = 0
        for expected_index, run_length in enumerate(run_lengths):
            segment = decoded_refs[start : start + run_length]
            self.assertEqual(segment, [expected_index] * run_length)
            start += run_length
        reloaded = Brain.load_snapshot(snap_path)
        self.assertEqual(len(reloaded.neurons), total)

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
        neuron_type_field = neurons_block.get("type_ids")
        syn_type_field = syn_block.get("type_ids")
        print(
            "narrow typecodes:",
            self._field_typecode(neuron_type_field),
            self._field_typecode(syn_type_field),
        )
        self.assertIn("type_ids", neurons_block)
        neuron_type_code = self._field_typecode(neuron_type_field)
        self.assertIn(neuron_type_code, {"b", "B", "h", "H", "i", "I", "q", "Q"})
        tensor_refs_field = neurons_block.get("tensor_refs")
        if tensor_refs_field is not None:
            self.assertIn(
                self._field_typecode(tensor_refs_field),
                ("b", "B", "h", "H", "i", "I", "q", "Q"),
            )
        self.assertIn("tensor_fill_refs", neurons_block)
        self.assertIn(
            self._field_typecode(neurons_block.get("tensor_fill_refs")),
            ("b", "B", "h", "H", "i", "I", "q", "Q"),
        )
        self.assertIn("source_indices", syn_block)
        source_field = syn_block["source_indices"]
        self.assertTrue(isinstance(source_field, (array, dict)))
        if isinstance(source_field, array):
            self.assertEqual(source_field.typecode, "B")
        self.assertIn("target_indices", syn_block)
        target_field = syn_block["target_indices"]
        self.assertTrue(isinstance(target_field, (array, dict)))
        if isinstance(target_field, array):
            self.assertEqual(target_field.typecode, "B")
        self.assertIn("type_ids", syn_block)
        self.assertIn(self._field_typecode(syn_type_field), {"b", "B", "h", "H", "i", "I", "q", "Q"})
        reloaded = Brain.load_snapshot(snap_path)
        self.assertEqual(len(reloaded.neurons), len(b.neurons))
        self.assertEqual(len(reloaded.synapses), len(b.synapses))
        reloaded_types = [n.type_name for n in reloaded.neurons.values()]
        self.assertIn("kind_a", reloaded_types)
        self.assertIn("kind_c", reloaded_types)

    def test_snapshot_uses_palette_for_type_ids(self):
        clear_report_group("brain")
        with tempfile.TemporaryDirectory() as tmp:
            brain = Brain(1, size=60, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
            previous = None
            type_cycle = [None, "alpha", "beta"]
            for idx in range(60):
                pos = (idx,)
                type_name = type_cycle[idx % len(type_cycle)]
                kwargs = {"tensor": [float(idx)]}
                if type_name is not None:
                    kwargs["type_name"] = type_name
                if previous is None:
                    brain.add_neuron(pos, **kwargs)
                else:
                    brain.add_neuron(pos, connect_to=previous, direction="uni", **kwargs)
                previous = pos
            snap_path = brain.save_snapshot()
            payload = self._latest_payload(snap_path)
            neurons_block = payload["neurons"]
            type_field = neurons_block.get("type_ids")
            self.assertIsNotNone(type_field)
            if isinstance(type_field, dict):
                encoding = type_field.get("enc")
                self.assertIn(encoding, {"palette", "pattern"})
            decoded = self._decode_numeric_field(type_field, len(brain.neurons), -1)
            self.assertIn(-1, decoded)
            self.assertIn(0, decoded)
            self.assertIn(1, decoded)
            string_entries = self._string_table_entries(payload.get("string_table"))
            resolved = [
                string_entries[idx] if 0 <= idx < len(string_entries) else None
                for idx in decoded
            ]
            self.assertIn(None, resolved)
            self.assertIn("alpha", resolved)
            self.assertIn("beta", resolved)
            reloaded = Brain.load_snapshot(snap_path)
            self.assertEqual(len(reloaded.neurons), len(brain.neurons))
            decoded_types = [
                getattr(neuron, "type_name", None) for neuron in reloaded.neurons.values()
            ]
            self.assertIn(None, decoded_types)
            self.assertIn("alpha", decoded_types)
            self.assertIn("beta", decoded_types)

    def test_snapshot_pattern_encoding_for_type_ids(self):
        clear_report_group("brain")
        with tempfile.TemporaryDirectory() as tmp:
            brain = Brain(1, size=40, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
            previous = None
            type_cycle = ["alpha", "beta", "gamma", "delta"]
            for idx in range(40):
                pos = (idx,)
                kwargs = {"tensor": [float(idx)], "type_name": type_cycle[idx % len(type_cycle)]}
                if previous is None:
                    brain.add_neuron(pos, **kwargs)
                else:
                    brain.add_neuron(pos, connect_to=previous, direction="uni", **kwargs)
                previous = pos
            snap_path = brain.save_snapshot()
            payload = self._latest_payload(snap_path)
            neurons_block = payload["neurons"]
            type_field = neurons_block.get("type_ids")
            self.assertIsInstance(type_field, dict)
            self.assertEqual(type_field.get("enc"), "pattern")
            pattern_payload = type_field.get("pattern")
            self.assertIsNotNone(pattern_payload)
            pattern_values = [int(v) for v in self._to_list(pattern_payload)]
            self.assertEqual(pattern_values, [0, 1, 2, 3])
            repeats = int(type_field.get("repeats", 0))
            self.assertGreaterEqual(repeats, 1)
            decoded_ids = self._decode_numeric_field(type_field, len(brain.neurons), -1)
            string_entries = self._string_table_entries(payload.get("string_table"))
            resolved = [
                string_entries[idx] if 0 <= idx < len(string_entries) else None
                for idx in decoded_ids
            ]
            expected = [type_cycle[idx % len(type_cycle)] for idx in range(len(brain.neurons))]
            self.assertEqual(resolved, expected)
            reloaded = Brain.load_snapshot(snap_path)
            reloaded_types = [getattr(neuron, "type_name", None) for neuron in reloaded.neurons.values()]
            self.assertEqual(reloaded_types, expected)

    def test_palette_bitpack_supports_five_bit_indices(self):
        clear_report_group("brain")
        with tempfile.TemporaryDirectory() as tmp:
            type_names = [f"type{idx}" for idx in range(32)]
            blocks = 20
            total_neurons = len(type_names) * blocks
            brain = Brain(1, size=total_neurons, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
            previous = None
            expected_sequence: list[str] = []
            for block_idx in range(blocks):
                rotation = type_names[block_idx % len(type_names) :] + type_names[: block_idx % len(type_names)]
                for offset, type_name in enumerate(rotation):
                    global_idx = block_idx * len(type_names) + offset
                    pos = (global_idx,)
                    kwargs = {"tensor": [float(global_idx)], "type_name": type_name}
                    if previous is None:
                        brain.add_neuron(pos, **kwargs)
                    else:
                        brain.add_neuron(pos, connect_to=previous, direction="uni", **kwargs)
                    previous = pos
                    expected_sequence.append(type_name)
            snap_path = brain.save_snapshot()
            payload = self._latest_payload(snap_path)
            neurons_block = payload["neurons"]
            type_field = neurons_block.get("type_ids")
            self.assertIsInstance(type_field, dict)
            self.assertEqual(type_field.get("enc"), "palette")
            indices_field = type_field.get("indices")
            self.assertIsInstance(indices_field, dict)
            self.assertEqual(indices_field.get("enc"), "bitpack")
            self.assertGreaterEqual(int(indices_field.get("bits_per", 0)), 5)
            decoded_ids = self._decode_numeric_field(type_field, len(brain.neurons), -1)
            string_entries = self._string_table_entries(payload.get("string_table"))
            resolved = [
                string_entries[idx] if 0 <= idx < len(string_entries) else None
                for idx in decoded_ids
            ]
            self.assertEqual(resolved, expected_sequence)
            reloaded = Brain.load_snapshot(snap_path)
            reloaded_types = [getattr(neuron, "type_name", None) for neuron in reloaded.neurons.values()]
            self.assertEqual(reloaded_types, expected_sequence)

    def test_sparse_float_values_use_palette_encoding(self):
        clear_report_group("brain")
        with tempfile.TemporaryDirectory() as tmp:
            brain = Brain(1, size=64, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
            previous = None
            bias_cycle = [0.25, -0.5, 0.75]
            for idx in range(64):
                pos = (idx,)
                if idx % 4 == 0:
                    bias_value = bias_cycle[(idx // 4) % len(bias_cycle)]
                else:
                    bias_value = 0.0
                kwargs = {"tensor": [0.0], "bias": bias_value}
                if previous is None:
                    brain.add_neuron(pos, **kwargs)
                else:
                    brain.add_neuron(pos, connect_to=previous, direction="uni", **kwargs)
                previous = pos
            snap_path = brain.save_snapshot()
            payload = self._latest_payload(snap_path)
            neurons_block = payload["neurons"]
            biases_field = neurons_block.get("biases")
            self.assertIsInstance(biases_field, dict)
            self.assertEqual(biases_field.get("enc"), "sparse")
            values_field = biases_field.get("values")
            self.assertIsInstance(values_field, dict)
            self.assertEqual(values_field.get("enc"), "palette")
            palette_values = self._to_list(values_field.get("values", []))
            self.assertGreaterEqual(len(palette_values), 2)
            indices_list = self._to_list(values_field.get("indices", []))
            self.assertEqual(len(indices_list), len(self._to_list(biases_field.get("indices", []))))
            reloaded = Brain.load_snapshot(snap_path)
            reloaded_biases = [
                getattr(neuron, "bias", 0.0) for neuron in reloaded.neurons.values()
            ]
            for idx, bias_value in enumerate(reloaded_biases):
                if idx % 4 == 0:
                    expected = bias_cycle[(idx // 4) % len(bias_cycle)]
                    self.assertAlmostEqual(bias_value, expected)
                else:
                    self.assertAlmostEqual(bias_value, 0.0)

    def test_sparse_float_values_use_rle_encoding(self):
        clear_report_group("brain")
        with tempfile.TemporaryDirectory() as tmp:
            run_a = 160
            run_b = 40
            total = 512
            brain = Brain(1, size=total, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
            previous = None
            for idx in range(total):
                pos = (idx,)
                if idx < run_a:
                    bias_value = 0.625
                elif idx < run_a + run_b:
                    bias_value = -0.5
                else:
                    bias_value = 0.0
                kwargs = {"tensor": [float(idx % 5)], "bias": bias_value}
                if previous is None:
                    brain.add_neuron(pos, **kwargs)
                else:
                    brain.add_neuron(pos, connect_to=previous, direction="uni", **kwargs)
                previous = pos
            snap_path = brain.save_snapshot()
            payload = self._latest_payload(snap_path)
            neurons_block = payload["neurons"]
            biases_field = neurons_block.get("biases")
            self.assertIsInstance(biases_field, dict)
            self.assertEqual(biases_field.get("enc"), "sparse")
            values_field = biases_field.get("values")
            self.assertIsInstance(values_field, dict)
            self.assertEqual(values_field.get("enc"), "rle")
            decoded_biases = [
                float(v)
                for v in self._decode_numeric_field(biases_field, len(brain.neurons), 0.0)
            ]
            for idx, value in enumerate(decoded_biases):
                if idx < run_a:
                    self.assertAlmostEqual(value, 0.625)
                elif idx < run_a + run_b:
                    self.assertAlmostEqual(value, -0.5)
                else:
                    self.assertAlmostEqual(value, 0.0)

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
        tensor_ref_typecode = None if tensor_refs_field is None else self._field_typecode(tensor_refs_field)
        source_field = syn_block["source_indices"]
        target_field = syn_block["target_indices"]
        print(
            "large snapshot typecodes:",
            tensor_ref_typecode,
            self._field_typecode(source_field),
            self._field_typecode(target_field),
        )
        if tensor_refs_field is not None and isinstance(tensor_refs_field, array):
            self.assertEqual(tensor_refs_field.typecode, "b")
        else:
            self.assertTrue(all(ref == -1 for ref in self._tensor_refs(neurons_block)))
        self.assertIn("tensor_fill_refs", neurons_block)
        fill_field = neurons_block.get("tensor_fill_refs")
        if isinstance(fill_field, dict):
            self.assertIn(fill_field.get("enc"), {"range", "const"})
        else:
            self.assertEqual(fill_field.typecode, "H")
        if isinstance(source_field, dict):
            self.assertEqual(source_field.get("enc"), "range")
            values_list = self._to_list(source_field)
            self.assertGreater(len(values_list), 1)
            self.assertEqual(values_list[1] - values_list[0], 1)
        else:
            self.assertEqual(source_field.typecode, "H")
        if isinstance(target_field, dict):
            self.assertEqual(target_field.get("enc"), "range")
            values_list = self._to_list(target_field)
            self.assertGreater(len(values_list), 1)
            self.assertEqual(values_list[1] - values_list[0], 1)
        else:
            self.assertEqual(target_field.typecode, "H")
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
        self.assertEqual(new_payload.get("layout", "columnar"), "columnar")
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
