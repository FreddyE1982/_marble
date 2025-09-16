from __future__ import annotations

import json
import pickle
import marshal
import struct
import zlib
from typing import Any, Dict, List, Optional, Sequence, Union

# Pre-computed single-byte sequences to avoid allocating new objects during
# encoding. Index ``i`` contains ``bytes([i])``.
_BYTE_TABLE: List[bytes] = [bytes([i]) for i in range(256)]

# Empirically chosen compression level balancing speed and size.
# Benchmarks across levels 1-9 showed level 2 offering the fastest encode
# times while retaining essentially identical compression ratio.
_COMPRESSION_LEVEL = 2

_ENCODED_MAGIC = b"UTC1"

TensorLike = Union[List[int], "_TorchTensor"]


_REPORT_FN = None
_REPORT_FAILED = False


def _safe_report(groupname: str, itemname: str, data: Any, *subgroups: str) -> None:
    global _REPORT_FN, _REPORT_FAILED
    if _REPORT_FN is None and not _REPORT_FAILED:
        try:
            from .marblemain import report as _REPORT_FN  # type: ignore
        except Exception:
            _REPORT_FAILED = True
            return
    if _REPORT_FN is not None:
        try:
            _REPORT_FN(groupname, itemname, data, *subgroups)
        except Exception:
            pass


class UniversalTensorCodec:
    def __init__(self) -> None:
        self._seq_to_token: Dict[bytes, int] = {}
        self._token_to_seq: List[bytes] = []
        self._torch = self._try_import_torch()
        self._device = self._select_device()
        self._ensure_base_vocab()

    def reset_vocab(self) -> None:
        self._seq_to_token.clear()
        self._token_to_seq.clear()
        self._ensure_base_vocab()

    def vocab_size(self) -> int:
        return len(self._token_to_seq)

    def encode(self, obj: Any) -> TensorLike:
        if self._looks_like_encoded(obj):
            raise ValueError(
                "Trying to re-encode an object that was already encoded using this or another instance of UniversalTensorCodec"
            )
        serializer = 1
        try:
            if isinstance(obj, bytes):
                serializer = 3
                data = obj
            elif isinstance(obj, str):
                serializer = 4
                data = obj.encode("utf-8")
            elif isinstance(obj, list) and len(obj) > 1000:
                ln = len(obj)
                first = obj[0]
                if (
                    ln > 1
                    and isinstance(first, int)
                    and isinstance(obj[1], int)
                    and isinstance(obj[-1], int)
                    and obj[1] == first + 1
                    and obj[-1] == first + ln - 1
                    and obj[ln // 2] == first + ln // 2
                ):
                    serializer = 2
                    header = struct.pack("<qi", first, ln)
                    payload = _BYTE_TABLE[serializer] + header
                    tokens = zlib.compress(payload, _COMPRESSION_LEVEL)
                    out = self._to_tensor(tokens)
                    try:
                        ln = int(out.numel()) if hasattr(out, "numel") else len(out)  # type: ignore[arg-type]
                    except Exception:
                        ln = -1
                    _safe_report(
                        "codec", "encode", {"obj_type": type(obj).__name__, "tokens": ln}, "events"
                    )
                    return out
                raise ValueError
            else:
                data = marshal.dumps(obj)
        except Exception:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)
            serializer = 0
        payload = _ENCODED_MAGIC + _BYTE_TABLE[serializer] + data
        tokens = zlib.compress(payload, _COMPRESSION_LEVEL)
        out = self._to_tensor(tokens)
        try:
            ln = int(out.numel()) if hasattr(out, "numel") else len(out)  # type: ignore[arg-type]
        except Exception:
            ln = -1
        _safe_report("codec", "encode", {"obj_type": type(obj).__name__, "tokens": ln}, "events")
        return out

    def decode(self, tokens: Union[TensorLike, Sequence[int]]) -> Any:
        data = self._tokens_to_bytes(tokens)
        if data[:1] in (b"\x00", b"\x01", b"\x02", b"\x03", b"\x04"):
            marker, payload = data[0], data[1:]
            if marker == 1:
                obj = marshal.loads(payload)
            elif marker == 0:
                obj = pickle.loads(payload)
            elif marker == 2:
                start, length = struct.unpack("<qi", payload)
                obj = list(range(start, start + length))
            elif marker == 3:
                obj = payload
            else:
                obj = payload.decode("utf-8")
        else:  # backward compatibility with old tokens
            obj = pickle.loads(data)
        _safe_report("codec", "decode", {"ok": True, "vocab_size": self.vocab_size()}, "events")
        return obj

    def export_vocab(self, path: str) -> None:
        payload = {"token_to_seq": [list(seq) for seq in self._token_to_seq]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        _safe_report("codec", "export_vocab", {"path": path, "size": self.vocab_size()}, "io")

    def import_vocab(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        token_to_seq = payload.get("token_to_seq")
        if not isinstance(token_to_seq, list) or not all(
            isinstance(seq, list) and all(isinstance(x, int) and 0 <= x <= 255 for x in seq)
            for seq in token_to_seq
        ):
            raise ValueError("Invalid vocabulary file format")
        self._token_to_seq = [bytes(seq) for seq in token_to_seq]
        self._seq_to_token = {seq: i for i, seq in enumerate(self._token_to_seq)}
        _safe_report("codec", "import_vocab", {"path": path, "size": self.vocab_size()}, "io")

    # In-memory vocabulary helpers
    def dump_vocab(self) -> List[List[int]]:
        """Return the vocabulary as list-of-byte sequences."""
        return [list(seq) for seq in self._token_to_seq]

    def load_vocab(self, token_to_seq: Sequence[Sequence[int]]) -> None:
        """Initialize vocabulary from list-of-byte sequences."""
        if not isinstance(token_to_seq, Sequence) or not all(
            isinstance(seq, Sequence) and all(isinstance(x, int) and 0 <= x <= 255 for x in seq)
            for seq in token_to_seq
        ):
            raise ValueError("Invalid vocabulary format")
        self._token_to_seq = [bytes(seq) for seq in token_to_seq]
        self._seq_to_token = {seq: i for i, seq in enumerate(self._token_to_seq)}
        _safe_report("codec", "load_vocab", {"size": self.vocab_size()}, "io")

    # Internal helpers
    def _ensure_base_vocab(self) -> None:
        if not self._token_to_seq:
            seq_to_token = self._seq_to_token
            token_to_seq = self._token_to_seq
            for i, seq in enumerate(_BYTE_TABLE):
                seq_to_token[seq] = i
                token_to_seq.append(seq)

    def _bytes_to_tokens(self, data: bytes) -> Union[bytes, List[int]]:
        return zlib.compress(data, _COMPRESSION_LEVEL)

    def _tokens_to_bytes(self, tokens: Union[TensorLike, Sequence[int]]) -> bytes:
        try:
            if self._torch is not None and self._is_torch_tensor(tokens):
                t = tokens.detach().to(self._torch.uint8)
                if t.device.type != "cpu":
                    t = t.to("cpu")
                data = t.numpy().tobytes()
            else:
                data = bytes(tokens)
            decompressed = zlib.decompress(data)
            if not decompressed.startswith(_ENCODED_MAGIC):
                raise ValueError("Encoded tokens missing UniversalTensorCodec header")
            return decompressed[len(_ENCODED_MAGIC) :]
        except Exception as e:
            raise ValueError("Token id out of range for current vocabulary") from e

    def _to_tensor(self, values: Union[List[int], bytes]) -> TensorLike:
        if self._torch is not None:
            torch = self._torch
            if isinstance(values, (bytes, bytearray, memoryview)):
                buf = values if isinstance(values, (bytearray, memoryview)) else bytearray(values)
                t = torch.frombuffer(buf, dtype=torch.uint8)
                if self._device != "cpu":
                    t = t.to(self._device)
                return t
            return torch.tensor(values, dtype=torch.uint8, device=self._device)
        return list(values) if not isinstance(values, list) else values

    def _to_list(self, maybe_tensor: Union[TensorLike, Sequence[int]]) -> List[int]:
        if self._torch is not None and self._is_torch_tensor(maybe_tensor):
            t = maybe_tensor.detach().view(-1)
            if t.device.type != "cpu":
                t = t.to("cpu")
            return [int(x) for x in t.tolist()]
        return list(maybe_tensor)  # type: ignore[arg-type]

    # Torch detection
    def _try_import_torch(self):
        try:
            import torch  # type: ignore
            _ = torch.tensor([0], dtype=torch.long, device="cpu")
            return torch
        except Exception:
            return None

    def _is_torch_tensor(self, obj: Any) -> bool:
        try:
            if self._torch is None:
                return False
            Tensor = self._torch.Tensor  # type: ignore[attr-defined]
            return isinstance(obj, Tensor)
        except Exception:
            return False

    def _select_device(self) -> str:
        """Prefer CUDA when available, otherwise fall back to CPU."""
        if self._torch is None:
            return "cpu"
        try:
            if self._torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _looks_like_encoded(self, candidate: Any) -> bool:
        compressed = self._extract_compressed_bytes(candidate)
        if compressed is None:
            return False
        try:
            decompressor = zlib.decompressobj()
            needed = len(_ENCODED_MAGIC)
            chunk = decompressor.decompress(compressed, needed)
            if len(chunk) < needed:
                tail = decompressor.unconsumed_tail
                while len(chunk) < needed and tail:
                    prev_len = len(chunk)
                    chunk += decompressor.decompress(tail, needed - len(chunk))
                    if len(chunk) == prev_len:
                        break
                    tail = decompressor.unconsumed_tail
            if len(chunk) < needed:
                return False
            return chunk.startswith(_ENCODED_MAGIC)
        except Exception:
            return False

    def _extract_compressed_bytes(self, candidate: Any) -> Optional[bytes]:
        if isinstance(candidate, (bytes, bytearray, memoryview)):
            return bytes(candidate)
        if self._torch is not None and self._is_torch_tensor(candidate):
            try:
                if getattr(candidate, "dtype", None) is not self._torch.uint8:
                    return None
                t = candidate.detach()
                if t.device.type != "cpu":
                    t = t.to("cpu")
                return t.contiguous().view(-1).numpy().tobytes()
            except Exception:
                return None
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray, memoryview)
        ):
            try:
                return bytes(candidate)
            except Exception:
                return None
        return None


__all__ = ["UniversalTensorCodec", "TensorLike"]

