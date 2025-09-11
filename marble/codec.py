from __future__ import annotations

import json
import pickle
import marshal
import zlib
from typing import Any, Dict, List, Sequence, Union

# Pre-computed single-byte sequences to avoid allocating new objects during
# encoding. Index ``i`` contains ``bytes([i])``.
_BYTE_TABLE: List[bytes] = [bytes([i]) for i in range(256)]

# Empirically chosen compression level balancing speed and size.
# Benchmarks across levels 1-9 showed level 2 offering the fastest encode
# times while retaining essentially identical compression ratio.
_COMPRESSION_LEVEL = 2

TensorLike = Union[List[int], "_TorchTensor"]


def _safe_report(groupname: str, itemname: str, data: Any, *subgroups: str) -> None:
    try:
        from .marblemain import report  # lazy to avoid circular import at module import time
        report(groupname, itemname, data, *subgroups)
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
        serializer = 1
        try:
            data = marshal.dumps(obj)
        except Exception:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)
            serializer = 0
        compressor = zlib.compressobj(_COMPRESSION_LEVEL)
        tokens = bytearray()
        # stream serializer marker and payload separately to avoid copying large buffers
        tokens.extend(compressor.compress(_BYTE_TABLE[serializer]))
        tokens.extend(compressor.compress(data))
        tokens.extend(compressor.flush())
        out = self._to_tensor(tokens)
        try:
            ln = int(out.numel()) if hasattr(out, "numel") else len(out)  # type: ignore[arg-type]
        except Exception:
            ln = -1
        _safe_report("codec", "encode", {"obj_type": type(obj).__name__, "tokens": ln}, "events")
        return out

    def decode(self, tokens: Union[TensorLike, Sequence[int]]) -> Any:
        data = self._tokens_to_bytes(tokens)
        if data[:1] in (b"\x00", b"\x01"):
            marker, payload = data[0], data[1:]
            if marker == 1:
                obj = marshal.loads(payload)
            else:
                obj = pickle.loads(payload)
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
            return zlib.decompress(data)
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


__all__ = ["UniversalTensorCodec", "TensorLike"]

