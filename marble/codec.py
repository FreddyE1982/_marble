from __future__ import annotations

import json
import pickle
import os
from typing import Any, Dict, Iterable, List, Sequence, Union

TensorLike = Union[List[int], "_TorchTensor"]


def _safe_report(groupname: str, itemname: str, data: Any, *subgroups: str) -> None:
    try:
        from .marblemain import report  # lazy to avoid circular import at module import time
        report(groupname, itemname, data, *subgroups)
    except Exception:
        pass


class UniversalTensorCodec:
    def __init__(self) -> None:
        self._byte_to_token: Dict[int, int] = {}
        self._token_to_byte: List[int] = []
        self._torch = self._try_import_torch()
        self._device = self._select_device()

    def reset_vocab(self) -> None:
        self._byte_to_token.clear()
        self._token_to_byte.clear()

    def vocab_size(self) -> int:
        return len(self._token_to_byte)

    def encode(self, obj: Any) -> TensorLike:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        tokens = self._bytes_to_tokens(data)
        out = self._to_tensor(tokens)
        try:
            ln = int(out.numel()) if hasattr(out, "numel") else len(out)  # type: ignore[arg-type]
        except Exception:
            ln = -1
        _safe_report("codec", "encode", {"obj_type": type(obj).__name__, "tokens": ln}, "events")
        return out

    def decode(self, tokens: Union[TensorLike, Sequence[int]]) -> Any:
        token_list = self._to_list(tokens)
        data = self._tokens_to_bytes(token_list)
        obj = pickle.loads(data)
        _safe_report("codec", "decode", {"ok": True, "vocab_size": self.vocab_size()}, "events")
        return obj

    def export_vocab(self, path: str) -> None:
        payload = {"token_to_byte": self._token_to_byte}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        _safe_report("codec", "export_vocab", {"path": path, "size": self.vocab_size()}, "io")

    def import_vocab(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        token_to_byte = payload.get("token_to_byte")
        if not isinstance(token_to_byte, list) or not all(isinstance(x, int) and 0 <= x <= 255 for x in token_to_byte):
            raise ValueError("Invalid vocabulary file format")
        self._token_to_byte = list(token_to_byte)
        self._byte_to_token = {b: i for i, b in enumerate(self._token_to_byte)}
        _safe_report("codec", "import_vocab", {"path": path, "size": self.vocab_size()}, "io")

    # Internal helpers
    def _bytes_to_tokens(self, data: bytes) -> List[int]:
        for b in data:
            if b not in self._byte_to_token:
                self._byte_to_token[b] = len(self._token_to_byte)
                self._token_to_byte.append(b)
        return [self._byte_to_token[b] for b in data]

    def _tokens_to_bytes(self, tokens: Iterable[int]) -> bytes:
        try:
            return bytes(self._token_to_byte[t] for t in tokens)
        except (IndexError, TypeError) as e:
            raise ValueError("Token id out of range for current vocabulary") from e

    def _to_tensor(self, values: List[int]) -> TensorLike:
        if self._torch is not None:
            return self._torch.tensor(values, dtype=self._torch.long, device=self._device)
        return values

    def _to_list(self, maybe_tensor: Union[TensorLike, Sequence[int]]) -> List[int]:
        if self._torch is not None and self._is_torch_tensor(maybe_tensor):
            t = maybe_tensor.detach().to("cpu")
            return [int(x) for x in t.view(-1).tolist()]
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
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"


__all__ = ["UniversalTensorCodec", "TensorLike"]

