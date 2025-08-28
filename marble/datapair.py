from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

from .codec import UniversalTensorCodec, TensorLike


def _safe_report(groupname: str, itemname: str, data: Any, *subgroups: str) -> None:
    try:
        from .marblemain import report  # lazy import to avoid circular deps
        report(groupname, itemname, data, *subgroups)
    except Exception:
        pass


class DataPair:
    def __init__(self, left: Any, right: Any) -> None:
        self.left = left
        self.right = right

    def encode(self, codec: UniversalTensorCodec) -> Tuple[TensorLike, TensorLike]:
        enc_l = codec.encode(self.left)
        enc_r = codec.encode(self.right)
        try:
            ln_l = int(enc_l.numel()) if hasattr(enc_l, "numel") else len(enc_l)  # type: ignore[arg-type]
        except Exception:
            ln_l = -1
        try:
            ln_r = int(enc_r.numel()) if hasattr(enc_r, "numel") else len(enc_r)  # type: ignore[arg-type]
        except Exception:
            ln_r = -1
        _safe_report("datapair", "encode", {"left_tokens": ln_l, "right_tokens": ln_r}, "events")
        return enc_l, enc_r

    @classmethod
    def decode(
        cls,
        encoded: Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]],
        codec: UniversalTensorCodec,
    ) -> "DataPair":
        enc_l, enc_r = encoded
        left = codec.decode(enc_l)
        right = codec.decode(enc_r)
        _safe_report("datapair", "decode", {"ok": True}, "events")
        return cls(left, right)


def make_datapair(left: Any, right: Any) -> DataPair:
    dp = DataPair(left, right)
    _safe_report("datapair", "make", {"left_type": type(left).__name__, "right_type": type(right).__name__}, "events")
    return dp


def encode_datapair(codec: UniversalTensorCodec, left: Any, right: Any) -> Tuple[TensorLike, TensorLike]:
    return DataPair(left, right).encode(codec)


def decode_datapair(
    codec: UniversalTensorCodec,
    encoded_left: Union[TensorLike, Sequence[int]],
    encoded_right: Union[TensorLike, Sequence[int]],
) -> Tuple[Any, Any]:
    dp = DataPair.decode((encoded_left, encoded_right), codec)
    return dp.left, dp.right


__all__ = [
    "DataPair",
    "make_datapair",
    "encode_datapair",
    "decode_datapair",
]

