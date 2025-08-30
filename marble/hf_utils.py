from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# We intentionally import from marble.marblemain for the reporter when available.
try:
    from .marblemain import report, make_default_codec
except Exception:  # pragma: no cover - fallback if circular during bootstrap
    def report(*args, **kwargs):
        return None

    def make_default_codec():
        from .marblemain import UniversalTensorCodec  # type: ignore
        return UniversalTensorCodec()


def _ensure_hf_imports() -> Tuple[Any, Any]:
    hf_hub = None
    ds_mod = None
    try:
        hf_hub = importlib.import_module("huggingface_hub")
    except Exception:
        hf_hub = None
    try:
        ds_mod = importlib.import_module("datasets")
    except Exception:
        ds_mod = None
    return hf_hub, ds_mod


def hf_login(token: Optional[str] = None, *, add_to_git_credential: bool = False, endpoint: Optional[str] = None) -> Dict[str, Any]:
    hf_hub, _ = _ensure_hf_imports()
    if hf_hub is None:
        raise RuntimeError("huggingface_hub is not installed. Please install 'huggingface_hub'.")

    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        raise ValueError("No Hugging Face token provided. Set HF_TOKEN env var or pass token explicitly.")

    kwargs: Dict[str, Any] = {"token": tok, "add_to_git_credential": bool(add_to_git_credential)}
    if endpoint:
        kwargs["endpoint"] = endpoint
    hf_hub.login(**kwargs)
    who: Dict[str, Any]
    try:
        who = hf_hub.whoami()
    except Exception:
        who = {"ok": True}
    try:
        report("huggingface", "login", {"ok": True, "user": who.get("name") if isinstance(who, dict) else None}, "auth")
    except Exception:
        pass
    return who if isinstance(who, dict) else {"ok": True}


def hf_logout() -> None:
    hf_hub, _ = _ensure_hf_imports()
    if hf_hub is None:
        return
    try:
        hf_hub.logout()
        report("huggingface", "logout", {"ok": True}, "auth")
    except Exception:
        pass


TensorLike = Union[list, "_TorchTensor"]  # runtime-only typing aid


class HFEncodedExample:
    def __init__(self, data: Dict[str, Any], codec) -> None:
        self._data = data
        self._codec = codec

    def get_raw(self, key: str) -> Any:
        return self._data[key]

    def __getitem__(self, key: str) -> TensorLike:
        val = self._data[key]
        enc = self._codec.encode(val)
        try:
            ln = int(enc.numel()) if hasattr(enc, "numel") else len(enc)
        except Exception:
            ln = -1
        try:
            report("huggingface", "encode_field", {"field": str(key), "tokens": ln}, "dataset")
        except Exception:
            pass
        return enc

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        for k in self._data.keys():
            yield self[k]

    def items(self):
        for k in self._data.keys():
            yield k, self[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict_encoded(self) -> Dict[str, TensorLike]:
        return {k: self[k] for k in self._data.keys()}


class HFStreamingDatasetWrapper:
    def __init__(self, raw_dataset: Any, codec) -> None:
        self._ds = raw_dataset
        self._codec = codec

    def __iter__(self):
        for ex in self._ds:
            if isinstance(ex, dict):
                yield HFEncodedExample(ex, self._codec)
            else:
                yield HFEncodedExample({"value": ex}, self._codec)

    def __getitem__(self, idx):
        ex = self._ds[idx]
        if isinstance(ex, list):
            return [HFEncodedExample(e, self._codec) if isinstance(e, dict) else HFEncodedExample({"value": e}, self._codec) for e in ex]
        return HFEncodedExample(ex, self._codec) if isinstance(ex, dict) else HFEncodedExample({"value": ex}, self._codec)

    def __len__(self) -> int:
        try:
            return len(self._ds)  # type: ignore[arg-type]
        except Exception:
            return 0

    def raw(self) -> Any:
        return self._ds


def load_hf_streaming_dataset(
    path: str,
    *,
    name: Optional[str] = None,
    split: Optional[str] = "train",
    codec=None,
    streaming: bool = True,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> HFStreamingDatasetWrapper:
    _, ds_mod = _ensure_hf_imports()
    if ds_mod is None:
        raise RuntimeError("datasets is not installed. Please install 'datasets'.")
    used_codec = codec if codec is not None else make_default_codec()
    import inspect

    ds_kwargs: Dict[str, Any] = {
        "path": path,
        "name": name,
        "split": split,
        "streaming": streaming,
        **kwargs,
    }
    if "trust_remote_code" in inspect.signature(ds_mod.load_dataset).parameters:
        ds_kwargs["trust_remote_code"] = trust_remote_code
    ds = ds_mod.load_dataset(**ds_kwargs)
    try:
        report("huggingface", "load_dataset", {"path": path, "name": name, "split": split, "streaming": bool(streaming)}, "dataset")
    except Exception:
        pass
    return HFStreamingDatasetWrapper(ds, used_codec)


__all__ = [
    "hf_login",
    "hf_logout",
    "HFEncodedExample",
    "HFStreamingDatasetWrapper",
    "load_hf_streaming_dataset",
]

