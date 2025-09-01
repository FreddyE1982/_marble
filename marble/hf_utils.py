from __future__ import annotations

import importlib
import os
import hashlib
from collections import OrderedDict
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


class _ImageEncodingLRUCache:
    """LRU cache storing encoded representations of images."""

    def __init__(self, max_items: int = 20, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.max_items = int(max_items)
        self._od: "OrderedDict[str, Any]" = OrderedDict()

    def _make_key(self, obj: Any) -> str:
        try:
            if isinstance(obj, dict):
                for k in ("url", "uri", "image_url", "path", "filename"):
                    if k in obj and isinstance(obj[k], str):
                        return f"url:{obj[k]}"
        except Exception:
            pass
        for attr in ("url", "uri", "path", "filename"):
            try:
                v = getattr(obj, attr, None)
                if isinstance(v, str) and v:
                    return f"url:{v}"
            except Exception:
                pass
        if isinstance(obj, str) and obj:
            return f"url:{obj}"
        try:
            r = repr(obj)
        except Exception:
            r = f"{type(obj).__name__}:{id(obj)}"
        return f"repr:{hashlib.sha256(r.encode('utf-8', errors='ignore')).hexdigest()}"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        if key in self._od:
            val = self._od.pop(key)
            self._od[key] = val
            return val
        return None

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        if key in self._od:
            self._od.pop(key)
        self._od[key] = value
        while len(self._od) > max(0, self.max_items):
            self._od.popitem(last=False)

    def get_or_encode(self, obj: Any, codec) -> Tuple[str, Any]:
        key = self._make_key(obj)
        cached = self.get(key)
        if cached is not None:
            return key, cached
        enc = codec.encode(obj)
        self.put(key, enc)
        return key, enc


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
    def __init__(
        self,
        raw_dataset: Any,
        codec,
        *,
        cache_size: int = 20,
        cache_enabled: bool = True,
    ) -> None:
        self._ds = raw_dataset
        self._codec = codec
        self._cache = _ImageEncodingLRUCache(max_items=cache_size, enabled=cache_enabled)

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

    # Image cache helpers -------------------------------------------------
    def cache_image(self, obj: Any) -> Tuple[str, Any]:
        """Return a stable key and encoded representation for ``obj``."""

        return self._cache.get_or_encode(obj, self._codec)

    def get_cached_image(self, key: str) -> Optional[Any]:
        """Retrieve an already encoded image by ``key`` if present."""

        return self._cache.get(key)


def load_hf_streaming_dataset(
    path: str,
    *,
    name: Optional[str] = None,
    split: Optional[str] = "train",
    codec=None,
    streaming: bool = True,
    trust_remote_code: bool = False,
    download_config=None,
    cache_images: bool = True,
    cache_size: int = 20,
    **kwargs: Any,
) -> HFStreamingDatasetWrapper:
    """Load a Hugging Face dataset with automatic encoding and streaming.

    Parameters
    ----------
    path: str
        Dataset identifier passed to ``datasets.load_dataset``.
    name: Optional[str]
        Optional configuration name.
    split: Optional[str]
        Dataset split to load, defaults to ``"train"``.
    codec: Optional[Any]
        Codec used for automatic tensor encoding.
    streaming: bool
        Whether to enable streaming mode (default: ``True``). When set to
        ``False`` the full dataset is materialized in memory but images are
        replaced by their URL/path so that they can be fetched lazily.
    trust_remote_code: bool
        Forwarded to ``datasets.load_dataset`` when supported.
    download_config: Optional[Any]
        ``datasets.DownloadConfig`` forwarded to ``datasets.load_dataset``.
    cache_images: bool
        Whether to cache encoded images for reuse (default: ``True``).
    cache_size: int
        Maximum number of encoded images kept in the cache (default: ``20``).
    **kwargs: Any
        Additional keyword arguments for ``datasets.load_dataset``.
    """
    _, ds_mod = _ensure_hf_imports()
    if ds_mod is None:
        raise RuntimeError("datasets is not installed. Please install 'datasets'.")
    used_codec = codec if codec is not None else make_default_codec()
    import inspect

    ds_kwargs: Dict[str, Any] = {
        "path": path,
        "name": name,
        "split": split,
        "streaming": True,
        **kwargs,
    }
    if download_config is not None:
        ds_kwargs["download_config"] = download_config
    if "trust_remote_code" in inspect.signature(ds_mod.load_dataset).parameters:
        ds_kwargs["trust_remote_code"] = trust_remote_code

    ds_stream = ds_mod.load_dataset(**ds_kwargs)
    if streaming:
        ds = ds_stream
    else:
        def _extract_image_url(obj: Any) -> Any:
            try:
                if isinstance(obj, dict):
                    for k in ("url", "path", "filename", "image_url"):
                        v = obj.get(k)
                        if isinstance(v, str):
                            return v
            except Exception:
                pass
            for attr in ("url", "path", "filename", "image_url"):
                try:
                    v = getattr(obj, attr)
                    if isinstance(v, str):
                        return v
                except Exception:
                    continue
            return obj

        materialized = []
        for ex in ds_stream:
            if isinstance(ex, dict):
                materialized.append({k: _extract_image_url(v) for k, v in ex.items()})
            else:
                materialized.append(_extract_image_url(ex))
        ds = materialized
    try:
        report(
            "huggingface",
            "load_dataset",
            {"path": path, "name": name, "split": split, "streaming": bool(streaming)},
            "dataset",
        )
    except Exception:
        pass
    return HFStreamingDatasetWrapper(ds, used_codec, cache_size=cache_size, cache_enabled=cache_images)


__all__ = [
    "hf_login",
    "hf_logout",
    "HFEncodedExample",
    "HFStreamingDatasetWrapper",
    "load_hf_streaming_dataset",
]

