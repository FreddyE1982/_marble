from __future__ import annotations

import importlib
import os
import hashlib
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Set

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
    def __init__(
        self,
        data: Dict[str, Any],
        codec,
        parent: Optional["HFStreamingDatasetWrapper"] = None,
        encoded_fields: Optional[Sequence[str]] = None,
    ) -> None:
        self._data = data
        self._codec = codec
        self._parent = parent
        self._encoded_fields: Set[str] = set(encoded_fields or [])

    def get_raw(self, key: str) -> Any:
        return self._data[key]

    def __getitem__(self, key: str) -> TensorLike:
        val = self._data[key]
        if key in self._encoded_fields:
            enc = val
        elif self._parent is not None and key in self._parent.image_fields:
            enc = self._parent._get_or_download_image(val)
            self._data[key] = enc
            self._encoded_fields.add(key)
        else:
            enc = self._codec.encode(val)
            self._data[key] = enc
            self._encoded_fields.add(key)
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
        image_fields: Optional[Sequence[str]] = None,
    ) -> None:
        self._ds = raw_dataset
        self._codec = codec
        self._cache = _ImageEncodingLRUCache(max_items=cache_size, enabled=cache_enabled)
        self._image_fields = set(image_fields or [])

    def _encode_all_image_fields(self, ex: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
        encoded: Set[str] = set()
        for field in self._image_fields:
            if field in ex:
                ex[field] = self._get_or_download_image(ex[field])
                encoded.add(field)
        return ex, encoded

    def __iter__(self):
        for ex in self._ds:
            if isinstance(ex, dict):
                ex, enc = self._encode_all_image_fields(ex)
                yield HFEncodedExample(ex, self._codec, self, enc)
            else:
                ex_dict = {"value": ex}
                ex_dict, enc = self._encode_all_image_fields(ex_dict)
                yield HFEncodedExample(ex_dict, self._codec, self, enc)

    def __getitem__(self, idx):
        ex = self._ds[idx]
        if isinstance(ex, list):
            result = []
            for e in ex:
                if isinstance(e, dict):
                    e, enc = self._encode_all_image_fields(e)
                    result.append(HFEncodedExample(e, self._codec, self, enc))
                else:
                    e_dict = {"value": e}
                    e_dict, enc = self._encode_all_image_fields(e_dict)
                    result.append(HFEncodedExample(e_dict, self._codec, self, enc))
            return result
        if isinstance(ex, dict):
            ex, enc = self._encode_all_image_fields(ex)
            return HFEncodedExample(ex, self._codec, self, enc)
        ex_dict = {"value": ex}
        ex_dict, enc = self._encode_all_image_fields(ex_dict)
        return HFEncodedExample(ex_dict, self._codec, self, enc)

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

    # Lazy image handling -------------------------------------------------
    @property
    def image_fields(self) -> Set[str]:
        return self._image_fields

    def _download_image(self, url: str) -> bytes:
        from urllib.parse import urlparse
        import urllib.request

        parsed = urlparse(url)
        path = url
        if parsed.scheme in {"http", "https"}:
            with urllib.request.urlopen(url) as r:
                return r.read()
        if parsed.scheme == "file":
            path = parsed.path
        with open(path, "rb") as f:
            return f.read()

    def _get_or_download_image(self, ref: Any) -> Any:
        if isinstance(ref, str):
            key = f"url:{ref}"
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            data = self._download_image(ref)
            enc = self._codec.encode(data)
            self._cache.put(key, enc)
            return enc
        key, enc = self.cache_image(ref)
        return enc


def load_hf_streaming_dataset(
    path: str,
    *,
    name: Optional[str] = None,
    split: Optional[str] = "train",
    codec=None,
    streaming: str = "memory",
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
    streaming: str
        Select how the dataset is materialized. Options are:
        ``"memory"`` (default) – stream directly into memory,
        ``"memory_lazy_images"`` – like ``"memory"`` but replace image objects
        with their URL so they are downloaded on demand,
        ``"disk"`` – store the dataset on disk and stream examples from there,
        ``"disk_lazy_images"`` – like ``"disk"`` but replace image objects with
        URLs,
        ``"memory_full"`` – materialize the entire dataset in memory,
        ``"memory_full_lazy_images"`` – materialize everything in memory but
        keep only image URLs,
        ``"disk_full_lazy_image"`` – materialize the dataset on disk with image
        URLs only.
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

    streaming_mode = str(streaming).lower()
    valid_modes = {
        "memory",
        "memory_lazy_images",
        "disk",
        "disk_lazy_images",
        "memory_full",
        "memory_full_lazy_images",
        "disk_full_lazy_image",
    }
    if streaming_mode not in valid_modes:
        raise ValueError(f"streaming must be one of {sorted(valid_modes)}")

    ds_kwargs: Dict[str, Any] = {
        "path": path,
        "name": name,
        "split": split,
        "streaming": streaming_mode.startswith("memory"),
        **kwargs,
    }
    if not streaming_mode.startswith("memory"):
        ds_kwargs["keep_in_memory"] = False
    if download_config is not None:
        ds_kwargs["download_config"] = download_config
    if "trust_remote_code" in inspect.signature(ds_mod.load_dataset).parameters:
        ds_kwargs["trust_remote_code"] = trust_remote_code

    ds_stream = ds_mod.load_dataset(**ds_kwargs)

    image_fields: Set[str] = set()
    try:
        feats = getattr(ds_stream, "features", {})
        for fname, ftype in getattr(feats, "items", lambda: [])():
            if getattr(ftype, "__class__", None) and ftype.__class__.__name__ == "Image":
                image_fields.add(str(fname))
    except Exception:
        pass

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

    def _replace_images_iter(ds):
        for ex in ds:
            if isinstance(ex, dict):
                yield {k: _extract_image_url(v) for k, v in ex.items()}
            else:
                yield _extract_image_url(ex)

    if streaming_mode == "memory":
        ds = ds_stream
    elif streaming_mode == "memory_lazy_images":
        ds = _replace_images_iter(ds_stream)
    elif streaming_mode == "disk":
        ds = ds_stream
    elif streaming_mode in {"disk_lazy_images", "disk_full_lazy_image"}:
        ds = ds_stream.map(lambda ex: {k: _extract_image_url(v) for k, v in ex.items()}, keep_in_memory=False)
    elif streaming_mode == "memory_full":
        ds = list(ds_stream)
    elif streaming_mode == "memory_full_lazy_images":
        ds = list(_replace_images_iter(ds_stream))
    else:  # safeguard
        ds = ds_stream
    try:
        report(
            "huggingface",
            "load_dataset",
            {"path": path, "name": name, "split": split, "streaming": streaming_mode},
            "dataset",
        )
    except Exception:
        pass
    return HFStreamingDatasetWrapper(
        ds,
        used_codec,
        cache_size=cache_size,
        cache_enabled=cache_images,
        image_fields=image_fields,
    )


__all__ = [
    "hf_login",
    "hf_logout",
    "HFEncodedExample",
    "HFStreamingDatasetWrapper",
    "load_hf_streaming_dataset",
]

