"""
UniversalTensorCodec: Encode any Python object to an integer tensor and decode it back.

Rules respected:
- Only this file contains imports.
- No other module imports anything.

Design:
- Uses pickle to serialize arbitrary Python objects to bytes.
- Builds a byte-level vocabulary on the fly (observed bytes -> token ids).
- Encodes to a 1D tensor of integer token ids. If torch is available, returns
  a CPU LongTensor; otherwise returns a plain Python list of ints.
- Vocabulary can be exported/imported to/from a JSON file.
"""

from __future__ import annotations

# Only file allowed to import
import json
import pickle
import math
import random
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable
import os
import time
import tempfile
import hashlib
import importlib
try:
    import msvcrt  # type: ignore
except Exception:
    msvcrt = None  # type: ignore


# Avoid initializing unsupported CPU backends that cause stderr warnings.
# Disables NNPACK probing on CPUs where it isn't available to prevent noisy logs.
try:
    os.environ.setdefault("PYTORCH_DISABLE_NNPACK", "1")
except Exception:
    pass

from .codec import UniversalTensorCodec, TensorLike

__all__ = ["UniversalTensorCodec"]


from .datapair import (
    DataPair,
    make_datapair,
    encode_datapair,
    decode_datapair,
)

__all__ += ["DataPair", "make_datapair", "encode_datapair", "decode_datapair"]


# -----------------------------
# Hugging Face utils (moved)
# -----------------------------
from .hf_utils import (
    hf_login,
    hf_logout,
    HFEncodedExample,
    HFStreamingDatasetWrapper,
    load_hf_streaming_dataset,
)

__all__ += [
    "hf_login",
    "hf_logout",
    "HFEncodedExample",
    "HFStreamingDatasetWrapper",
    "load_hf_streaming_dataset",
]


from .graph import (
    _DeviceHelper,
    _NEURON_TYPES,
    _SYNAPSE_TYPES,
    register_neuron_type,
    register_synapse_type,
    Neuron,
    Synapse,
)

__all__ += [
    "_DeviceHelper",
    "_NEURON_TYPES",
    "_SYNAPSE_TYPES",
    "register_neuron_type",
    "register_synapse_type",
    "Neuron",
    "Synapse",
]


# -----------------------------
# Neuron Plugin: Conv1D (pure Python, parameterized by incoming neurons)
# -----------------------------

class Conv1DNeuronPlugin:
    """Pure-Python 1D convolution whose parameters are driven by connected neurons.

    Requires at least 5 incoming synapses to the target neuron; parameters are read
    from the source neurons' tensors in a deterministic order (sorted by source
    position when available, else by source id):
      1) kernel: list[float] (1D filter coefficients)
      2) stride: int >= 1
      3) padding: int >= 0 (zero-pad both sides)
      4) dilation: int >= 1
      5) bias: float (added to each output element)

    The input signal is taken from the provided input_value (if not None) or the
    neuron's own tensor, coerced to a flat list[float]. The output is a list[float].
    """

    def on_init(self, neuron: "Neuron") -> None:
        # Strict wiring validation at creation time: exactly 5 PARAM inputs and exactly 1 outgoing
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"Conv1D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "conv1d_init", {"incoming": len(inc), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def _key_src(self, syn: "Synapse"):
        src = syn.source
        pos = getattr(src, "position", None)
        if isinstance(pos, tuple):
            return (0, tuple(pos))
        return (1, id(src))

    def _to_list1d(self, x) -> list:
        try:
            if hasattr(x, "detach") and hasattr(x, "tolist"):
                lst = x.detach().to("cpu").view(-1).tolist()
            elif isinstance(x, (list, tuple)):
                lst = list(x)
            else:
                lst = [x]
        except Exception:
            lst = []
        out = []
        for v in lst:
            try:
                out.append(float(v))
            except Exception:
                out.append(0.0)
        return out

    def _first_scalar(self, x, *, default: float = 0.0, min_val: Optional[float] = None) -> float:
        vals = self._to_list1d(x)
        v = float(vals[0]) if vals else float(default)
        if min_val is not None and v < min_val:
            v = float(min_val)
        return v

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        # Select parameter synapses explicitly when labeled; fallback to first 5
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) >= 5:
            param_incs.sort(key=self._key_src)
            sel = param_incs[:5]
        else:
            if len(incoming) < 5:
                raise ValueError("Conv1D plugin requires 5 incoming synapses (kernel, stride, padding, dilation, bias)")
            incoming.sort(key=self._key_src)
            sel = incoming[:5]
        k_src = sel[0].source
        s_src = sel[1].source
        p_src = sel[2].source
        d_src = sel[3].source
        b_src = sel[4].source

        kernel = self._to_list1d(getattr(k_src, "tensor", [])) or [1.0]
        stride = int(self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0))
        padding = int(max(0.0, self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)))
        dilation = int(self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0))
        bias = float(self._first_scalar(getattr(b_src, "tensor", 0.0), default=0.0))

        # Prefer per-neuron learnable parameters if provided by SelfAttention
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        learn_kernel = lstore.get("kernel")
        learn_bias = lstore.get("conv_bias")

        # Build input from incoming DATA synapses if present; else use input_value/tensor
        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            x_list_all: List[float] = []
            for s in data_incs:
                x_list_all += self._to_list1d(getattr(s.source, "tensor", []))
            x1 = x_list_all
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x1 = self._to_list1d(x)

        # If torch is available, perform computation on the neuron's device (CUDA preferred)
        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        # Use high-performance torch path only on CUDA; on CPU prefer pure-Python to avoid
        # backend initialization warnings on some hardware (e.g., NNPACK unsupported).
        if torch is not None and str(device) == "cuda":
            try:
                xt = torch.tensor(x1, dtype=torch.float32, device=device).view(1, 1, -1)
                wt = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel, dtype=torch.float32, device=device)).view(1, 1, -1)
                bt_val = float(bias)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bt_val], dtype=torch.float32, device=device))
                # Use conv1d with provided stride/padding/dilation; this runs on CUDA when available
                y = torch.nn.functional.conv1d(xt, wt, bias=bt, stride=stride, padding=padding, dilation=dilation)
                y = y.view(-1)
                try:
                    report("neuron", "conv1d", {"in": int(xt.numel()), "out": int(y.numel()), "k": int(wt.numel()), "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                # Fallback to pure Python path below
                pass

        # Zero-pad and compute in pure Python when torch is unavailable or failed
        if padding > 0:
            x1 = ([0.0] * padding) + x1 + ([0.0] * padding)
        n = len(x1)
        klen = len(kernel)
        span = (klen - 1) * dilation + 1
        out_len = 0
        if n >= span:
            out_len = 1 + (n - span) // stride
        y_list: list = []
        # If a learnable kernel is present but torch path failed, convert to list
        if hasattr(learn_kernel, "detach"):
            try:
                kernel = list(learn_kernel.detach().to("cpu").view(-1).tolist())
                klen = len(kernel)
            except Exception:
                pass
        if hasattr(learn_bias, "detach"):
            try:
                bias = float(learn_bias.detach().to("cpu").view(-1)[0].item())
            except Exception:
                try:
                    bias = float(learn_bias)
                except Exception:
                    pass
        for t in range(out_len):
            base = t * stride
            acc = 0.0
            for i in range(klen):
                xi = base + i * dilation
                acc += kernel[i] * x1[xi]
            y_list.append(acc + bias)
        try:
            report("neuron", "conv1d", {"in": len(x1), "out": len(y_list), "k": klen, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)  # type: ignore[attr-defined]
        except Exception:
            return y_list


# Register conv1d plugin via dedicated module
try:
    from .plugins.conv1d import Conv1DNeuronPlugin as _Conv1DPlugin
    register_neuron_type("conv1d", _Conv1DPlugin())
except Exception:
    pass

__all__ += ["Conv1DNeuronPlugin"]


# -----------------------------
# Convolution wiring helpers
# -----------------------------

def wire_param_synapses(brain: "Brain", conv_neuron: "Neuron", params: Sequence["Neuron"]) -> List["Synapse"]:
    syns: List["Synapse"] = []
    for pn in params:
        syns.append(brain.connect(getattr(pn, "position"), getattr(conv_neuron, "position"), direction="uni", type_name="param"))
    try:
        report("builder", "wire_param_synapses", {"count": len(syns)}, "conv")
    except Exception:
        pass
    return syns


def wire_data_synapses(brain: "Brain", conv_neuron: "Neuron", data: Sequence["Neuron"]) -> List["Synapse"]:
    syns: List["Synapse"] = []
    for dn in data:
        syns.append(brain.connect(getattr(dn, "position"), getattr(conv_neuron, "position"), direction="uni", type_name="data"))
    try:
        report("builder", "wire_data_synapses", {"count": len(syns)}, "conv")
    except Exception:
        pass
    return syns


def create_conv1d_from_existing(brain: "Brain", dst: "Neuron", params: Sequence["Neuron"], data: Optional[Sequence["Neuron"]] = None) -> "Neuron":
    if len(params) != 5:
        raise ValueError("create_conv1d_from_existing requires exactly 5 parameter neurons")
    # Create base neuron, wire, then promote
    conv = brain.add_neuron(brain.available_indices()[0] if brain.available_indices() else (0,) * int(getattr(brain, "n", 1)), tensor=[0.0])
    wire_param_synapses(brain, conv, params)
    if data:
        wire_data_synapses(brain, conv, data)
    brain.connect(getattr(conv, "position"), getattr(dst, "position"), direction="uni")
    # Promote
    conv.type_name = "conv1d"
    plugin = _NEURON_TYPES.get("conv1d")
    if plugin is not None and hasattr(plugin, "on_init"):
        plugin.on_init(conv)  # type: ignore[attr-defined]
    try:
        report("builder", "create_conv1d_from_existing", {"ok": True}, "conv")
    except Exception:
        pass
    return conv


__all__ += ["wire_param_synapses", "wire_data_synapses", "create_conv1d_from_existing"]


# -----------------------------
# Pooling wiring helpers
# -----------------------------

def create_maxpool1d_from_existing(
    brain: "Brain",
    dst: "Neuron",
    params: Sequence["Neuron"],
    data: Optional[Sequence["Neuron"]] = None,
) -> "Neuron":
    if len(params) != 3:
        raise ValueError("create_maxpool1d_from_existing requires exactly 3 parameter neurons (k,s,p)")
    pool = brain.add_neuron(brain.available_indices()[0] if brain.available_indices() else (0,) * int(getattr(brain, "n", 1)), tensor=[0.0])
    wire_param_synapses(brain, pool, params)
    if data:
        wire_data_synapses(brain, pool, data)
    brain.connect(getattr(pool, "position"), getattr(dst, "position"), direction="uni")
    pool.type_name = "maxpool1d"
    plugin = _NEURON_TYPES.get("maxpool1d")
    if plugin is not None and hasattr(plugin, "on_init"):
        plugin.on_init(pool)  # type: ignore[attr-defined]
    try:
        report("builder", "create_maxpool1d_from_existing", {"ok": True}, "pool")
    except Exception:
        pass
    return pool


def create_maxpool2d_from_existing(
    brain: "Brain",
    dst: "Neuron",
    params: Sequence["Neuron"],
    data: Optional[Sequence["Neuron"]] = None,
) -> "Neuron":
    if len(params) != 3:
        raise ValueError("create_maxpool2d_from_existing requires exactly 3 parameter neurons (k,s,p)")
    pool = brain.add_neuron(brain.available_indices()[0] if brain.available_indices() else (0,) * int(getattr(brain, "n", 1)), tensor=[0.0])
    wire_param_synapses(brain, pool, params)
    if data:
        wire_data_synapses(brain, pool, data)
    brain.connect(getattr(pool, "position"), getattr(dst, "position"), direction="uni")
    pool.type_name = "maxpool2d"
    plugin = _NEURON_TYPES.get("maxpool2d")
    if plugin is not None and hasattr(plugin, "on_init"):
        plugin.on_init(pool)  # type: ignore[attr-defined]
    try:
        report("builder", "create_maxpool2d_from_existing", {"ok": True}, "pool")
    except Exception:
        pass
    return pool


def create_maxpool3d_from_existing(
    brain: "Brain",
    dst: "Neuron",
    params: Sequence["Neuron"],
    data: Optional[Sequence["Neuron"]] = None,
) -> "Neuron":
    if len(params) != 3:
        raise ValueError("create_maxpool3d_from_existing requires exactly 3 parameter neurons (k,s,p)")
    pool = brain.add_neuron(brain.available_indices()[0] if brain.available_indices() else (0,) * int(getattr(brain, "n", 1)), tensor=[0.0])
    wire_param_synapses(brain, pool, params)
    if data:
        wire_data_synapses(brain, pool, data)
    brain.connect(getattr(pool, "position"), getattr(dst, "position"), direction="uni")
    pool.type_name = "maxpool3d"
    plugin = _NEURON_TYPES.get("maxpool3d")
    if plugin is not None and hasattr(plugin, "on_init"):
        plugin.on_init(pool)  # type: ignore[attr-defined]
    try:
        report("builder", "create_maxpool3d_from_existing", {"ok": True}, "pool")
    except Exception:
        pass
    return pool


__all__ += [
    "create_maxpool1d_from_existing",
    "create_maxpool2d_from_existing",
    "create_maxpool3d_from_existing",
]


# -----------------------------
# Neuron Plugins: Conv2D and Conv3D (CUDA-aware via torch if available)
# -----------------------------

class _ConvNDCommon:
    def _key_src(self, syn: "Synapse"):
        src = syn.source
        pos = getattr(src, "position", None)
        if isinstance(pos, tuple):
            return (0, tuple(pos))
        return (1, id(src))

    def _to_list1d(self, x) -> list:
        try:
            if hasattr(x, "detach") and hasattr(x, "tolist"):
                lst = x.detach().to("cpu").view(-1).tolist()
            elif isinstance(x, (list, tuple)):
                # Flatten nested lists
                def flat(z):
                    for it in z:
                        if isinstance(it, (list, tuple)):
                            yield from flat(it)
                        else:
                            yield it
                lst = list(flat(list(x)))
            else:
                lst = [x]
        except Exception:
            lst = []
        out = []
        for v in lst:
            try:
                out.append(float(v))
            except Exception:
                out.append(0.0)
        return out

    def _first_scalar(self, x, *, default: float = 0.0, min_val: Optional[float] = None) -> float:
        vals = self._to_list1d(x)
        v = float(vals[0]) if vals else float(default)
        if min_val is not None and v < min_val:
            v = float(min_val)
        return v


class Conv2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"Conv2D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "conv2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 5:
            raise ValueError("Conv2D plugin requires 5 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src, b_src = [s.source for s in param_incs[:5]]

        kernel_1d = self._to_list1d(getattr(k_src, "tensor", [])) or [1.0]
        stride = int(self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0))
        padding = int(max(0.0, self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)))
        dilation = int(self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0))
        bias = float(self._first_scalar(getattr(b_src, "tensor", 0.0), default=0.0))
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        learn_kernel = lstore.get("kernel")
        learn_bias = lstore.get("conv_bias")

        # Build input from incoming DATA synapses if present; fallback to input_value/tensor
        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            rows: List[List[float]] = []
            for s in data_incs:
                rows.append(self._to_list1d(getattr(s.source, "tensor", [])))
            # Use common width (min length) across rows
            if rows:
                width = min((len(r) for r in rows if r), default=0)
            else:
                width = 0
            if width <= 0:
                x_vals = []
                H = 0
                W = 0
            else:
                rows = [r[:width] for r in rows]
                H = len(rows)
                W = width
                x_vals = [v for r in rows for v in r]
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x_vals = self._to_list1d(x)
            N = max(1, len(x_vals))
            rh = int(math.isqrt(N))
            if rh * rh == N:
                H = W = rh
            else:
                H, W = N, 1

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0 and str(device) == "cuda":
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, H, W)
                # Infer kernel shape: prefer square if perfect square else (len,1)
                L = max(1, len(kernel_1d))
                r = int(math.isqrt(L))
                if r * r == L:
                    kh = kw = r
                else:
                    kh, kw = L, 1
                base_w = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel_1d, dtype=torch.float32, device=device))
                wt = base_w.view(1, 1, kh, kw)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bias], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv2d(
                    xt, wt, bias=bt, stride=(stride, stride), padding=(padding, padding), dilation=(dilation, dilation)
                )
                y = y.view(-1)
                try:
                    report("neuron", "conv2d", {"inHW": [H, W], "out": int(y.numel()), "kHW": [kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure Python fallback
        if padding > 0 and H > 0 and W > 0:
            padded = []
            zero_row = [0.0] * (W + 2 * padding)
            for _ in range(padding):
                padded.append(list(zero_row))
            for r_ in range(H):
                row = [0.0] * padding + x_vals[r_ * W:(r_ + 1) * W] + [0.0] * padding
                padded.append(row)
            for _ in range(padding):
                padded.append(list(zero_row))
            H2, W2 = len(padded), len(padded[0])
        else:
            padded = [x_vals[r_ * W:(r_ + 1) * W] for r_ in range(H)] if H > 0 and W > 0 else []
            H2, W2 = (len(padded), len(padded[0])) if padded else (0, 0)
        # Infer kernel shape
        # If learnable present but torch path failed, convert to list
        if hasattr(learn_kernel, "detach"):
            try:
                kernel_1d = list(learn_kernel.detach().to("cpu").view(-1).tolist())
            except Exception:
                pass
        if hasattr(learn_bias, "detach"):
            try:
                bias = float(learn_bias.detach().to("cpu").view(-1)[0].item())
            except Exception:
                try:
                    bias = float(learn_bias)
                except Exception:
                    pass
        L = max(1, len(kernel_1d))
        r = int(math.isqrt(L))
        if r * r == L:
            kh = kw = r
        else:
            kh, kw = L, 1
        out_h = 0 if H2 < kh else 1 + (H2 - ((kh - 1) * dilation + 1)) // stride
        out_w = 0 if W2 < kw else 1 + (W2 - ((kw - 1) * dilation + 1)) // stride
        y_list: List[float] = []
        for oy in range(out_h):
            base_y = oy * stride
            for ox in range(out_w):
                base_x = ox * stride
                acc = 0.0
                for ky in range(kh):
                    for kx in range(kw):
                        iy = base_y + ky * dilation
                        ix = base_x + kx * dilation
                        acc += kernel_1d[ky * kw + kx] * padded[iy][ix]
                y_list.append(acc + bias)
        try:
            report("neuron", "conv2d", {"inHW": [H, W], "out": len(y_list), "kHW": [kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


class Conv3DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"Conv3D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "conv3d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 5:
            raise ValueError("Conv3D plugin requires 5 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src, b_src = [s.source for s in param_incs[:5]]

        kernel_1d = self._to_list1d(getattr(k_src, "tensor", [])) or [1.0]
        stride = int(self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0))
        padding = int(max(0.0, self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)))
        dilation = int(self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0))
        bias = float(self._first_scalar(getattr(b_src, "tensor", 0.0), default=0.0))
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        learn_kernel = lstore.get("kernel")
        learn_bias = lstore.get("conv_bias")

        # Build input from incoming DATA synapses if present; fallback to input_value/tensor
        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            # Treat each data source as one 2D slice; infer H,W per slice as square if possible else (len,1)
            slices: List[List[float]] = []
            dims: List[Tuple[int, int]] = []
            for s in data_incs:
                vals = self._to_list1d(getattr(s.source, "tensor", []))
                N = max(1, len(vals))
                r = int(math.isqrt(N))
                if r * r == N:
                    H = W = r
                else:
                    H, W = N, 1
                dims.append((H, W))
                slices.append(vals[: H * W])
            # Harmonize to the smallest H,W across slices
            if dims:
                Hmin = min(h for h, _ in dims)
                Wmin = min(w for _, w in dims)
            else:
                Hmin = Wmin = 0
            x_vals = []
            D = len(slices)
            for sl, (h, w) in zip(slices, dims):
                # reshape sl to h x w then crop to Hmin x Wmin and flatten
                for rr in range(Hmin):
                    row = sl[rr * w:(rr + 1) * w]
                    x_vals.extend(row[:Wmin])
            H, W = Hmin, Wmin
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            vals = self._to_list1d(x)
            N = max(1, len(vals))
            r3 = round(N ** (1.0 / 3.0))
            if r3 > 0 and (r3 * r3 * r3) == N:
                D = H = W = int(r3)
                x_vals = vals[: D * H * W]
            else:
                D, H, W = N, 1, 1
                x_vals = vals

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0 and str(device) == "cuda":
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, D, H, W)
                # Infer kernel shape: prefer cube if perfect cube; else (L,1,1)
                L = max(1, len(kernel_1d))
                r = round(L ** (1.0 / 3.0))
                if r > 0 and (r * r * r) == L:
                    kd = kh = kw = int(r)
                else:
                    kd, kh, kw = L, 1, 1
                base_w = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel_1d, dtype=torch.float32, device=device))
                wt = base_w.view(1, 1, kd, kh, kw)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bias], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv3d(
                    xt, wt, bias=bt, stride=(stride, stride, stride), padding=(padding, padding, padding), dilation=(dilation, dilation, dilation)
                )
                y = y.view(-1)
                try:
                    report("neuron", "conv3d", {"inDHW": [D, H, W], "out": int(y.numel()), "kDHW": [kd, kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure Python fallback
        if padding > 0 and H > 0 and W > 0:
            D2, H2, W2 = D + 2 * padding, H + 2 * padding, W + 2 * padding
            padded = [0.0] * (D2 * H2 * W2)
            for dz in range(D):
                for yy in range(H):
                    for xx in range(W):
                        pd = dz + padding
                        py = yy + padding
                        px = xx + padding
                        padded[pd * (H2 * W2) + py * W2 + px] = x_vals[dz * (H * W) + yy * W + xx]
        else:
            padded = list(x_vals)
            D2, H2, W2 = D, H, W
        out_d = 0 if D2 < 1 else 1 + (D2 - ((1 - 1) * dilation + 1)) // stride  # kernel depth inferred below
        # Infer kernel shape
        if hasattr(learn_kernel, "detach"):
            try:
                kernel_1d = list(learn_kernel.detach().to("cpu").view(-1).tolist())
            except Exception:
                pass
        if hasattr(learn_bias, "detach"):
            try:
                bias = float(learn_bias.detach().to("cpu").view(-1)[0].item())
            except Exception:
                try:
                    bias = float(learn_bias)
                except Exception:
                    pass
        L = max(1, len(kernel_1d))
        r = round(L ** (1.0 / 3.0))
        if r > 0 and (r * r * r) == L:
            kd = kh = kw = int(r)
        else:
            kd, kh, kw = L, 1, 1
        def idx3(a, d, h, w, H_, W_):
            return a[d * (H_ * W_) + h * W_ + w]
        out_d = 0 if D2 < kd else 1 + (D2 - ((kd - 1) * dilation + 1)) // stride
        out_h = 0 if H2 < kh else 1 + (H2 - ((kh - 1) * dilation + 1)) // stride
        out_w = 0 if W2 < kw else 1 + (W2 - ((kw - 1) * dilation + 1)) // stride
        y_list: List[float] = []
        for od in range(out_d):
            base_d = od * stride
            for oy in range(out_h):
                base_y = oy * stride
                for ox in range(out_w):
                    base_x = ox * stride
                    acc = 0.0
                    for kz in range(kd):
                        for ky in range(kh):
                            for kx in range(kw):
                                iz = base_d + kz * dilation
                                iy = base_y + ky * dilation
                                ix = base_x + kx * dilation
                                acc += kernel_1d[(kz * kh + ky) * kw + kx] * idx3(padded, iz, iy, ix, H2, W2)
                    y_list.append(acc + bias)
        try:
            report("neuron", "conv3d", {"inDHW": [D, H, W], "out": len(y_list), "kDHW": [kd, kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


try:
    from .plugins.conv2d import Conv2DNeuronPlugin as _Conv2DPlugin
    from .plugins.conv3d import Conv3DNeuronPlugin as _Conv3DPlugin
    register_neuron_type("conv2d", _Conv2DPlugin())
    register_neuron_type("conv3d", _Conv3DPlugin())
except Exception:
    pass

__all__ += ["Conv2DNeuronPlugin", "Conv3DNeuronPlugin"]


# -----------------------------
# Neuron Plugins: ConvTranspose1D/2D/3D
# -----------------------------

class ConvTranspose1DNeuronPlugin(Conv1DNeuronPlugin):
    def on_init(self, neuron: "Neuron") -> None:
        # Reuse strict rule: 5 PARAM + 1 outgoing
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"ConvTranspose1D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "convtranspose1d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        # Gather params and data using Conv1D helpers
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 5:
            raise ValueError("ConvTranspose1D requires 5 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src, b_src = [s.source for s in param_incs[:5]]
        kernel = self._to_list1d(getattr(k_src, "tensor", [])) or [1.0]
        stride = int(self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0))
        padding = int(max(0.0, self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)))
        dilation = int(self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0))
        bias = float(self._first_scalar(getattr(b_src, "tensor", 0.0), default=0.0))
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        learn_kernel = lstore.get("kernel")
        learn_bias = lstore.get("conv_bias")

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            x1 = []
            for s in data_incs:
                x1 += self._to_list1d(getattr(s.source, "tensor", []))
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x1 = self._to_list1d(x)

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and str(device) == "cuda":
            try:
                xt = torch.tensor(x1, dtype=torch.float32, device=device).view(1, 1, -1)
                wt = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel, dtype=torch.float32, device=device)).view(1, 1, -1)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bias], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv_transpose1d(xt, wt, bias=bt, stride=stride, padding=padding, dilation=dilation)
                y = y.view(-1)
                try:
                    report("neuron", "convtranspose1d", {"in": int(xt.numel()), "out": int(y.numel()), "k": int(wt.numel()), "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure-Python fallback
        n = len(x1)
        klen = len(kernel)
        out_len = (n - 1) * stride - 2 * padding + (klen - 1) * dilation + 1
        y_list = [0.0] * max(0, out_len)
        if hasattr(learn_kernel, "detach"):
            try:
                kernel = list(learn_kernel.detach().to("cpu").view(-1).tolist())
                klen = len(kernel)
            except Exception:
                pass
        if hasattr(learn_bias, "detach"):
            try:
                bias = float(learn_bias.detach().to("cpu").view(-1)[0].item())
            except Exception:
                try:
                    bias = float(learn_bias)
                except Exception:
                    pass
        for t in range(n):
            base = t * stride
            for i in range(klen):
                oi = base + i * dilation - padding
                if 0 <= oi < len(y_list):
                    y_list[oi] += kernel[i] * x1[t]
        y_list = [v + bias for v in y_list]
        try:
            report("neuron", "convtranspose1d", {"in": n, "out": len(y_list), "k": klen, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


class ConvTranspose2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"ConvTranspose2D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "convtranspose2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 5:
            raise ValueError("ConvTranspose2D requires 5 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src, b_src = [s.source for s in param_incs[:5]]
        kernel_1d = self._to_list1d(getattr(k_src, "tensor", [])) or [1.0]
        stride = int(self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0))
        padding = int(max(0.0, self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)))
        dilation = int(self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0))
        bias = float(self._first_scalar(getattr(b_src, "tensor", 0.0), default=0.0))
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        learn_kernel = lstore.get("kernel")
        learn_bias = lstore.get("conv_bias")

        # Build 2D input from DATA rows
        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            rows = [self._to_list1d(getattr(s.source, "tensor", [])) for s in data_incs]
            width = min((len(r) for r in rows if r), default=0)
            if width <= 0:
                x_vals = []
                H = W = 0
            else:
                rows = [r[:width] for r in rows]
                H = len(rows)
                W = width
                x_vals = [v for r in rows for v in r]
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x_vals = self._to_list1d(x)
            N = max(1, len(x_vals))
            rh = int(math.isqrt(N))
            if rh * rh == N:
                H = W = rh
            else:
                H, W = N, 1

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0 and str(device) == "cuda":
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, H, W)
                L = max(1, len(kernel_1d))
                r = int(math.isqrt(L))
                if r * r == L:
                    kh = kw = r
                else:
                    kh, kw = L, 1
                base_w = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel_1d, dtype=torch.float32, device=device))
                wt = base_w.view(1, 1, kh, kw)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bias], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv_transpose2d(
                    xt, wt, bias=bt, stride=(stride, stride), padding=(padding, padding), dilation=(dilation, dilation)
                )
                y = y.view(-1)
                try:
                    report("neuron", "convtranspose2d", {"inHW": [H, W], "out": int(y.numel()), "kHW": [kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure Python fallback (simplified)
        # Compute output dims
        if hasattr(learn_kernel, "detach"):
            try:
                kernel_1d = list(learn_kernel.detach().to("cpu").view(-1).tolist())
            except Exception:
                pass
        if hasattr(learn_bias, "detach"):
            try:
                bias = float(learn_bias.detach().to("cpu").view(-1)[0].item())
            except Exception:
                try:
                    bias = float(learn_bias)
                except Exception:
                    pass
        L = max(1, len(kernel_1d))
        r = int(math.isqrt(L))
        if r * r == L:
            kh = kw = r
        else:
            kh, kw = L, 1
        out_h = (H - 1) * stride - 2 * padding + (kh - 1) * dilation + 1
        out_w = (W - 1) * stride - 2 * padding + (kw - 1) * dilation + 1
        y2 = [[0.0 for _ in range(max(0, out_w))] for _ in range(max(0, out_h))]
        for iy in range(H):
            for ix in range(W):
                base_y = iy * stride
                base_x = ix * stride
                val = x_vals[iy * W + ix]
                for ky in range(kh):
                    for kx in range(kw):
                        oy = base_y + ky * dilation - padding
                        ox = base_x + kx * dilation - padding
                        if 0 <= oy < len(y2) and 0 <= ox < len(y2[0]):
                            y2[oy][ox] += kernel_1d[ky * kw + kx] * val
        y_list = [v + bias for row in y2 for v in row]
        try:
            report("neuron", "convtranspose2d", {"inHW": [H, W], "out": len(y_list), "kHW": [kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


class ConvTranspose3DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"ConvTranspose3D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "convtranspose3d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 5:
            raise ValueError("ConvTranspose3D requires 5 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src, b_src = [s.source for s in param_incs[:5]]
        kernel_1d = self._to_list1d(getattr(k_src, "tensor", [])) or [1.0]
        stride = int(self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0))
        padding = int(max(0.0, self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)))
        dilation = int(self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0))
        bias = float(self._first_scalar(getattr(b_src, "tensor", 0.0), default=0.0))
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        learn_kernel = lstore.get("kernel")
        learn_bias = lstore.get("conv_bias")

        # Build 3D input from DATA slices
        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            slices = []
            dims = []
            for s in data_incs:
                vals = self._to_list1d(getattr(s.source, "tensor", []))
                N = max(1, len(vals))
                r = int(math.isqrt(N))
                if r * r == N:
                    H = W = r
                else:
                    H, W = N, 1
                dims.append((H, W))
                slices.append(vals[: H * W])
            Hmin = min((h for h, _ in dims), default=0)
            Wmin = min((w for _, w in dims), default=0)
            x_vals = []
            D = len(slices)
            for sl, (h, w) in zip(slices, dims):
                for rr in range(Hmin):
                    row = sl[rr * w:(rr + 1) * w]
                    x_vals.extend(row[:Wmin])
            H, W = Hmin, Wmin
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            vals = self._to_list1d(x)
            N = max(1, len(vals))
            r3 = round(N ** (1.0 / 3.0))
            if r3 > 0 and (r3 * r3 * r3) == N:
                D = H = W = int(r3)
                x_vals = vals[: D * H * W]
            else:
                D, H, W = N, 1, 1
                x_vals = vals

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0 and str(device) == "cuda":
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, D, H, W)
                L = max(1, len(kernel_1d))
                r = round(L ** (1.0 / 3.0))
                if r > 0 and (r * r * r) == L:
                    kd = kh = kw = int(r)
                else:
                    kd, kh, kw = L, 1, 1
                base_w = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel_1d, dtype=torch.float32, device=device))
                wt = base_w.view(1, 1, kd, kh, kw)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bias], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv_transpose3d(
                    xt, wt, bias=bt, stride=(stride, stride, stride), padding=(padding, padding, padding), dilation=(dilation, dilation, dilation)
                )
                y = y.view(-1)
                try:
                    report("neuron", "convtranspose3d", {"inDHW": [D, H, W], "out": int(y.numel()), "kDHW": [kd, kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure Python fallback
        L = max(1, len(kernel_1d))
        r = round(L ** (1.0 / 3.0))
        if r > 0 and (r * r * r) == L:
            kd = kh = kw = int(r)
        else:
            kd, kh, kw = L, 1, 1
        out_d = (D - 1) * stride - 2 * padding + (kd - 1) * dilation + 1
        out_h = (H - 1) * stride - 2 * padding + (kh - 1) * dilation + 1
        out_w = (W - 1) * stride - 2 * padding + (kw - 1) * dilation + 1
        y = [0.0] * max(0, out_d * out_h * out_w)
        def idx3(a, d, h, w, H_, W_):
            return d * (H_ * W_) + h * W_ + w
        for iz in range(D):
            for iy in range(H):
                for ix in range(W):
                    base_d = iz * stride
                    base_y = iy * stride
                    base_x = ix * stride
                    val = x_vals[idx3(x_vals, 0, 0, 0, 1, 1)] if False else x_vals[iz * (H * W) + iy * W + ix]
                    for kz in range(kd):
                        for ky in range(kh):
                            for kx in range(kw):
                                od = base_d + kz * dilation - padding
                                oh = base_y + ky * dilation - padding
                                ow = base_x + kx * dilation - padding
                                if 0 <= od < out_d and 0 <= oh < out_h and 0 <= ow < out_w:
                                    y[idx3(y, od, oh, ow, out_h, out_w)] += kernel_1d[(kz * kh + ky) * kw + kx] * val
        y = [v + bias for v in y]
        try:
            report("neuron", "convtranspose3d", {"inDHW": [D, H, W], "out": len(y), "kDHW": [kd, kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y)
        except Exception:
            return y


try:
    from .plugins.conv_transpose1d import ConvTranspose1DNeuronPlugin as _CT1
    from .plugins.conv_transpose2d import ConvTranspose2DNeuronPlugin as _CT2
    from .plugins.conv_transpose3d import ConvTranspose3DNeuronPlugin as _CT3
    register_neuron_type("conv_transpose1d", _CT1())
    register_neuron_type("conv_transpose2d", _CT2())
    register_neuron_type("conv_transpose3d", _CT3())
except Exception:
    pass

__all__ += [
    "ConvTranspose1DNeuronPlugin",
    "ConvTranspose2DNeuronPlugin",
    "ConvTranspose3DNeuronPlugin",
]


# -----------------------------
# Neuron Plugins: MaxPool1D/2D/3D
# -----------------------------

class MaxPool1DNeuronPlugin(Conv1DNeuronPlugin):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxPool1D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxpool1d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 3:
            raise ValueError("MaxPool1D requires 3 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src = [s.source for s in param_incs[:3]]
        # Allow learnable hyperparameters (stored as floats, later coerced to ints)
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        lk = lstore.get("kernel_size")
        ls = lstore.get("stride")
        lp = lstore.get("padding")
        try:
            ksize = int(max(1, round(float(lk.detach().to("cpu").view(-1)[0].item())))) if hasattr(lk, "detach") else int(max(1, round(base_ks)))
        except Exception:
            ksize = int(max(1, round(base_ks)))
        try:
            stride = int(max(1, round(float(ls.detach().to("cpu").view(-1)[0].item())))) if hasattr(ls, "detach") else int(max(1, round(base_st)))
        except Exception:
            stride = int(max(1, round(base_st)))
        try:
            padding = int(max(0, round(float(lp.detach().to("cpu").view(-1)[0].item())))) if hasattr(lp, "detach") else int(max(0, round(base_pd)))
        except Exception:
            padding = int(max(0, round(base_pd)))

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            x1 = []
            for s in data_incs:
                x1 += self._to_list1d(getattr(s.source, "tensor", []))
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x1 = self._to_list1d(x)

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None:
            try:
                xt = torch.tensor(x1, dtype=torch.float32, device=device).view(1, 1, -1)
                y = torch.nn.functional.max_pool1d(xt, kernel_size=ksize, stride=stride, padding=padding)
                y = y.view(-1)
                try:
                    report("neuron", "maxpool1d", {"in": int(xt.numel()), "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure Python fallback using -inf as padding
        if padding > 0:
            x1 = ([float("-inf")] * padding) + x1 + ([float("-inf")] * padding)
        n = len(x1)
        out_len = 0 if n < ksize else 1 + (n - ksize) // stride
        y_list = []
        for t in range(out_len):
            base = t * stride
            y_list.append(max(x1[base: base + ksize]))
        try:
            report("neuron", "maxpool1d", {"in": len(x1), "out": len(y_list), "k": ksize, "stride": stride, "pad": padding}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


class MaxPool2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxPool2D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxpool2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 3:
            raise ValueError("MaxPool2D requires 3 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src = [s.source for s in param_incs[:3]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        lk = lstore.get("kernel_size")
        ls = lstore.get("stride")
        lp = lstore.get("padding")
        try:
            ksize = int(max(1, round(float(lk.detach().to("cpu").view(-1)[0].item())))) if hasattr(lk, "detach") else int(max(1, round(base_ks)))
        except Exception:
            ksize = int(max(1, round(base_ks)))
        try:
            stride = int(max(1, round(float(ls.detach().to("cpu").view(-1)[0].item())))) if hasattr(ls, "detach") else int(max(1, round(base_st)))
        except Exception:
            stride = int(max(1, round(base_st)))
        try:
            padding = int(max(0, round(float(lp.detach().to("cpu").view(-1)[0].item())))) if hasattr(lp, "detach") else int(max(0, round(base_pd)))
        except Exception:
            padding = int(max(0, round(base_pd)))

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            rows = [self._to_list1d(getattr(s.source, "tensor", [])) for s in data_incs]
            width = min((len(r) for r in rows if r), default=0)
            if width <= 0:
                x_vals = []
                H = W = 0
            else:
                rows = [r[:width] for r in rows]
                H = len(rows)
                W = width
                x_vals = [v for r in rows for v in r]
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x_vals = self._to_list1d(x)
            N = max(1, len(x_vals))
            rh = int(math.isqrt(N))
            if rh * rh == N:
                H = W = rh
            else:
                H, W = N, 1

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0:
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, H, W)
                y = torch.nn.functional.max_pool2d(xt, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(padding, padding))
                y = y.view(-1)
                try:
                    report("neuron", "maxpool2d", {"inHW": [H, W], "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        if H <= 0 or W <= 0:
            return neuron._ensure_tensor([]) if hasattr(neuron, "_ensure_tensor") else []
        if padding > 0:
            padded = []
            zero_row = [float("-inf")] * (W + 2 * padding)
            for _ in range(padding):
                padded.append(list(zero_row))
            for r_ in range(H):
                row = [float("-inf")] * padding + x_vals[r_ * W:(r_ + 1) * W] + [float("-inf")] * padding
                padded.append(row)
            for _ in range(padding):
                padded.append(list(zero_row))
            H2, W2 = len(padded), len(padded[0])
        else:
            padded = [x_vals[r_ * W:(r_ + 1) * W] for r_ in range(H)]
            H2, W2 = H, W
        out_h = 0 if H2 < ksize else 1 + (H2 - ksize) // stride
        out_w = 0 if W2 < ksize else 1 + (W2 - ksize) // stride
        y2 = []
        for oy in range(out_h):
            base_y = oy * stride
            for ox in range(out_w):
                base_x = ox * stride
                m = float("-inf")
                for ky in range(ksize):
                    for kx in range(ksize):
                        vy = padded[base_y + ky][base_x + kx]
                        if vy > m:
                            m = vy
                y2.append(m)
        try:
            report("neuron", "maxpool2d", {"inHW": [H, W], "out": len(y2), "k": ksize, "stride": stride, "pad": padding}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y2)
        except Exception:
            return y2


class MaxPool3DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxPool3D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxpool3d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 3:
            raise ValueError("MaxPool3D requires 3 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src = [s.source for s in param_incs[:3]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        lk = lstore.get("kernel_size")
        ls = lstore.get("stride")
        lp = lstore.get("padding")
        try:
            ksize = int(max(1, round(float(lk.detach().to("cpu").view(-1)[0].item())))) if hasattr(lk, "detach") else int(max(1, round(base_ks)))
        except Exception:
            ksize = int(max(1, round(base_ks)))
        try:
            stride = int(max(1, round(float(ls.detach().to("cpu").view(-1)[0].item())))) if hasattr(ls, "detach") else int(max(1, round(base_st)))
        except Exception:
            stride = int(max(1, round(base_st)))
        try:
            padding = int(max(0, round(float(lp.detach().to("cpu").view(-1)[0].item())))) if hasattr(lp, "detach") else int(max(0, round(base_pd)))
        except Exception:
            padding = int(max(0, round(base_pd)))

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            slices = []
            dims = []
            for s in data_incs:
                vals = self._to_list1d(getattr(s.source, "tensor", []))
                N = max(1, len(vals))
                r = int(math.isqrt(N))
                if r * r == N:
                    H = W = r
                else:
                    H, W = N, 1
                dims.append((H, W))
                slices.append(vals[: H * W])
            Hmin = min((h for h, _ in dims), default=0)
            Wmin = min((w for _, w in dims), default=0)
            x_vals = []
            D = len(slices)
            for sl, (h, w) in zip(slices, dims):
                for rr in range(Hmin):
                    row = sl[rr * w:(rr + 1) * w]
                    x_vals.extend(row[:Wmin])
            H, W = Hmin, Wmin
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            vals = self._to_list1d(x)
            N = max(1, len(vals))
            r3 = round(N ** (1.0 / 3.0))
            if r3 > 0 and (r3 * r3 * r3) == N:
                D = H = W = int(r3)
                x_vals = vals[: D * H * W]
            else:
                D, H, W = N, 1, 1
                x_vals = vals

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0:
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, D, H, W)
                y = torch.nn.functional.max_pool3d(xt, kernel_size=(ksize, ksize, ksize), stride=(stride, stride, stride), padding=(padding, padding, padding))
                y = y.view(-1)
                try:
                    report("neuron", "maxpool3d", {"inDHW": [D, H, W], "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure Python fallback with -inf padding
        if padding > 0 and H > 0 and W > 0:
            D2, H2, W2 = D + 2 * padding, H + 2 * padding, W + 2 * padding
            padval = float("-inf")
            padded = [padval] * (D2 * H2 * W2)
            for dz in range(D):
                for yy in range(H):
                    for xx in range(W):
                        pd = dz + padding
                        py = yy + padding
                        px = xx + padding
                        padded[pd * (H2 * W2) + py * W2 + px] = x_vals[dz * (H * W) + yy * W + xx]
        else:
            padded = list(x_vals)
            D2, H2, W2 = D, H, W
        out_d = 0 if D2 < ksize else 1 + (D2 - ksize) // stride
        out_h = 0 if H2 < ksize else 1 + (H2 - ksize) // stride
        out_w = 0 if W2 < ksize else 1 + (W2 - ksize) // stride
        y_list = []
        for od in range(out_d):
            base_d = od * stride
            for oy in range(out_h):
                base_y = oy * stride
                for ox in range(out_w):
                    base_x = ox * stride
                    m = float("-inf")
                    for kz in range(ksize):
                        for ky in range(ksize):
                            for kx in range(ksize):
                                val = padded[(base_d + kz) * (H2 * W2) + (base_y + ky) * W2 + (base_x + kx)]
                                if val > m:
                                    m = val
                    y_list.append(m)
        try:
            report("neuron", "maxpool3d", {"inDHW": [D, H, W], "out": len(y_list), "k": ksize, "stride": stride, "pad": padding}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


try:
    from .plugins.maxpool1d import MaxPool1DNeuronPlugin as _MP1
    from .plugins.maxpool2d import MaxPool2DNeuronPlugin as _MP2
    from .plugins.maxpool3d import MaxPool3DNeuronPlugin as _MP3
    register_neuron_type("maxpool1d", _MP1())
    register_neuron_type("maxpool2d", _MP2())
    register_neuron_type("maxpool3d", _MP3())
except Exception:
    pass

__all__ += [
    "MaxPool1DNeuronPlugin",
    "MaxPool2DNeuronPlugin",
    "MaxPool3DNeuronPlugin",
]


# -----------------------------
# Neuron Plugins: Unfold2D and Fold2D
# -----------------------------

class Unfold2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 4 or len(out) != 1:
            raise ValueError(
                f"Unfold2D neuron requires exactly 4 incoming PARAM synapses (kernel,stride,padding,dilation) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "unfold2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 4:
            raise ValueError("Unfold2D requires 4 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src = [s.source for s in param_incs[:4]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        base_dl = self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        lk = lstore.get("kernel_size"); ls = lstore.get("stride"); lp = lstore.get("padding"); ld = lstore.get("dilation")
        def to_int(val, default, minv):
            try:
                return int(max(minv, round(float(val.detach().to("cpu").view(-1)[0].item())))) if hasattr(val, "detach") else int(max(minv, round(default)))
            except Exception:
                return int(max(minv, round(default)))
        ksize = to_int(lk, base_ks, 1)
        stride = to_int(ls, base_st, 1)
        padding = to_int(lp, base_pd, 0)
        dilation = to_int(ld, base_dl, 1)

        # Build 2D input from DATA rows
        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            rows = [self._to_list1d(getattr(s.source, "tensor", [])) for s in data_incs]
            width = min((len(r) for r in rows if r), default=0)
            if width <= 0:
                x_vals = []
                H = W = 0
            else:
                rows = [r[:width] for r in rows]
                H = len(rows)
                W = width
                x_vals = [v for r in rows for v in r]
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x_vals = self._to_list1d(x)
            N = max(1, len(x_vals))
            r = int(math.isqrt(N))
            H = W = r if r * r == N else (N, 1)
            if isinstance(H, tuple):
                H, W = H

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0:
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, H, W)
                y = torch.nn.functional.unfold(xt, kernel_size=(ksize, ksize), dilation=(dilation, dilation), padding=(padding, padding), stride=(stride, stride))
                y = y.view(-1)
                try:
                    report("neuron", "unfold2d", {"inHW": [H, W], "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass
        # Pure-Python unfold
        if H <= 0 or W <= 0:
            return neuron._ensure_tensor([]) if hasattr(neuron, "_ensure_tensor") else []
        # Apply zero padding
        if padding > 0:
            padded = []
            zero_row = [0.0] * (W + 2 * padding)
            for _ in range(padding):
                padded.append(list(zero_row))
            for r_ in range(H):
                row = [0.0] * padding + x_vals[r_ * W:(r_ + 1) * W] + [0.0] * padding
                padded.append(row)
            for _ in range(padding):
                padded.append(list(zero_row))
            H2, W2 = len(padded), len(padded[0])
        else:
            padded = [x_vals[r_ * W:(r_ + 1) * W] for r_ in range(H)]
            H2, W2 = H, W
        out_h = 0 if H2 < ((ksize - 1) * dilation + 1) else 1 + (H2 - ((ksize - 1) * dilation + 1)) // stride
        out_w = 0 if W2 < ((ksize - 1) * dilation + 1) else 1 + (W2 - ((ksize - 1) * dilation + 1)) // stride
        cols: List[float] = []
        for oy in range(out_h):
            base_y = oy * stride
            for ox in range(out_w):
                base_x = ox * stride
                for ky in range(ksize):
                    for kx in range(ksize):
                        iy = base_y + ky * dilation
                        ix = base_x + kx * dilation
                        cols.append(padded[iy][ix])
        try:
            report("neuron", "unfold2d", {"inHW": [H, W], "out": len(cols), "k": ksize, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        return neuron._ensure_tensor(cols)


class Fold2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 6 or len(out) != 1:
            raise ValueError(
                f"Fold2D neuron requires exactly 6 incoming PARAM synapses (out_h,out_w,kernel,stride,padding,dilation) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "fold2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 6:
            raise ValueError("Fold2D requires 6 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        oh_src, ow_src, k_src, s_src, p_src, d_src = [s.source for s in param_incs[:6]]
        base_oh = self._first_scalar(getattr(oh_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_ow = self._first_scalar(getattr(ow_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        base_dl = self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        def to_int(val, default, minv):
            try:
                return int(max(minv, round(float(val.detach().to("cpu").view(-1)[0].item())))) if hasattr(val, "detach") else int(max(minv, round(default)))
            except Exception:
                return int(max(minv, round(default)))
        out_h = to_int(lstore.get("out_h"), base_oh, 1)
        out_w = to_int(lstore.get("out_w"), base_ow, 1)
        ksize = to_int(lstore.get("kernel_size"), base_ks, 1)
        stride = to_int(lstore.get("stride"), base_st, 1)
        padding = to_int(lstore.get("padding"), base_pd, 0)
        dilation = to_int(lstore.get("dilation"), base_dl, 1)

        # Build columns input from DATA sources (concatenate into 1D)
        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            cols = []
            for s in data_incs:
                cols += self._to_list1d(getattr(s.source, "tensor", []))
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            cols = self._to_list1d(x)

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        # Expect length multiple of ksize*ksize
        kk = max(1, ksize * ksize)
        L = len(cols) // kk if len(cols) >= kk else 0
        if torch is not None and L > 0:
            try:
                ct = torch.tensor(cols[: L * kk], dtype=torch.float32, device=device).view(1, 1 * kk, L)
                y = torch.nn.functional.fold(ct, output_size=(out_h, out_w), kernel_size=(ksize, ksize), dilation=(dilation, dilation), padding=(padding, padding), stride=(stride, stride))
                y = y.view(-1)
                try:
                    report("neuron", "fold2d", {"outHW": [out_h, out_w], "in_cols": L, "k": ksize, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass
        # Pure-Python simple overlap-add
        if L <= 0:
            return neuron._ensure_tensor([]) if hasattr(neuron, "_ensure_tensor") else []
        out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
        idx = 0
        # Compute number of sliding positions from out_h/out_w, stride/dilation/padding heuristic
        # We place windows in raster order
        # Determine number of windows that fits given stride
        def npos(size):
            # minimal model: try as many as fit when starting at 0
            cnt = 0
            pos = 0
            span = (ksize - 1) * dilation + 1
            while pos + span <= size:
                cnt += 1
                pos += stride
            return cnt
        nh = npos(out_h)
        nw = npos(out_w)
        total = nh * nw
        used = min(L, total)
        for oy in range(nh):
            base_y = oy * stride
            for ox in range(nw):
                if idx >= used:
                    break
                base_x = ox * stride
                # one column corresponds to ksize*ksize elements
                for ky in range(ksize):
                    for kx in range(ksize):
                        iy = base_y + ky * dilation
                        ix = base_x + kx * dilation
                        if 0 <= iy < out_h and 0 <= ix < out_w:
                            out[iy][ix] += cols[idx * kk + ky * ksize + kx]
                idx += 1
        flat = [v for row in out for v in row]
        try:
            report("neuron", "fold2d", {"outHW": [out_h, out_w], "in_cols": used, "k": ksize, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        return neuron._ensure_tensor(flat)


try:
    from .plugins.unfold2d import Unfold2DNeuronPlugin as _UF
    from .plugins.fold2d import Fold2DNeuronPlugin as _FD
    register_neuron_type("unfold2d", _UF())
    register_neuron_type("fold2d", _FD())
except Exception:
    pass

__all__ += ["Unfold2DNeuronPlugin", "Fold2DNeuronPlugin"]


# -----------------------------
# Neuron Plugins: MaxUnpool1D/2D/3D
# -----------------------------

class MaxUnpool1DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxUnpool1D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxunpool1d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src = [s.source for s in param_incs[:3]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        def to_int(val, default, minv):
            try:
                return int(max(minv, round(float(val.detach().to("cpu").view(-1)[0].item())))) if hasattr(val, "detach") else int(max(minv, round(default)))
            except Exception:
                return int(max(minv, round(default)))
        ksize = to_int(lstore.get("kernel_size"), base_ks, 1)
        stride = to_int(lstore.get("stride"), base_st, 1)
        padding = to_int(lstore.get("padding"), base_pd, 0)

        # DATA sources: first = values, second = indices
        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        data_incs.sort(key=self._key_src)
        vals = self._to_list1d(getattr(data_incs[0].source, "tensor", [])) if len(data_incs) >= 1 else self._to_list1d(input_value)
        idxs = self._to_list1d(getattr(data_incs[1].source, "tensor", [])) if len(data_incs) >= 2 else []

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and vals and str(device) == "cuda":
            try:
                vt = torch.tensor(vals, dtype=torch.float32, device=device).view(1, 1, -1)
                it = torch.tensor([int(i) for i in idxs[: len(vals)]], dtype=torch.long, device=device).view(1, 1, -1)
                out_len = (vt.shape[-1] - 1) * stride - 2 * padding + ksize
                y = torch.nn.functional.max_unpool1d(vt, it, kernel_size=ksize, stride=stride, padding=padding, output_size=(1, 1, int(out_len)))
                y = y.view(-1)
                try:
                    report("neuron", "maxunpool1d", {"in": int(vt.numel()), "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass
        # Fallback: return values unchanged
        return neuron._ensure_tensor(vals)


class MaxUnpool2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxUnpool2D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxunpool2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src = [s.source for s in param_incs[:3]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        def to_int(val, default, minv):
            try:
                return int(max(minv, round(float(val.detach().to("cpu").view(-1)[0].item())))) if hasattr(val, "detach") else int(max(minv, round(default)))
            except Exception:
                return int(max(minv, round(default)))
        ksize = to_int(lstore.get("kernel_size"), base_ks, 1)
        stride = to_int(lstore.get("stride"), base_st, 1)
        padding = to_int(lstore.get("padding"), base_pd, 0)

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        data_incs.sort(key=self._key_src)
        vals = []
        idxs = []
        if len(data_incs) >= 1:
            vals = self._to_list1d(getattr(data_incs[0].source, "tensor", []))
        if len(data_incs) >= 2:
            idxs = self._to_list1d(getattr(data_incs[1].source, "tensor", []))
        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and vals and str(device) == "cuda":
            try:
                # Infer pooled H,W as square if possible
                N = len(vals)
                r = int(math.isqrt(N))
                H = W = r if r * r == N else (N, 1)
                if isinstance(H, tuple):
                    H, W = H
                vt = torch.tensor(vals, dtype=torch.float32, device=device).view(1, 1, H, W)
                it = torch.tensor([int(i) for i in idxs[: len(vals)]], dtype=torch.long, device=device).view(1, 1, H, W)
                out_h = (H - 1) * stride - 2 * padding + ksize
                out_w = (W - 1) * stride - 2 * padding + ksize
                y = torch.nn.functional.max_unpool2d(vt, it, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(padding, padding), output_size=(1, 1, int(out_h), int(out_w)))
                y = y.view(-1)
                try:
                    report("neuron", "maxunpool2d", {"inHW": [H, W], "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass
        return neuron._ensure_tensor(vals)


class MaxUnpool3DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxUnpool3D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxunpool3d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src = [s.source for s in param_incs[:3]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        def to_int(val, default, minv):
            try:
                return int(max(minv, round(float(val.detach().to("cpu").view(-1)[0].item())))) if hasattr(val, "detach") else int(max(minv, round(default)))
            except Exception:
                return int(max(minv, round(default)))
        ksize = to_int(lstore.get("kernel_size"), base_ks, 1)
        stride = to_int(lstore.get("stride"), base_st, 1)
        padding = to_int(lstore.get("padding"), base_pd, 0)

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        data_incs.sort(key=self._key_src)
        vals = []
        idxs = []
        if len(data_incs) >= 1:
            vals = self._to_list1d(getattr(data_incs[0].source, "tensor", []))
        if len(data_incs) >= 2:
            idxs = self._to_list1d(getattr(data_incs[1].source, "tensor", []))
        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and vals:
            try:
                N = len(vals)
                r = int(round(N ** (1.0 / 3.0)))
                if r > 0 and r * r * r == N:
                    D = H = W = r
                else:
                    D, H, W = N, 1, 1
                vt = torch.tensor(vals, dtype=torch.float32, device=device).view(1, 1, D, H, W)
                it = torch.tensor([int(i) for i in idxs[: len(vals)]], dtype=torch.long, device=device).view(1, 1, D, H, W)
                out_d = (D - 1) * stride - 2 * padding + ksize
                out_h = (H - 1) * stride - 2 * padding + ksize
                out_w = (W - 1) * stride - 2 * padding + ksize
                y = torch.nn.functional.max_unpool3d(
                    vt, it, kernel_size=(ksize, ksize, ksize), stride=(stride, stride, stride), padding=(padding, padding, padding),
                    output_size=(1, 1, int(out_d), int(out_h), int(out_w))
                )
                y = y.view(-1)
                try:
                    report("neuron", "maxunpool3d", {"inDHW": [D, H, W], "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass
        return neuron._ensure_tensor(vals)


try:
    from .plugins.maxunpool1d import MaxUnpool1DNeuronPlugin as _MUP1
    from .plugins.maxunpool2d import MaxUnpool2DNeuronPlugin as _MUP2
    from .plugins.maxunpool3d import MaxUnpool3DNeuronPlugin as _MUP3
    register_neuron_type("maxunpool1d", _MUP1())
    register_neuron_type("maxunpool2d", _MUP2())
    register_neuron_type("maxunpool3d", _MUP3())
except Exception:
    pass

__all__ += [
    "MaxUnpool1DNeuronPlugin",
    "MaxUnpool2DNeuronPlugin",
    "MaxUnpool3DNeuronPlugin",
]

# -----------------------------
# Learning Paradigm Plugins (Brain-level orchestration)
# -----------------------------

_PARADIGM_TYPES: Dict[str, Any] = {}


def register_learning_paradigm_type(name: str, plugin: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Learning paradigm type name must be a non-empty string")
    mod = getattr(getattr(plugin, "__class__", object), "__module__", "")
    if isinstance(mod, str) and mod.startswith("marble.") and not mod.startswith("marble.plugins."):
        raise ValueError(f"Learning paradigm '{name}' must be in marble.plugins.*; got module '{mod}'")
    _PARADIGM_TYPES[name] = plugin


class _AdaptiveLRRoutine:
    def __init__(self, factor_down: float = 0.5, factor_up: float = 1.1, min_lr: float = 1e-5, max_lr: float = 1.0):
        self.factor_down = float(factor_down)
        self.factor_up = float(factor_up)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self._last_loss = None

    def on_init(self, selfattention: "SelfAttention") -> None:
        pass

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        try:
            cur = float(ctx.get("cur_loss_tensor").detach().to("cpu").item()) if ctx.get("cur_loss_tensor") is not None else None
        except Exception:
            cur = None
        # Determine base lr
        base_lr = getattr(wanderer, "current_lr", None)
        try:
            base_lr = float(base_lr) if base_lr is not None else 1e-2
        except Exception:
            base_lr = 1e-2
        new_lr = base_lr
        if cur is not None:
            if self._last_loss is not None and cur > self._last_loss:
                new_lr = max(self.min_lr, base_lr * self.factor_down)
            elif self._last_loss is not None and cur <= self._last_loss:
                new_lr = min(self.max_lr, base_lr * self.factor_up)
        self._last_loss = cur if cur is not None else self._last_loss
        try:
            selfattention.set_param("lr_override", new_lr)
        except Exception:
            pass
        return None


class AdaptiveLRParadigm:
    """Example learning paradigm that wires a SelfAttention routine to adapt LR.

    Hooks used:
    - Registers no new neuron types but leverages SelfAttention to adjust Wanderer settings.
    - Exposes on_wanderer hook to attach the routine to any created Wanderer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.routine = _AdaptiveLRRoutine(
            factor_down=float(cfg.get("factor_down", 0.5)),
            factor_up=float(cfg.get("factor_up", 1.1)),
            min_lr=float(cfg.get("min_lr", 1e-5)),
            max_lr=float(cfg.get("max_lr", 1.0)),
        )
        self._selfattention = SelfAttention(routines=[self.routine])

    def on_wanderer(self, wanderer: "Wanderer") -> None:
        try:
            attach_selfattention(wanderer, self._selfattention)
            report("training", "paradigm_attach", {"type": "adaptive_lr"}, "events")
        except Exception:
            pass


try:
    register_learning_paradigm_type("adaptive_lr", AdaptiveLRParadigm)
except Exception:
    pass

def add_paradigm(brain: "Brain", name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Load and attach a learning paradigm onto a Brain (alias to Brain.load_paradigm)."""
    return brain.load_paradigm(name, config)


def ensure_paradigm_loaded(brain: "Brain", name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Idempotently load a paradigm by name; if an instance of the same class is present, return it."""
    plug = _PARADIGM_TYPES.get(str(name))
    if plug is None:
        raise ValueError(f"Unknown learning paradigm: {name}")
    cls = plug if isinstance(plug, type) else plug.__class__
    for obj in getattr(brain, "_paradigms", []) or []:
        try:
            if isinstance(obj, cls):
                return obj
        except Exception:
            continue
    return brain.load_paradigm(name, config)


def list_paradigms(brain: "Brain") -> List[Dict[str, Any]]:
    """List paradigms loaded on this Brain with basic metadata."""
    out: List[Dict[str, Any]] = []
    for obj in getattr(brain, "_paradigms", []) or []:
        try:
            out.append({
                "id": id(obj),
                "class": obj.__class__.__name__,
                "module": getattr(obj.__class__, "__module__", None),
            })
        except Exception:
            out.append({"id": id(obj)})
    return out


def remove_paradigm(brain: "Brain", name_or_obj: Any) -> bool:
    """Remove a paradigm by name or object instance; returns True if removed."""
    lst = getattr(brain, "_paradigms", []) or []
    if isinstance(name_or_obj, str):
        target_cls = _PARADIGM_TYPES.get(name_or_obj)
        if target_cls is None:
            return False
        target_cls = target_cls if isinstance(target_cls, type) else target_cls.__class__
        for i, obj in enumerate(lst):
            try:
                if isinstance(obj, target_cls):
                    del lst[i]
                    return True
            except Exception:
                continue
        return False
    else:
        for i, obj in enumerate(lst):
            if obj is name_or_obj:
                del lst[i]
                return True
        return False


def apply_paradigms_to_wanderer(brain: "Brain", wanderer: "Wanderer") -> None:
    """Invoke `on_wanderer` on all enabled paradigms for explicit wiring."""
    act = brain.active_paradigms() if hasattr(brain, "active_paradigms") else (getattr(brain, "_paradigms", []) or [])
    for paradigm in act:
        try:
            if hasattr(paradigm, "on_wanderer"):
                paradigm.on_wanderer(wanderer)
        except Exception:
            pass


from .wanderer import push_temporary_plugins, pop_temporary_plugins


__all__ += [
    "register_learning_paradigm_type",
    "AdaptiveLRParadigm",
    "add_paradigm",
    "ensure_paradigm_loaded",
    "list_paradigms",
    "remove_paradigm",
    "apply_paradigms_to_wanderer",
]


# -----------------------------
# Synapse Plugins: Noisy Transmission
# -----------------------------

from .plugins.synapse_noisy import NoisySynapsePlugin as _NoisySynapsePlugin  # noqa: F401


# -----------------------------
# Wanderer Plugins: L2 Weight Penalty
# -----------------------------

class L2WeightPenaltyPlugin:
    """Adds L2 penalty over visited neurons' weights and biases to the loss.

    Reads lambda from wanderer._neuro_cfg['l2_lambda'] (default 0.0).
    """

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch = getattr(wanderer, "_torch", None)
        lam = float(getattr(wanderer, "_neuro_cfg", {}).get("l2_lambda", 0.0))
        if lam <= 0.0 or torch is None:
            # No contribution
            return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        total = None
        for n in getattr(wanderer, "_visited", []) or []:
            try:
                w_param, b_param = wanderer._param_map[id(n)]
                term = (w_param.view(-1) ** 2).sum() + (b_param.view(-1) ** 2).sum()
                total = term if total is None else (total + term)
            except Exception:
                continue
        if total is None:
            return torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        return lam * total


try:
    register_wanderer_type("l2_weight_penalty", L2WeightPenaltyPlugin())
    __all__ += ["L2WeightPenaltyPlugin"]
except Exception:
    pass


# -----------------------------
# Wanderer Plugins: Contrastive, TD Q-Learning, Distillation
# -----------------------------

class ContrastiveInfoNCEPlugin:
    """Adds an InfoNCE-style contrastive loss across walk outputs.

    Config via wanderer._neuro_cfg:
      - contrastive_tau (temperature, default 0.1)
      - contrastive_lambda (weight, default 1.0)
    Positives are adjacent outputs in the same walk; negatives are all other outputs.
    """

    def _normalize(self, torch, x):
        x = x.view(-1)
        n = x.norm(p=2) + 1e-8
        return x / n

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch = getattr(wanderer, "_torch", None)
        if torch is None or len(outputs) < 2:
            return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        tau = float(getattr(wanderer, "_neuro_cfg", {}).get("contrastive_tau", 0.1))
        w = float(getattr(wanderer, "_neuro_cfg", {}).get("contrastive_lambda", 1.0))
        dev = getattr(wanderer, "_device", "cpu")
        # Build matrix of normalized embeddings
        vecs = []
        for y in outputs:
            if hasattr(y, "detach"):
                v = y.detach().to(dev).float().view(-1)
            else:
                v = torch.tensor([float(vv) for vv in (y if isinstance(y, (list, tuple)) else [y])], dtype=torch.float32, device=dev)
            vecs.append(self._normalize(torch, v))
        X = torch.stack(vecs, dim=0)  # [T, D]
        # Similarities
        S = X @ X.t()  # [T, T]
        S = S / max(1e-8, float(tau))
        # For each i>=1, positive is (i, i-1)
        loss_terms = []
        for i in range(1, X.shape[0]):
            logits = S[i]
            pos = logits[i - 1]
            # Exclude self from denominator by masking
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[i] = False
            denom = torch.logsumexp(logits[mask], dim=0)
            loss_i = - (pos - denom)
            loss_terms.append(loss_i)
        if not loss_terms:
            return torch.tensor(0.0, device=dev)
        return w * (torch.stack(loss_terms).mean())


class TDQLearningPlugin:
    """Tabular TD(0) Q-learning over synapses stored in synapse._plugin_state['q'].

    - choose_next: epsilon-greedy over Q(synapse) for available choices.
    - on_step: TD update for the previous chosen synapse using reward = -current per-step loss.
    Config via wanderer._neuro_cfg: rl_epsilon (0.1), rl_alpha (0.1), rl_gamma (0.9)
    """

    def __init__(self) -> None:
        self._last_syn: Optional["Synapse"] = None

    def _q(self, syn: "Synapse") -> float:
        st = getattr(syn, "_plugin_state", None)
        if st is None:
            syn._plugin_state = {}
            st = syn._plugin_state
        q = st.get("q", 0.0)
        try:
            return float(q)
        except Exception:
            return 0.0

    def _set_q(self, syn: "Synapse", q: float) -> None:
        st = getattr(syn, "_plugin_state", None)
        if st is None:
            syn._plugin_state = {}
            st = syn._plugin_state
        st["q"] = float(q)

    def choose_next(self, wanderer: "Wanderer", current: "Neuron", choices: List[Tuple["Synapse", str]]):
        import random as _r
        if not choices:
            return None, "forward"
        eps = float(getattr(wanderer, "_neuro_cfg", {}).get("rl_epsilon", 0.1))
        if _r.random() < eps:
            return choices[_r.randrange(len(choices))]
        # Greedy by Q
        best = choices[0]
        best_q = self._q(best[0])
        for s, d in choices[1:]:
            q = self._q(s)
            if q > best_q:
                best = (s, d); best_q = q
        # Track chosen for TD update on next step
        self._last_syn = best[0]
        return best

    def on_step(self, wanderer: "Wanderer", current: "Neuron", next_syn: Optional["Synapse"], direction: str, step_index: int, out_value: Any) -> None:
        # TD update for previously chosen synapse using reward from current step loss
        if self._last_syn is None:
            return
        try:
            cur_loss_t = wanderer._walk_ctx.get("cur_loss_tensor")  # type: ignore[attr-defined]
            r = -float(cur_loss_t.detach().to("cpu").item()) if cur_loss_t is not None else 0.0
        except Exception:
            r = 0.0
        alpha = float(getattr(wanderer, "_neuro_cfg", {}).get("rl_alpha", 0.1))
        gamma = float(getattr(wanderer, "_neuro_cfg", {}).get("rl_gamma", 0.9))
        # Estimate max_a' Q(s',a') from the NEXT state's outgoing choices
        max_next = 0.0
        try:
            if next_syn is not None:
                next_node = next_syn.target if direction == "forward" else next_syn.source
            else:
                next_node = current
            choices = wanderer._gather_choices(next_node)
            if choices:
                max_next = max(self._q(s) for s, _ in choices)
        except Exception:
            pass
        q = self._q(self._last_syn)
        td_target = r + gamma * max_next
        new_q = q + alpha * (td_target - q)
        self._set_q(self._last_syn, new_q)
        try:
            report("wanderer", "td_q_update", {"q": new_q, "r": r}, "plugins")
        except Exception:
            pass
        # Reset last_syn only after applying update once; keep it tied to previous transition
        self._last_syn = next_syn


class DistillationPlugin:
    """Adds a simple distillation loss to match a moving-average 'teacher' of outputs.

    Config via wanderer._neuro_cfg:
      - distill_lambda (default 0.1)
      - teacher_momentum (EMA, default 0.9)
    """

    def __init__(self) -> None:
        self._ema: Optional[Any] = None

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch = getattr(wanderer, "_torch", None)
        if torch is None or not outputs:
            return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        dev = getattr(wanderer, "_device", "cpu")
        lam = float(getattr(wanderer, "_neuro_cfg", {}).get("distill_lambda", 0.1))
        mom = float(getattr(wanderer, "_neuro_cfg", {}).get("teacher_momentum", 0.9))
        # Use the last output for current prediction
        y = outputs[-1]
        if not hasattr(y, "detach"):
            y = torch.tensor([float(v) for v in (y if isinstance(y, (list, tuple)) else [y])], dtype=torch.float32, device=dev)
        yt = y.detach().to(dev).float().view(-1)
        if self._ema is None:
            self._ema = yt.detach()
        else:
            self._ema = (mom * self._ema) + ((1.0 - mom) * yt.detach())
        # MSE between current y and teacher ema
        if self._ema.shape != yt.shape:
            # Align to min length
            m = min(int(self._ema.numel()), int(yt.numel()))
            ema = self._ema.view(-1)[:m]
            yv = yt.view(-1)[:m]
        else:
            ema = self._ema
            yv = yt
        return lam * ((yv - ema).pow(2).mean())


try:
    register_wanderer_type("contrastive_infonce", ContrastiveInfoNCEPlugin())
    register_wanderer_type("td_qlearning", TDQLearningPlugin())
    register_wanderer_type("distillation", DistillationPlugin())
    __all__ += ["ContrastiveInfoNCEPlugin", "TDQLearningPlugin", "DistillationPlugin"]
except Exception:
    pass


# -----------------------------
# SelfAttention Routine: Adaptive Grad Clip
# -----------------------------

class AdaptiveGradClipRoutine:
    """Adjusts gradient clipping based on step loss spikes.

    If current per-step loss grows by more than `threshold_ratio` over the
    previous step, set gradient clipping (method='norm', max_norm).
    Fields (constructor/defaults): threshold_ratio=1.5, max_norm=1.0, cooldown=5
    """

    def __init__(self, threshold_ratio: float = 1.5, max_norm: float = 1.0, cooldown: int = 5) -> None:
        self.threshold_ratio = float(threshold_ratio)
        self.max_norm = float(max_norm)
        self.cooldown = int(max(0, cooldown))
        self._since = 0

    def on_init(self, selfattention: "SelfAttention") -> None:
        self._since = 0

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        try:
            cur = float(ctx.get("cur_loss_tensor").detach().to("cpu").item()) if ctx.get("cur_loss_tensor") is not None else None
        except Exception:
            cur = None
        prev = None
        try:
            last = reporter_ro.item(f"step_{max(1, getattr(wanderer, '_global_step_counter', 1)) - 1}", "wanderer_steps", "logs")
            if isinstance(last, dict) and ("current_loss" in last):
                prev = float(last.get("current_loss"))
        except Exception:
            prev = None
        if cur is None or prev is None:
            return None
        if prev <= 0.0:
            return None
        ratio = cur / prev
        if ratio >= self.threshold_ratio and self._since <= 0:
            try:
                selfattention.set_param("_grad_clip", {"method": "norm", "max_norm": float(self.max_norm), "norm_type": 2.0})
                report("selfattention", "gradclip_enable", {"ratio": ratio, "max_norm": float(self.max_norm)}, "events")
            except Exception:
                pass
            self._since = self.cooldown
        else:
            self._since = max(0, self._since - 1)
        return None


try:
    register_selfattention_type("adaptive_grad_clip", AdaptiveGradClipRoutine())
    __all__ += ["AdaptiveGradClipRoutine"]
except Exception:
    pass


# -----------------------------
# Brain Training Plugin: Warmup-Decay
# -----------------------------

_BRAIN_TRAIN_TYPES: Dict[str, Any] = globals().get("_BRAIN_TRAIN_TYPES", {})  # type: ignore[var-annotated]


def register_brain_train_type(name: str, plugin: Any) -> None:  # re-affirm in case not in scope here
    if not isinstance(name, str) or not name:
        raise ValueError("Brain train type name must be a non-empty string")
    mod = getattr(getattr(plugin, "__class__", object), "__module__", "")
    if isinstance(mod, str) and mod.startswith("marble.") and not mod.startswith("marble.plugins."):
        raise ValueError(f"Brain training plugin '{name}' must be in marble.plugins.*; got module '{mod}'")
    _BRAIN_TRAIN_TYPES[name] = plugin


class WarmupDecayTrainPlugin:
    """Per-walk scheduler that warms up LR then decays it, and grows steps.

    Config (in `Brain.train(..., type_name='warmup_decay', ...)` or stacked):
      - warmup_walks (int, default 3)
      - base_lr (float, default 1e-2)
      - peak_lr (float, default 5e-2)
      - decay (float, default 0.9) multiplicative each walk after warmup
      - start_steps (int, default 2) and step_increment (int, default 1)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.warmup = int(cfg.get("warmup_walks", 3))
        self.base_lr = float(cfg.get("base_lr", 1e-2))
        self.peak_lr = float(cfg.get("peak_lr", 5e-2))
        self.decay = float(cfg.get("decay", 0.9))
        self.start_steps = int(cfg.get("start_steps", 2))
        self.step_inc = int(cfg.get("step_increment", 1))

    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:
        try:
            report("training", "warmup_decay_init", {"warmup": self.warmup}, "brain")
        except Exception:
            pass

    def choose_start(self, brain: "Brain", wanderer: "Wanderer", i: int):
        return None

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        steps = self.start_steps + i * self.step_inc
        if i < self.warmup:
            # Linear warmup
            t = (i + 1) / float(max(1, self.warmup))
            lr = self.base_lr + t * (self.peak_lr - self.base_lr)
        else:
            # Exponential decay from peak
            k = i - self.warmup
            lr = self.peak_lr * (self.decay ** k)
        try:
            report("training", "warmup_decay_before", {"walk": i, "steps": steps, "lr": lr}, "brain")
        except Exception:
            pass
        return {"max_steps": int(max(1, steps)), "lr": float(lr)}

    def after_walk(self, brain: "Brain", wanderer: "Wanderer", i: int, stats: Dict[str, Any]) -> None:
        try:
            report("training", "warmup_decay_after", {"walk": i, "loss": stats.get("loss")}, "brain")
        except Exception:
            pass

    def on_end(self, brain: "Brain", wanderer: "Wanderer", history: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"warmup_decay": {"walks": len(history), "final_loss": history[-1].get("loss") if history else None}}


try:
    register_brain_train_type("warmup_decay", WarmupDecayTrainPlugin())
    __all__ += ["WarmupDecayTrainPlugin"]
except Exception:
    pass

# Example growth paradigm: grow a neuron when stuck (neuroplasticity-like behavior)
class GrowthParadigm:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.grow_on_step = bool(cfg.get("grow_on_step_when_stuck", True))
        self.grow_on_end = bool(cfg.get("grow_on_end_if_no_outgoing", True))
        self.max_new_per_walk = int(cfg.get("max_new_per_walk", 1))

    def _first_free(self, brain: "Brain", avoid: Optional[Any]) -> Optional[Any]:
        try:
            for idx in brain.available_indices():
                if idx != avoid and brain.get_neuron(idx) is None:
                    return idx
        except Exception:
            pass
        # Try grid neighbors around avoid
        try:
            pos = tuple(getattr(avoid, "__iter__", None) and list(avoid) or list(getattr(avoid, "position", [])))
            if isinstance(pos, (list, tuple)) and pos:
                pos = tuple(int(x) for x in pos)
                size = tuple(int(s) for s in getattr(brain, "size", ()))
                deltas = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,-1)]
                for dx, dy in deltas:
                    cand = (pos[0]+dx, pos[1]+dy) if len(pos)>=2 else (pos[0]+dx,)
                    try:
                        if brain.is_inside(cand) and brain.get_neuron(cand) is None:
                            return cand
                    except Exception:
                        continue
        except Exception:
            pass
        try:
            for idx in brain.available_indices():
                return idx
        except Exception:
            pass
        return None

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        if not self.grow_on_step:
            return
        try:
            if getattr(current, "outgoing", None) and len(current.outgoing) > 0:
                return
            cur_new = int(getattr(wanderer, "_plugin_state", {}).get("growth_paradigm_added", 0))
            if cur_new >= self.max_new_per_walk:
                return
            brain = wanderer.brain
            idx = self._first_free(brain, getattr(current, "position", None))
            if idx is None:
                return
            brain.add_neuron(idx, tensor=0.0)
            brain.connect(getattr(current, "position"), idx, direction="uni")
            wanderer._plugin_state["growth_paradigm_added"] = cur_new + 1
            report("training", "paradigm_growth_step", {"from": getattr(current, "position", None), "to": idx}, "events")
        except Exception:
            pass

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        if not self.grow_on_end:
            return
        try:
            visited = getattr(wanderer, "_visited", [])
            if not visited:
                return
            last = visited[-1]
            if getattr(last, "outgoing", None) and len(last.outgoing) > 0:
                return
            cur_new = int(getattr(wanderer, "_plugin_state", {}).get("growth_paradigm_added", 0))
            if cur_new >= self.max_new_per_walk:
                return
            brain = wanderer.brain
            idx = self._first_free(brain, getattr(last, "position", None))
            if idx is None:
                return
            brain.add_neuron(idx, tensor=0.0)
            brain.connect(getattr(last, "position"), idx, direction="uni")
            wanderer._plugin_state["growth_paradigm_added"] = cur_new + 1
            report("training", "paradigm_growth_end", {"from": getattr(last, "position", None), "to": idx}, "events")
        except Exception:
            pass


try:
    register_learning_paradigm_type("growth", GrowthParadigm)
except Exception:
    pass


class SupervisedConvParadigm:
    """Supervised training helper paradigm that inserts conv neurons and enables learnables.

    - Attaches Conv1DRandomInsertionRoutine via SelfAttention to periodically insert conv1d neurons.
    - Ensures per-neuron learnables are created for any existing conv neurons (kernel and conv_bias).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        period = int(cfg.get("period", 10))
        eval_after = int(cfg.get("eval_after", 5))
        kernel = cfg.get("kernel", [1.0, 0.0, -1.0])
        max_data = int(cfg.get("max_data_sources", 1))
        self._sa = SelfAttention(routines=[Conv1DRandomInsertionRoutine(period=period, eval_after=eval_after, kernel=kernel, max_data_sources=max_data)])

    def on_wanderer(self, wanderer: "Wanderer") -> None:
        attach_selfattention(wanderer, self._sa)
        # Ensure existing conv neurons have learnable kernel/bias tensors
        try:
            for n in getattr(wanderer.brain, "neurons", {}).values():
                if getattr(n, "type_name", None) == "conv1d":
                    # Seed learnables from current PARAM neurons or defaults
                    lstore = getattr(n, "_plugin_state", {}).setdefault("learnable_params", {})
                    if "kernel" not in lstore:
                        self._sa.ensure_learnable_param(n, "kernel", [1.0, 0.0, -1.0], requires_grad=True)
                    if "conv_bias" not in lstore:
                        self._sa.ensure_learnable_param(n, "conv_bias", [0.0], requires_grad=True)
        except Exception:
            pass


class EpsilonGreedyParadigm:
    """Adds epsilon-greedy exploration to Wanderer by stacking the epsilongreedy and weights chooser plugins."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(config or {})

    def on_wanderer(self, wanderer: "Wanderer") -> None:
        # Attach epsilon-greedy and weight chooser plugins additively
        eg = _WANDERER_TYPES.get("epsilongreedy")
        wt = _WANDERER_TYPES.get("wanderalongsynapseweights")
        if eg is not None:
            getattr(wanderer, "_wplugins").append(eg)
        if wt is not None:
            getattr(wanderer, "_wplugins").append(wt)
        # Merge epsilon into neuro_config
        try:
            if hasattr(wanderer, "_neuro_cfg"):
                if "epsilongreedy_epsilon" in self.cfg:
                    wanderer._neuro_cfg["epsilongreedy_epsilon"] = float(self.cfg["epsilongreedy_epsilon"])  # type: ignore[attr-defined]
        except Exception:
            pass


class EvolutionaryPathsParadigm:
    """Simple evolutionary strategy: create alternate paths and mutate synapse weights on walk end.

    - Stacks alternatepathscreator plugin to ensure diversity.
    - Mutates synapse weights of visited edges by a small random factor.
    Config (neuro_config merged into Wanderer):
      * altpaths_min_len / altpaths_max_len / altpaths_max_paths_per_walk
      * mutate_prob (default 0.2), mutate_scale (default 0.1) => w *= (1 + U(-scale, +scale)) when applied
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(config or {})

    def on_wanderer(self, wanderer: "Wanderer") -> None:
        ap = _WANDERER_TYPES.get("alternatepathscreator")
        if ap is not None:
            getattr(wanderer, "_wplugins").append(ap)
        # Merge configs
        for k in ("altpaths_min_len", "altpaths_max_len", "altpaths_max_paths_per_walk"):
            if k in self.cfg:
                wanderer._neuro_cfg[k] = self.cfg[k]

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        import random as _r
        mutate_prob = float(self.cfg.get("mutate_prob", 0.2))
        scale = float(self.cfg.get("mutate_scale", 0.1))
        # Mutate weights on visited outgoing synapses
        try:
            for n in getattr(wanderer, "_visited", []) or []:
                for s in list(getattr(n, "outgoing", []) or []):
                    if _r.random() < mutate_prob:
                        try:
                            w = float(getattr(s, "weight", 1.0))
                            delta = (2.0 * _r.random() - 1.0) * scale
                            s.weight = w * (1.0 + delta)
                        except Exception:
                            pass
        except Exception:
            pass


try:
    register_learning_paradigm_type("supervised_conv", SupervisedConvParadigm)
    register_learning_paradigm_type("epsilon_greedy", EpsilonGreedyParadigm)
    register_learning_paradigm_type("evolutionary_paths", EvolutionaryPathsParadigm)
except Exception:
    pass


class SineWaveEncodingParadigm:
    """Random sine-wave encoding of start neuron input at the beginning of each walk.

    Config (passed to constructor):
    - sine_dim: output length (default 64)
    - num_waves: number of random sine components to sum (default 8)
    - freq_range: (fmin, fmax), default (0.1, 2.0)
    - amp_range: (amin, amax), default (0.5, 1.0)
    - phase_range: (0, 2π), default (0.0, 6.283185307179586)
    - seed_per_walk: if True, reseed parameters per walk; else reuse across walks

    The encoding replaces the start neuron's tensor with a length-`sine_dim` vector built from
    the sum of random sine waves; if torch is available, placed on the Wanderer's device.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.sine_dim = int(cfg.get("sine_dim", 64))
        self.num_waves = int(cfg.get("num_waves", 8))
        self.freq_range = tuple(cfg.get("freq_range", (0.1, 2.0)))
        self.amp_range = tuple(cfg.get("amp_range", (0.5, 1.0)))
        self.phase_range = tuple(cfg.get("phase_range", (0.0, 2.0 * 3.141592653589793)))
        self.seed_per_walk = bool(cfg.get("seed_per_walk", True))
        self._params = None  # cached list of (freq, amp, phase)

    def _ensure_params(self, rng) -> None:
        if self._params is not None and not self.seed_per_walk:
            return
        fmin, fmax = self.freq_range
        amin, amax = self.amp_range
        pmin, pmax = self.phase_range
        self._params = []
        for _ in range(max(1, self.num_waves)):
            f = rng.uniform(fmin, fmax)
            a = rng.uniform(amin, amax)
            p = rng.uniform(pmin, pmax)
            self._params.append((f, a, p))

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        import math as _m
        import random as _r
        self._ensure_params(_r)
        # Build encoding
        n = max(1, int(self.sine_dim))
        t = [i / float(n) for i in range(n)]
        vec = [0.0] * n
        for (f, a, p) in (self._params or []):
            for i, ti in enumerate(t):
                vec[i] += a * _m.sin(2.0 * _m.pi * f * ti + p)
        # Place on device
        try:
            torch = getattr(wanderer, "_torch", None)
            if torch is not None:
                dev = getattr(wanderer, "_device", "cpu")
                ten = torch.tensor(vec, dtype=torch.float32, device=dev)
                start.receive(ten)
            else:
                start.receive(vec)
        except Exception:
            try:
                start.receive(vec)
            except Exception:
                pass
        try:
            report("training", "sine_encoding", {"dim": n, "waves": len(self._params or [])}, "events")
        except Exception:
            pass


try:
    register_learning_paradigm_type("sine_encoding", SineWaveEncodingParadigm)
except Exception:
    pass

# Additional Learning Paradigms: Hebbian, Contrastive, Reinforcement, Student-Teacher

class HebbianParadigm:
    """Hebbian-like plasticity on synapse weights using pre/post activity correlation.

    Rule per transition (prev_syn -> current output):
        w_prev += eta * pre_mean * post_mean
        w_prev *= (1 - decay)
    Config: hebbian_eta (default 0.01), hebbian_decay (default 0.0)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.eta = float(cfg.get("hebbian_eta", 0.01))
        self.decay = float(cfg.get("hebbian_decay", 0.0))
        self._prev_syn: Optional["Synapse"] = None
        self._prev_mean: Optional[float] = None

    def _mean_val(self, x: Any) -> Optional[float]:
        try:
            if hasattr(x, "detach") and hasattr(x, "to"):
                v = x.detach().to("cpu").view(-1).float()
                return float(v.mean().item()) if v.numel() > 0 else None
            if isinstance(x, (list, tuple)) and x:
                return float(sum(float(v) for v in x) / len(x))
            return float(x)
        except Exception:
            return None

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: Optional["Synapse"], direction: str, step_index: int, out_value: Any) -> None:
        # Update previous synapse using correlation between previous out and current out
        post = self._mean_val(out_value)
        if self._prev_syn is not None and self._prev_mean is not None and post is not None:
            try:
                w = float(getattr(self._prev_syn, "weight", 1.0))
                w = w * (1.0 - max(0.0, self.decay)) + (self.eta * float(self._prev_mean) * float(post))
                self._prev_syn.weight = w
                report("training", "hebbian_update", {"w": w}, "paradigms")
            except Exception:
                pass
        # Prepare for next step: current synapse and pre (current out)
        self._prev_syn = syn
        self._prev_mean = self._mean_val(out_value)


class ContrastiveParadigm:
    """Attach a contrastive loss plugin (InfoNCE) to the Wanderer."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(config or {})

    def on_wanderer(self, wanderer: "Wanderer") -> None:
        plug = _WANDERER_TYPES.get("contrastive_infonce")
        if plug is not None:
            getattr(wanderer, "_wplugins").append(plug)
        # Merge tau/lambda into neuro_config
        for k in ("contrastive_tau", "contrastive_lambda"):
            if k in self.cfg:
                wanderer._neuro_cfg[k] = self.cfg[k]


class ReinforcementParadigm:
    """Attach TD(0) Q-learning choice optimization to the Wanderer."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(config or {})

    def on_wanderer(self, wanderer: "Wanderer") -> None:
        plug = _WANDERER_TYPES.get("td_qlearning")
        if plug is not None:
            getattr(wanderer, "_wplugins").append(plug)
        # Merge epsilon/alpha/gamma
        for k in ("rl_epsilon", "rl_alpha", "rl_gamma"):
            if k in self.cfg:
                wanderer._neuro_cfg[k] = self.cfg[k]


class StudentTeacherParadigm:
    """Adds a distillation loss against a moving-average teacher."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(config or {})

    def on_wanderer(self, wanderer: "Wanderer") -> None:
        plug = _WANDERER_TYPES.get("distillation")
        if plug is not None:
            getattr(wanderer, "_wplugins").append(plug)
        for k in ("distill_lambda", "teacher_momentum"):
            if k in self.cfg:
                wanderer._neuro_cfg[k] = self.cfg[k]


try:
    register_learning_paradigm_type("hebbian", HebbianParadigm)
    register_learning_paradigm_type("contrastive", ContrastiveParadigm)
    register_learning_paradigm_type("reinforcement", ReinforcementParadigm)
    register_learning_paradigm_type("student_teacher", StudentTeacherParadigm)
except Exception:
    pass
# -----------------------------
# Brain: n-Dimensional Structure
# -----------------------------

class Brain:
    """n-dimensional digital representation of space defined by a formula or Mandelbrot.

    - n: number of dimensions
    - size: grid resolution per dimension; int or tuple of ints
    - bounds: list/tuple of (min,max) for each dimension; defaults sensible ranges
    - formula: string expression referencing variables n1..nN, or 'mandelbrot'
    - max_iters, escape_radius: parameters for Mandelbrot computations

    The brain maintains a discrete occupancy grid; neurons/synapses must be placed
    at indices that are inside this shape.
    """

    def __init__(
        self,
        n: int,
        *,
        size: Union[int, Sequence[int]] = 32,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        formula: Optional[str] = None,
        max_iters: int = 50,
        escape_radius: float = 2.0,
        mode: str = "grid",
        sparse_bounds: Optional[Sequence[Union[Tuple[float, float], Tuple[float, None], Tuple[float]]]] = None,
        allow_dissimilar_datasets_in_wanderers: bool = False,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self.mode = str(mode)
        self._dataset_signature: Optional[str] = None
        self.allow_dissimilar_datasets_in_wanderers = bool(allow_dissimilar_datasets_in_wanderers)
        self._lock_dir = os.path.join(tempfile.gettempdir(), f"marble_brainlocks_{os.getpid()}_{id(self)}")
        try:
            os.makedirs(self._lock_dir, exist_ok=True)
        except Exception:
            pass

        if self.mode not in ("grid", "sparse"):
            raise ValueError("mode must be 'grid' or 'sparse'")

        # Shared storage for both modes
        self.synapses: List[Synapse] = []

        if self.mode == "grid":
            self.size: Tuple[int, ...] = self._normalize_size(size)
            if len(self.size) != self.n:
                raise ValueError("size must have length n")
            self.bounds: Tuple[Tuple[float, float], ...] = self._normalize_bounds(bounds)
            if len(self.bounds) != self.n:
                raise ValueError("bounds must provide (min,max) for each dimension")
            self.formula = formula
            self.max_iters = int(max_iters)
            self.escape_radius = float(escape_radius)

            # Prepare safe evaluation context
            self._eval_env = self._build_eval_env()

            # Build occupancy grid
            self.occupancy: Dict[Tuple[int, ...], bool] = {}
            self._populate_occupancy()
            try:
                report("brain", "occupancy", {"inside": sum(1 for v in self.occupancy.values() if v), "total": len(self.occupancy)}, "metrics")
            except Exception:
                pass

            # Storage of neurons by index
            self.neurons: Dict[Tuple[int, ...], Neuron] = {}
        else:
            # Sparse mode: only track occupied coordinates (floats)
            if sparse_bounds is None:
                raise ValueError("sparse_bounds must be provided in sparse mode")
            self.sparse_bounds: Tuple[Tuple[float, Optional[float]], ...] = self._normalize_sparse_bounds(sparse_bounds)
            if len(self.sparse_bounds) != self.n:
                raise ValueError("sparse_bounds must have length n")
            self.neurons: Dict[Tuple[float, ...], Neuron] = {}
        try:
            report("brain", "init", {"n": self.n, "mode": self.mode}, "events")
        except Exception:
            pass
        # Loaded learning paradigms (objects)
        self._paradigms: List[Any] = []
        # Disabled set of paradigm ids
        self._paradigm_disabled: set = set()

    # --- Public API ---
    def is_inside(self, index: Sequence[int]) -> bool:
        if self.mode == "grid":
            key = tuple(int(i) for i in index)
            return bool(self.occupancy.get(key, False))
        else:
            coords = tuple(float(v) for v in index)
            if len(coords) != self.n:
                return False
            for d in range(self.n):
                mn, mx = self.sparse_bounds[d]
                v = coords[d]
                if v < mn:
                    return False
                if mx is not None and v > mx:
                    return False
            return True

    def world_coords(self, index: Sequence[int]) -> Tuple[float, ...]:
        if self.mode == "grid":
            idx = tuple(int(i) for i in index)
            if len(idx) != self.n:
                raise ValueError("index length must equal n")
            return tuple(
                self._map_index_to_coord(i, dim)
                for dim, i in enumerate(idx)
            )
        else:
            # In sparse mode, indices are world coordinates already
            coords = tuple(float(v) for v in index)
            if len(coords) != self.n:
                raise ValueError("coordinate length must equal n")
            return coords

    def add_neuron(self, index: Sequence[int], *, tensor: Union[TensorLike, Sequence[float], float, int] = 0.0, **kwargs: Any) -> Neuron:
        if self.mode == "grid":
            idx = tuple(int(i) for i in index)
            if not self.is_inside(idx):
                raise ValueError("Neuron index is outside the brain shape")
            if idx in self.neurons:
                raise ValueError("Neuron already exists at this index")
            neuron = Neuron(tensor, **kwargs)
            setattr(neuron, "position", idx)
            self.neurons[idx] = neuron
            try:
                report("brain", "add_neuron", {"position": idx}, "events")
            except Exception:
                pass
            return neuron
        else:
            coords = tuple(float(v) for v in index)
            if not self.is_inside(coords):
                raise ValueError("Neuron coordinates are outside the brain bounds")
            if coords in self.neurons:
                raise ValueError("Neuron already exists at these coordinates")
            neuron = Neuron(tensor, **kwargs)
            setattr(neuron, "position", coords)
            self.neurons[coords] = neuron
            try:
                report("brain", "add_neuron", {"coords": coords}, "events")
            except Exception:
                pass
            return neuron

    def get_neuron(self, index: Sequence[int]) -> Optional[Neuron]:
        if self.mode == "grid":
            return self.neurons.get(tuple(int(i) for i in index))
        else:
            return self.neurons.get(tuple(float(v) for v in index))

    def connect(self, src_index: Sequence[int], dst_index: Sequence[int], *, direction: str = "uni", **kwargs: Any) -> Synapse:
        if self.mode == "grid":
            sidx = tuple(int(i) for i in src_index)
            didx = tuple(int(i) for i in dst_index)
            src = self.get_neuron(sidx)
            dst = self.get_neuron(didx)
        else:
            src = self.get_neuron(tuple(float(v) for v in src_index))
            dst = self.get_neuron(tuple(float(v) for v in dst_index))
        if src is None or dst is None:
            raise ValueError("Both source and target neurons must exist to connect")
        syn = Synapse(src, dst, direction=direction, **kwargs)
        self.synapses.append(syn)
        try:
            report("brain", "connect", {"direction": direction}, "events")
        except Exception:
            pass
        return syn

    # Learning paradigms API
    def load_paradigm(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        plug = _PARADIGM_TYPES.get(str(name))
        if plug is None:
            raise ValueError(f"Unknown learning paradigm: {name}")
        try:
            obj = plug(dict(config or {})) if callable(plug) else plug
        except Exception:
            # Try no-arg constructor
            obj = plug()
        self._paradigms.append(obj)
        try:
            report("training", "paradigm_load", {"name": str(name)}, "events")
        except Exception:
            pass
        return obj

    def enable_paradigm(self, name_or_obj: Any, *, enabled: bool = True) -> bool:
        lst = getattr(self, "_paradigms", []) or []
        def _id_of(o):
            return id(o)
        target_ids = []
        if isinstance(name_or_obj, str):
            plug = _PARADIGM_TYPES.get(name_or_obj)
            if plug is None:
                return False
            cls = plug if isinstance(plug, type) else plug.__class__
            for o in lst:
                try:
                    if isinstance(o, cls):
                        target_ids.append(_id_of(o))
                except Exception:
                    continue
        else:
            target_ids.append(_id_of(name_or_obj))
        if not target_ids:
            return False
        for tid in target_ids:
            if enabled:
                self._paradigm_disabled.discard(tid)
            else:
                self._paradigm_disabled.add(tid)
        try:
            report("training", "paradigm_toggle", {"enabled": enabled, "count": len(target_ids)}, "events")
        except Exception:
            pass
        return True

    def active_paradigms(self) -> List[Any]:
        return [p for p in (self._paradigms or []) if id(p) not in self._paradigm_disabled]

    # Remove a synapse and clean references
    def remove_synapse(self, synapse: "Synapse") -> None:
        if synapse in self.synapses:
            self.synapses.remove(synapse)
        try:
            if synapse in synapse.source.outgoing:
                synapse.source.outgoing.remove(synapse)
            if synapse in synapse.source.incoming:
                synapse.source.incoming.remove(synapse)
            if synapse in synapse.target.outgoing:
                synapse.target.outgoing.remove(synapse)
            if synapse in synapse.target.incoming:
                synapse.target.incoming.remove(synapse)
        except Exception:
            pass
        try:
            report("brain", "remove_synapse", {"direction": synapse.direction}, "events")
        except Exception:
            pass

    # Remove a neuron and all of its connected synapses
    def remove_neuron(self, neuron: "Neuron") -> None:
        try:
            # Copy lists to avoid modification during iteration
            for syn in list(getattr(neuron, "incoming", []) or []):
                self.remove_synapse(syn)
            for syn in list(getattr(neuron, "outgoing", []) or []):
                # Some synapses may already be removed from incoming loop; guard
                if syn in self.synapses:
                    self.remove_synapse(syn)
        except Exception:
            pass
        # Remove from neurons dictionary by its position key
        try:
            pos = getattr(neuron, "position", None)
            if pos is not None and pos in self.neurons:
                try:
                    del self.neurons[pos]
                except Exception:
                    # Fallback: rebuild neurons map without this neuron
                    self.neurons = {k: v for k, v in self.neurons.items() if v is not neuron}
            report("brain", "remove_neuron", {"position": pos}, "events")
        except Exception:
            pass

    # --- Cross-process locking helpers ---
    def _lockfile_path(self, key: str) -> str:
        return os.path.join(self._lock_dir, f"{key}.lck")

    class _FileLock:
        def __init__(self, path: str, timeout: Optional[float]) -> None:
            self.path = path
            self.timeout = timeout
            self._fh = None  # type: ignore
        def __enter__(self):
            start = time.time()
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._fh = open(self.path, "a+b")
            if msvcrt is None:
                return self
            while True:
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    return self
                except Exception:
                    if self.timeout is not None and (time.time() - start) > self.timeout:
                        raise TimeoutError(f"Timeout acquiring lock {self.path}")
                    time.sleep(0.01)
        def __exit__(self, exc_type, exc, tb):
            try:
                if self._fh is not None and msvcrt is not None:
                    try:
                        self._fh.seek(0)
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
            finally:
                try:
                    if self._fh is not None:
                        self._fh.close()
                except Exception:
                    pass

    def lock_neuron(self, neuron: "Neuron", timeout: Optional[float] = None):
        key = f"neuron_{str(getattr(neuron, 'position', id(neuron)))}"
        return self._FileLock(self._lockfile_path(key), timeout)

    def lock_synapse(self, synapse: "Synapse", timeout: Optional[float] = None):
        key = f"synapse_{id(synapse)}"
        return self._FileLock(self._lockfile_path(key), timeout)

    def available_indices(self) -> List[Tuple[int, ...]]:
        if self.mode == "grid":
            out = [idx for idx, inside in self.occupancy.items() if inside]
            try:
                report("brain", "available_indices", {"count": len(out)}, "metrics")
            except Exception:
                pass
            return out
        else:
            # In sparse mode, return occupied coordinates (floats)
            out = list(self.neurons.keys())  # type: ignore[return-value]
            try:
                report("brain", "available_indices", {"count": len(out)}, "metrics")
            except Exception:
                pass
            return out

    # --- Sparse mode utilities ---
    def bulk_add_neurons(
        self,
        positions: Sequence[Sequence[float]],
        *,
        tensor: Union[TensorLike, Sequence[float], float, int] = 0.0,
        **kwargs: Any,
    ) -> List[Neuron]:
        """Add multiple neurons in one call. In sparse mode, positions are world coords.
        In grid mode, positions are indices. Returns the created neurons in order.
        """
        created: List[Neuron] = []
        for pos in positions:
            created.append(self.add_neuron(pos, tensor=tensor, **kwargs))
        return created

    def export_sparse(self, path: str, include_synapses: bool = True) -> None:
        """Export sparse brain state to a JSON file. Only valid in sparse mode.

        Stores n, sparse_bounds, neurons (coords, weight, bias, age, type_name, tensor list),
        and synapses (source, target, direction, age, type_name) if requested.
        """
        if self.mode != "sparse":
            raise ValueError("export_sparse is only available in sparse mode")

        def tensor_to_list(t: Any) -> List[float]:
            try:
                # torch tensor path
                if hasattr(t, "detach") and hasattr(t, "tolist"):
                    return [float(x) for x in t.detach().to("cpu").view(-1).tolist()]
            except Exception:
                pass
            # assume list-like
            return [float(x) for x in (t if isinstance(t, (list, tuple)) else [t])]

        data: Dict[str, Any] = {
            "version": 1,
            "n": self.n,
            "mode": self.mode,
            "sparse_bounds": [
                [mn, mx if mx is not None else None] for (mn, mx) in self.sparse_bounds  # type: ignore[attr-defined]
            ],
            "neurons": [],
        }

        for coords, neuron in self.neurons.items():  # type: ignore[union-attr]
            item = {
                "coords": list(coords),
                "weight": neuron.weight,
                "bias": neuron.bias,
                "age": neuron.age,
                "type_name": neuron.type_name,
                "tensor": tensor_to_list(neuron.tensor),
            }
            data["neurons"].append(item)

        if include_synapses:
            syn_list = []
            for s in self.synapses:
                src = getattr(s.source, "position", None)
                dst = getattr(s.target, "position", None)
                syn_list.append(
                    {
                        "source": list(src),
                        "target": list(dst),
                        "direction": s.direction,
                        "age": s.age,
                        "type_name": s.type_name,
                    }
                )
            data["synapses"] = syn_list

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        try:
            report("brain", "export_sparse", {"path": path, "neurons": len(data["neurons"]), "synapses": len(data.get("synapses", []))}, "io")
        except Exception:
            pass

    @classmethod
    def import_sparse(cls, path: str) -> "Brain":
        """Load a sparse brain from a JSON file created by export_sparse."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("mode") != "sparse":
            raise ValueError("Invalid sparse brain file")
        n = int(data["n"])
        sb_raw = data["sparse_bounds"]
        sparse_bounds: List[Tuple[float, Optional[float]]] = []
        for b in sb_raw:
            mn = float(b[0])
            mx = None if b[1] is None else float(b[1])
            sparse_bounds.append((mn, mx))
        brain = cls(n, mode="sparse", sparse_bounds=tuple(sparse_bounds))

        for item in data.get("neurons", []):
            coords = item["coords"]
            tensor = item.get("tensor", 0.0)
            weight = item.get("weight", 1.0)
            bias = item.get("bias", 0.0)
            age = item.get("age", 0)
            type_name = item.get("type_name", None)
            brain.add_neuron(coords, tensor=tensor, weight=weight, bias=bias, age=age, type_name=type_name)

        for s in data.get("synapses", []):
            src = s["source"]
            dst = s["target"]
            direction = s.get("direction", "uni")
            age = s.get("age", 0)
            type_name = s.get("type_name", None)
            brain.connect(src, dst, direction=direction, age=age, type_name=type_name)

        try:
            report("brain", "import_sparse", {"path": path, "neurons": len(brain.neurons), "synapses": len(brain.synapses)}, "io")
        except Exception:
            pass
        return brain

    # --- Occupancy/grid helpers ---
    def _normalize_size(self, size: Union[int, Sequence[int]]) -> Tuple[int, ...]:
        if isinstance(size, int):
            return tuple([size] * self.n)
        return tuple(int(s) for s in size)

    def _normalize_bounds(self, bounds: Optional[Sequence[Tuple[float, float]]]) -> Tuple[Tuple[float, float], ...]:
        if bounds is None:
            # Defaults: Mandelbrot-ish plane for first two dims, [-1,1] for others
            default = [(-2.0, 1.0), (-1.5, 1.5)] + [(-1.0, 1.0)] * max(0, self.n - 2)
            return tuple(default[: self.n])
        return tuple((float(a), float(b)) for (a, b) in bounds)

    def _normalize_sparse_bounds(self, bounds: Sequence[Union[Tuple[float, float], Tuple[float, None], Tuple[float]]]) -> Tuple[Tuple[float, Optional[float]], ...]:
        norm: List[Tuple[float, Optional[float]]] = []
        for b in bounds:
            if len(b) == 1:
                mn = float(b[0])
                norm.append((mn, None))
            elif len(b) == 2:
                mn = float(b[0])
                mx = b[1]
                if mx is None:
                    norm.append((mn, None))
                else:
                    mx_f = float(mx)
                    if mx_f < mn:
                        raise ValueError("max must be >= min for each dimension")
                    norm.append((mn, mx_f))
            else:
                raise ValueError("Each sparse bound must be (min,) or (min,max)")
        return tuple(norm)

    def _map_index_to_coord(self, i: int, dim: int) -> float:
        npoints = max(1, self.size[dim])
        a, b = self.bounds[dim]
        if npoints == 1:
            return (a + b) / 2.0
        # map i in [0, npoints-1] to [a,b]
        t = float(i) / float(npoints - 1)
        return a * (1.0 - t) + b * t

    def _build_eval_env(self) -> Dict[str, Any]:
        env: Dict[str, Any] = {}
        # Math functions/constants
        safe_math = {
            k: getattr(math, k)
            for k in (
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "sqrt",
                "exp",
                "log",
                "log10",
                "fabs",
                "floor",
                "ceil",
                "pow",
                "pi",
                "e",
            )
            if hasattr(math, k)
        }
        env.update(safe_math)
        # Builtins allowed
        env.update({"abs": abs, "min": min, "max": max})
        # Special functions
        env["mandelbrot"] = self._mandelbrot
        env["mandelbrot_nd"] = self._mandelbrot_nd
        return env

    def _mandelbrot(self, n1: float, n2: float, *, max_iters: Optional[int] = None, escape_radius: Optional[float] = None) -> int:
        # Classic 2D Mandelbrot on complex plane
        iters = self.max_iters if max_iters is None else int(max_iters)
        R = self.escape_radius if escape_radius is None else float(escape_radius)
        c = complex(n1, n2)
        z = 0j
        for i in range(iters):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) > R * R:
                return i
        return iters

    def _mandelbrot_nd(self, *coords: float, max_iters: Optional[int] = None, escape_radius: Optional[float] = None) -> int:
        # Simple n-D generalization using element-wise square and L2 norm for escape
        iters = self.max_iters if max_iters is None else int(max_iters)
        R = self.escape_radius if escape_radius is None else float(escape_radius)
        c = list(float(x) for x in coords)
        z = [0.0 for _ in c]
        for i in range(iters):
            z = [v * v + c[j] for j, v in enumerate(z)]
            norm2 = sum(v * v for v in z)
            if norm2 > R * R:
                return i
        return iters

    def _eval_formula(self, coords: Tuple[float, ...]) -> float:
        # If formula is None, default behavior: use mandelbrot (2D) or inside hypercube elsewhere
        if self.formula is None:
            if self.n == 2:
                return float(self._mandelbrot(coords[0], coords[1]))
            else:
                # Inside everywhere by default
                return 1.0
        expr = self.formula.strip()
        # Provide variables n1..nN
        local_vars = {f"n{i+1}": coords[i] for i in range(self.n)}
        try:
            val = eval(expr, {"__builtins__": {}}, {**self._eval_env, **local_vars})
        except Exception as e:
            raise ValueError(f"Error evaluating formula: {e}")
        # Interpret numeric/boolean
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        try:
            return float(val)
        except Exception:
            raise ValueError("Formula must evaluate to a boolean or numeric value")

    def _populate_occupancy(self) -> None:
        # If a boolean-like formula (<=,>=,<,>), inside where True
        # If numeric (e.g., mandelbrot iteration count), inside if val>0
        # Iterate over all index tuples
        def rec_build(dim: int, prefix: List[int]) -> None:
            if dim == self.n:
                idx = tuple(prefix)
                coords = tuple(self._map_index_to_coord(prefix[d], d) for d in range(self.n))
                val = self._eval_formula(coords)
                inside = bool(val != 0.0)
                self.occupancy[idx] = inside
                return
            for i in range(self.size[dim]):
                prefix.append(i)
                rec_build(dim + 1, prefix)
                prefix.pop()

        rec_build(0, [])


__all__ += ["Brain"]


# -----------------------------
# Brain Training Plugins + Method
# -----------------------------

_BRAIN_TRAIN_TYPES: Dict[str, Any] = {}


def register_brain_train_type(name: str, plugin: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Brain train type name must be a non-empty string")
    _BRAIN_TRAIN_TYPES[name] = plugin


class CurriculumTrainPlugin:
    """Brain-train plugin that increases walk max_steps across walks.

    Config keys (read from the `config` dict passed to on_init):
    - start_steps (default 1)
    - step_increment (default 1)
    - max_cap (optional int): if provided, cap max_steps to this value.
    """
    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.start_steps = int(cfg.get("start_steps", 1))
        self.inc = int(cfg.get("step_increment", 1))
        self.cap = cfg.get("max_cap")

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        ms = self.start_steps + i * self.inc
        if self.cap is not None:
            try:
                ms = min(ms, int(self.cap))
            except Exception:
                pass
        return {"max_steps": int(ms)}


try:
    register_brain_train_type("curriculum", CurriculumTrainPlugin())
except Exception:
    pass


def _merge_dict_safe(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _normalize_walk_overrides(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in ("max_steps", "lr"):
        if k in d:
            out[k] = d[k]
    return out


def _call_safely(fn: Optional[Callable], *args, **kwargs):
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _maybe_report(group: str, item: str, data: Any, *subs: str) -> None:
    try:
        report(group, item, data, *subs)
    except Exception:
        pass


def _select_start(brain: "Brain", wanderer: "Wanderer", i: int, plugin: Optional[Any], start_selector: Optional[Callable[["Brain"], Optional["Neuron"]]]):
    # Plugin choice first
    if plugin is not None and hasattr(plugin, "choose_start"):
        res = _call_safely(getattr(plugin, "choose_start"), brain, wanderer, i)
        if res is not None:
            return res
    # Fallback to start_selector
    if start_selector is not None:
        return start_selector(brain)
    return None


def _before_walk_overrides(plugin: Optional[Any], brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
    if plugin is not None and hasattr(plugin, "before_walk"):
        res = _call_safely(getattr(plugin, "before_walk"), brain, wanderer, i)
        return _normalize_walk_overrides(res)
    return {}


def _after_walk(plugin: Optional[Any], brain: "Brain", wanderer: "Wanderer", i: int, stats: Dict[str, Any]) -> None:
    if plugin is not None and hasattr(plugin, "after_walk"):
        _call_safely(getattr(plugin, "after_walk"), brain, wanderer, i, stats)


def _on_init_train(plugin: Optional[Any], brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:
    if plugin is not None and hasattr(plugin, "on_init"):
        _call_safely(getattr(plugin, "on_init"), brain, wanderer, config)


def _on_end_train(plugin: Optional[Any], brain: "Brain", wanderer: "Wanderer", history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if plugin is not None and hasattr(plugin, "on_end"):
        res = _call_safely(getattr(plugin, "on_end"), brain, wanderer, history)
        return res if isinstance(res, dict) else None
    return None


def _brain_train(
    brain: "Brain",
    wanderer: "Wanderer",
    *,
    num_walks: int,
    max_steps: int,
    lr: float,
    start_selector: Optional[Callable[["Brain"], Optional["Neuron"]]],
    callback: Optional[Callable[[int, Dict[str, Any]], None]],
    type_name: Optional[str],
) -> Dict[str, Any]:
    # Resolve stacked brain-train plugins (comma-separated str or list)
    plugins: List[Any] = []
    if isinstance(type_name, str):
        names = [s.strip() for s in type_name.split(",") if s.strip()]
        for nm in names:
            p = _BRAIN_TRAIN_TYPES.get(nm)
            if p is not None:
                plugins.append(p)
    elif isinstance(type_name, (list, tuple)):
        for nm in type_name:
            p = _BRAIN_TRAIN_TYPES.get(str(nm))
            if p is not None:
                plugins.append(p)
    config = {
        "num_walks": num_walks,
        "max_steps": max_steps,
        "lr": lr,
        "type_name": type_name,
    }
    for p in plugins:
        _on_init_train(p, brain, wanderer, config)
    history: List[Dict[str, Any]] = []
    for i in range(num_walks):
        # Merge overrides from all plugins; later plugins win on conflicts
        overrides: Dict[str, Any] = {}
        for p in plugins:
            ovr = _before_walk_overrides(p, brain, wanderer, i)
            overrides.update(ovr)
        ms = int(overrides.get("max_steps", max_steps))
        lr_i = float(overrides.get("lr", lr))
        # Apply additive choose_start: last non-None from plugins wins; else external selector
        sel = None
        for p in plugins:
            s = _select_start(brain, wanderer, i, p, None)
            if s is not None:
                sel = s
        start = sel if sel is not None else _select_start(brain, wanderer, i, None, start_selector)
        stats = wanderer.walk(max_steps=ms, start=start, lr=lr_i)
        history.append(stats)
        for p in plugins:
            _after_walk(p, brain, wanderer, i, stats)
        _maybe_report("training", f"brain_walk_{i}", {"loss": stats.get("loss", 0.0), "steps": stats.get("steps", 0)}, "brain")
        if callback is not None:
            _call_safely(callback, i, stats)
    final_loss = history[-1]["loss"] if history else 0.0
    # Merge extras; later plugins win on conflicts
    result = {"history": history, "final_loss": final_loss}
    for p in plugins:
        extra = _on_end_train(p, brain, wanderer, history)
        result = _merge_dict_safe(result, extra)
    _maybe_report("training", "brain_summary", {"num_walks": num_walks, "final_loss": final_loss}, "brain")
    return result


# Method attached to Brain via class definition
def _brain_train_method(
    self: "Brain",
    wanderer: "Wanderer",
    *,
    num_walks: int = 10,
    max_steps: int = 10,
    lr: float = 1e-2,
    start_selector: Optional[Callable[["Brain"], Optional["Neuron"]]] = None,
    callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    type_name: Optional[str] = None,
) -> Dict[str, Any]:
    return _brain_train(
        self,
        wanderer,
        num_walks=num_walks,
        max_steps=max_steps,
        lr=lr,
        start_selector=start_selector,
        callback=callback,
        type_name=type_name,
    )


# Bind train to Brain
setattr(Brain, "train", _brain_train_method)

__all__ += ["register_brain_train_type"]


# -----------------------------
# Wanderer with Autograd + Plugins
# -----------------------------

# Plugin registry for Wanderer (moved). Import registry and registrar.
from .wanderer import register_wanderer_type  # re-exported below
from .wanderer import WANDERER_TYPES_REGISTRY as _WANDERER_TYPES
try:
    # Ensure built-in Wanderer plugins are loaded (self-register on import)
    from .plugins.wanderer_weights import WanderAlongSynapseWeightsPlugin as _WPL  # noqa: F401
    from .plugins.wanderer_bestpath import BestLossPathPlugin as _BPL  # noqa: F401
    from .plugins.wanderer_altpaths import AlternatePathsCreatorPlugin as _APL  # noqa: F401
    from .plugins.wanderer_epsgreedy import EpsilonGreedyChooserPlugin as _EPL  # noqa: F401
except Exception:
    pass


class BestLossPathPlugin:
    """On first choose_next in a walk, search paths up to max_steps and upweight best path synapses.

    The synapse weights are increased along the best path so that a weights-driven
    plugin (e.g., WanderAlongSynapseWeightsPlugin) will prefer it.
    """
    def _simulate(self, wanderer: "Wanderer", start: "Neuron", max_steps: int):
        # DFS over choices; returns (best_loss, best_path_edges)
        best = (float("inf"), [])
        seen = set()

        def rec(node: "Neuron", carried, depth: int, outputs: List[Any], edges: List[Tuple["Synapse", str]]):
            nonlocal best
            if depth >= max_steps:
                # Evaluate
                loss_t = wanderer._compute_loss(outputs)
                try:
                    loss_v = float(loss_t.detach().to("cpu").item())
                except Exception:
                    loss_v = float("inf")
                if loss_v < best[0]:
                    best = (loss_v, list(edges))
                return
            # Compute current output
            out = node.forward(carried)
            outputs2 = outputs + [out]
            # Choices from node
            choices = wanderer._gather_choices(node)
            if not choices:
                loss_t = wanderer._compute_loss(outputs2)
                try:
                    loss_v = float(loss_t.detach().to("cpu").item())
                except Exception:
                    loss_v = float("inf")
                if loss_v < best[0]:
                    best = (loss_v, list(edges))
                return
            for syn, dir_str in choices:
                # Determine next node
                nxt = syn.target if dir_str == "forward" else syn.source
                rec(nxt, out, depth + 1, outputs2, edges + [(syn, dir_str)])

        rec(start, None, 0, [], [])
        return best

    def _bump_weights(self, path_edges: List[Tuple["Synapse", str]], factor: float = 1.0, add: float = 1.0):
        for syn, _ in path_edges:
            try:
                w = float(getattr(syn, "weight", 1.0))
                syn.weight = w * factor + add
            except Exception:
                pass

    def choose_next(self, wanderer: "Wanderer", current: "Neuron", choices: List[Tuple["Synapse", str]]):
        # Run once per walk
        if not getattr(wanderer, "_plugin_state", None):
            wanderer._plugin_state = {}
        if not wanderer._plugin_state.get("bestlosspath_applied"):
            # Read config from wanderer._neuro_cfg
            cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
            max_steps_cfg = int(cfg.get("bestlosspath_search_steps", 3))
            bump_factor = float(cfg.get("bestlosspath_bump_factor", 1.0))
            bump_add = float(cfg.get("bestlosspath_bump_add", 1.0))
            best_loss, best_path = self._simulate(wanderer, current, max_steps_cfg)
            self._bump_weights(best_path, factor=bump_factor, add=bump_add)
            wanderer._plugin_state["bestlosspath_applied"] = True
            try:
                # Log node positions along path for audit
                nodes = []
                for syn, d in best_path:
                    try:
                        nodes.append({
                            "src": getattr(getattr(syn, "source", None), "position", None),
                            "dst": getattr(getattr(syn, "target", None), "position", None),
                            "dir": d,
                        })
                    except Exception:
                        pass
                report("wanderer", "bestlosspath", {"best_loss": best_loss, "edges": len(best_path), "nodes": nodes}, "events")
            except Exception:
                pass
        # Return current best by weight; a weights plugin stacked later will override anyway
        if choices:
            best = max(choices, key=lambda cd: float(getattr(cd[0], "weight", 1.0)))
            return best
        return (None, "forward")


try:
    register_wanderer_type("bestlosspath", BestLossPathPlugin())
except Exception:
    pass


class AlternatePathsCreatorPlugin:
    """At the end of each walk, create a new alternate path of random length and connect it to a random visited neuron.

    Config keys read from `wanderer._neuro_cfg`:
    - `altpaths_min_len` (default 2)
    - `altpaths_max_len` (default 4)
    """

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        import random as _rand
        brain = wanderer.brain
        visited = getattr(wanderer, "_visited", []) or []
        if not visited:
            return
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        min_len = int(cfg.get("altpaths_min_len", 2))
        max_len = int(cfg.get("altpaths_max_len", 4))
        if max_len < min_len:
            max_len = min_len
        # Per-walk limit on number of new paths
        max_paths = int(cfg.get("altpaths_max_paths_per_walk", 1))
        created = int(getattr(wanderer, "_plugin_state", {}).get("altpaths_created", 0))
        if created >= max_paths:
            return
        length = _rand.randint(min_len, max_len)
        # Pick an anchor neuron from the visited path
        anchor = _rand.choice(visited)
        # Build a chain of new neurons
        new_nodes = []
        try:
            for _ in range(length):
                # pick first free index
                idx = None
                for cand in brain.available_indices():
                    try:
                        if brain.get_neuron(cand) is None:
                            idx = cand
                            break
                    except Exception:
                        continue
                if idx is None:
                    # try neighbor of last node (or anchor)
                    base = new_nodes[-1] if new_nodes else anchor
                    bpos = getattr(base, "position", None)
                    if isinstance(bpos, tuple):
                        for dx, dy in [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,-1)]:
                            cand = (bpos[0]+dx, bpos[1]+dy) if len(bpos)>=2 else (bpos[0]+dx,)
                            try:
                                if brain.is_inside(cand) and brain.get_neuron(cand) is None:
                                    idx = cand
                                    break
                            except Exception:
                                continue
                if idx is None:
                    break
                n = brain.add_neuron(idx, tensor=0.0)
                new_nodes.append(n)
        except Exception:
            new_nodes = new_nodes
        if not new_nodes:
            return
        # Connect chain
        try:
            prev = anchor
            for node in new_nodes:
                brain.connect(getattr(prev, "position"), getattr(node, "position"), direction="uni")
                prev = node
            report("training", "altpaths_create", {"anchor": getattr(anchor, "position", None), "len": len(new_nodes)}, "events")
            # Track created count this walk
            if not getattr(wanderer, "_plugin_state", None):
                wanderer._plugin_state = {}
            wanderer._plugin_state["altpaths_created"] = created + 1
        except Exception:
            pass


try:
    register_wanderer_type("alternatepathscreator", AlternatePathsCreatorPlugin())
except Exception:
    pass


 


class HyperEvolutionPlugin:
    """Evolutionary architecture search over plugin stacks and parameters without cloning the brain.

    Modes (via `wanderer._neuro_cfg['hyper_evo_mode']`):
    - "per_walk" (default): before the first training walk, run `hyper_evo_steps` evolution steps
      using temporary changes and full rollback per step. After finishing, configure the best
      architecture, deactivate this plugin, and proceed with normal training.
    - "per_step": at the beginning of each training step, run `hyper_evo_steps` evolution steps,
      configure the best architecture so far, then continue the step. This lets the search continue
      during training.

    Config (via `wanderer._neuro_cfg`):
      - hyper_evo_steps (int, default 50): evolution steps per run
      - hyper_eval_steps (int, default 3): bounded evaluation walk steps (with lr=0)
      - hyper_evo_mode (str, default "per_walk"): "per_walk" | "per_step"

    This search mutates across all registered plugin families (wanderer, neuroplasticity, paradigms)
    and any numeric parameters present in `wanderer._neuro_cfg`. No plugins or parameters are excluded.
    Each evolution step applies a temporary mutation, performs a short evaluation walk (lr=0) to get
    loss and speed metrics, and then rolls back every change unless both metrics improved; accepted
    mutations are accumulated into the current best genome. At the end of the run, the best genome is
    applied persistently (no temporary stacks), and this plugin deactivates itself in per_walk mode.
    """

    def on_init(self, wanderer: "Wanderer") -> None:
        state = getattr(wanderer, "_plugin_state", None)
        if state is None:
            wanderer._plugin_state = {}
            state = wanderer._plugin_state
        state.setdefault("hyper", {
            "gen": 0,
            "pop": [],  # legacy fields retained (not used by new evo loop)
            "idx": 0,
            "last_handle": None,
            "last_paradigms": None,
            "best": None,
            "best_score": None,
            "best_loss": None,
            "best_speed": None,
            "done": False,
            "disabled": False,
        })
        if not state["hyper"]["pop"]:
            self._init_population(wanderer)

    def _init_population(self, wanderer: "Wanderer") -> None:
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        pop_size = int(cfg.get("hyper_pop", 8))
        import random as _r
        pop = []
        wnames = list(_WANDERER_TYPES.keys())
        nnames = list(_NEURO_TYPES.keys())
        pnames = list(_PARADIGM_TYPES.keys()) if '_PARADIGM_TYPES' in globals() else []
        for _ in range(pop_size):
            genome = {
                "w": _r.sample(wnames, k=min(len(wnames), _r.randint(0, min(3, len(wnames))))) if wnames else [],
                "n": _r.sample(nnames, k=min(len(nnames), _r.randint(0, min(2, len(nnames))))) if nnames else [],
                "p": _r.sample(pnames, k=min(len(pnames), _r.randint(0, min(2, len(pnames))))) if pnames else [],
                "cfg": {},
                "score": None,
            }
            # Ensure at least one wanderer plugin is present to make evolution impactful.
            if not genome["w"] and wnames:
                try:
                    genome["w"] = [_r.choice(wnames)]
                except Exception:
                    pass
            # Encourage at least one paradigm to be active for diversity
            if not genome["p"] and pnames:
                try:
                    genome["p"] = [_r.choice(pnames)]
                except Exception:
                    pass
            pop.append(genome)
        # Seed population with a strong, commonly useful combination if available
        try:
            if wnames and ("bestlosspath" in wnames) and ("wanderalongsynapseweights" in wnames):
                pop[0] = {
                    "w": ["bestlosspath", "wanderalongsynapseweights"],
                    "n": [],
                    "p": [],
                    "cfg": {},
                    "score": None,
                }
        except Exception:
            pass
        wanderer._plugin_state["hyper"]["pop"] = pop
        wanderer._plugin_state["hyper"]["idx"] = 0

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        # Run evolution per mode and apply the best architecture if needed
        st = wanderer._plugin_state["hyper"]
        if st.get("disabled"):
            return
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        mode = str(cfg.get("hyper_evo_mode", "per_walk")).lower()
        steps = int(cfg.get("hyper_evo_steps", 50))
        if mode == "per_walk" and not st.get("done"):
            self._run_evolution(wanderer, start, steps)
            self._apply_best_persistently(wanderer)
            st["done"] = True
            st["disabled"] = True
            return

        # Legacy selection/apply path retained for compatibility
        pop = st["pop"]
        if not pop:
            self._init_population(wanderer)
            pop = st["pop"]
        # Selection strategy: exploit best-so-far or explore via tournament
        import random as _r
        idx = 0
        try:
            # Identify scored genomes
            scored = [(i, float(g.get("score_avg", g.get("score", float("inf"))))) for i, g in enumerate(pop)]
            scored = [(i, s) for (i, s) in scored if s != float("inf")]
            best_idx = st.get("best_idx")
            if scored:
                # Track current best
                scored.sort(key=lambda t: t[1])
                current_best_idx = scored[0][0]
                st["best_idx"] = current_best_idx
                # Epsilon-greedy: mostly exploit best, sometimes explore
                eps = 0.3
                if best_idx is not None and _r.random() > eps:
                    idx = int(best_idx)
                else:
                    # Tournament among a random subset
                    k = max(2, min(5, len(pop)))
                    cand = _r.sample(range(len(pop)), k=k)
                    # Rank by available scores, unknown treated as mid-rank
                    def score_of(i):
                        g = pop[i]
                        s = g.get("score_avg", g.get("score"))
                        return float(s) if s is not None else float("inf")
                    idx = min(cand, key=score_of)
            else:
                # No scores yet: random selection
                idx = _r.randrange(0, len(pop))
        except Exception:
            idx = 0
        st["idx"] = idx
        genome = pop[idx]
        # Apply stacks
        handle = push_temporary_plugins(wanderer, wanderer_types=genome.get("w"), neuro_types=genome.get("n"))
        st["last_handle"] = handle
        # Toggle paradigms: enable selected, disable others previously managed
        prev = st.get("last_paradigms")
        act = []
        try:
            # disable previous
            if prev:
                for nm in prev:
                    try:
                        wanderer.brain.enable_paradigm(nm, enabled=False)
                    except Exception:
                        pass
            for nm in genome.get("p", []) or []:
                try:
                    wanderer.brain.enable_paradigm(nm, enabled=True)
                    act.append(nm)
                except Exception:
                    pass
            st["last_paradigms"] = act
        except Exception:
            pass
        # Snapshot current neuro config to isolate genome effects
        try:
            st["cfg_prev"] = dict(getattr(wanderer, "_neuro_cfg", {}) or {})
        except Exception:
            st["cfg_prev"] = None
        # Snapshot current LR override if any
        try:
            st["lr_prev"] = getattr(wanderer, "lr_override", None)
        except Exception:
            st["lr_prev"] = None
        # Merge any genome cfg params into wanderer._neuro_cfg (restored on walk end)
        try:
            for k, v in (genome.get("cfg") or {}).items():
                wanderer._neuro_cfg[k] = v
        except Exception:
            pass
        # If genome proposes an LR override, apply it (treated like any other evolvable param)
        try:
            lr_prop = (genome.get("cfg") or {}).get("lr_override", None)
            if lr_prop is not None:
                try:
                    wanderer.lr_override = float(lr_prop)
                except Exception:
                    pass
        except Exception:
            pass

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        # Per-step mode: run micro evolution before continuing the step
        st = wanderer._plugin_state.get("hyper", {})
        if st.get("disabled"):
            return
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        mode = str(cfg.get("hyper_evo_mode", "per_walk")).lower()
        if mode != "per_step":
            return
        steps = int(cfg.get("hyper_evo_steps", 50))
        self._run_evolution(wanderer, current, steps)
        self._apply_best_persistently(wanderer)
        return

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        # Evaluate fitness
        st = wanderer._plugin_state.get("hyper", {})
        pop = st.get("pop", [])
        idx = st.get("idx", 0) % (len(pop) if pop else 1)
        if not pop:
            return
        genome = pop[idx]
        # Compute objectives (used only for legacy population mode)
        loss = float(stats.get("loss", 0.0))
        steps = int(stats.get("steps", 0))
        sm = stats.get("step_metrics", []) or []
        # per-step time
        dts = [m.get("dt") for m in sm if m.get("dt") is not None]
        mean_dt = (sum(dts) / len(dts)) if dts else 0.0
        # loss decrease speed (positive good): use -avg(delta) truncated at 0
        deltas = [float(m.get("delta", 0.0)) for m in sm]
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            speed = max(0.0, -avg_delta)
        else:
            speed = 0.0
        # brain size
        try:
            brain_size = len(getattr(wanderer.brain, "neurons", {}))
        except Exception:
            brain_size = 0
        # Score: minimize loss, mean_dt, brain_size; maximize speed => use reciprocal
        score = (loss) + (mean_dt) + (brain_size * 1e-3) + (1.0 / (1e-6 + speed) if speed > 0 else 1.0)
        # Update instantaneous and running-average scores for the genome
        genome["score"] = float(score)
        try:
            prev_avg = float(genome.get("score_avg", score))
            trials = int(genome.get("trials", 0)) + 1
            new_avg = (prev_avg * (trials - 1) + float(score)) / max(1, trials)
            genome["score_avg"] = float(new_avg)
            genome["trials"] = trials
        except Exception:
            genome["score_avg"] = float(score)
            genome["trials"] = 1
        # Restore stacks
        try:
            if st.get("last_handle") is not None:
                pop_handle = st["last_handle"]
                pop_temporary_plugins(wanderer, pop_handle)
                st["last_handle"] = None
        except Exception:
            pass
        # Restore neuro config snapshot
        try:
            prev = st.get("cfg_prev", None)
            if prev is not None:
                wanderer._neuro_cfg = dict(prev)
                st["cfg_prev"] = None
        except Exception:
            pass
        # Restore LR override
        try:
            if "lr_prev" in st:
                setattr(wanderer, "lr_override", st.get("lr_prev", None))
                st["lr_prev"] = None
        except Exception:
            pass
        # Harvest observed neuro_cfg keys to expand future mutation search space
        try:
            kb = st.setdefault("key_bank", set())
            for k in (getattr(wanderer, "_neuro_cfg", {}) or {}).keys():
                try:
                    kb.add(str(k))
                except Exception:
                    pass
        except Exception:
            pass

        # Advance and evolve at end of a population cycle (count-based, independent of chosen indices)
        st["eval_count"] = int(st.get("eval_count", 0)) + 1
        if st["eval_count"] >= max(1, len(pop)):
            st["eval_count"] = 0
            self._evolve(wanderer)

    def _evolve(self, wanderer: "Wanderer") -> None:
        import random as _r
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        # Adaptive mutation rate around configured baseline
        base_mut = float(cfg.get("hyper_mut", 0.3))
        st = wanderer._plugin_state["hyper"]
        cur_mut = float(st.get("mut_rate", base_mut))
        mut_rate = max(0.05, min(0.8, cur_mut))
        keep = int(cfg.get("hyper_keep", 2))
        pop = st["pop"]
        # Sort by running-average score ascending when available (else instantaneous)
        pop.sort(key=lambda g: float(g.get("score_avg", g.get("score", float("inf")))))
        keepers = pop[:max(1, min(keep, len(pop)))]
        # Track best score to adapt mutation rate
        try:
            best_now = float(keepers[0].get("score_avg", keepers[0].get("score", float("inf")))) if keepers else float("inf")
            last_best = float(st.get("last_best", float("inf")))
            if best_now < last_best:
                st["mut_rate"] = max(0.05, mut_rate * 0.9)
            else:
                st["mut_rate"] = min(0.8, mut_rate * 1.1)
            st["last_best"] = best_now
        except Exception:
            st["mut_rate"] = mut_rate
        # Refill
        wnames = list(_WANDERER_TYPES.keys())
        nnames = list(_NEURO_TYPES.keys())
        pnames = list(_PARADIGM_TYPES.keys()) if '_PARADIGM_TYPES' in globals() else []
        new_pop = []
        for g in keepers:
            new_pop.append({"w": list(g["w"]), "n": list(g["n"]), "p": list(g["p"]), "cfg": dict(g.get("cfg") or {}), "score": None, "score_avg": g.get("score_avg"), "trials": g.get("trials")})
        while len(new_pop) < len(pop):
            # Crossover (pick two parents)
            a, b = _r.sample(keepers, k=2) if len(keepers) >= 2 else (keepers[0], keepers[0])
            child = {
                "w": list(set(_r.sample(a["w"] + b["w"], k=min(len(a["w"] + b["w"]), _r.randint(0, min(3, len(wnames))))))) if (a["w"] or b["w"]) else [],
                "n": list(set(_r.sample(a["n"] + b["n"], k=min(len(a["n"] + b["n"]), _r.randint(0, min(2, len(nnames))))))) if (a["n"] or b["n"]) else [],
                "p": list(set(_r.sample(a["p"] + b["p"], k=min(len(a["p"] + b["p"]), _r.randint(0, min(2, len(pnames))))))) if (a["p"] or b["p"]) else [],
                "cfg": dict(a.get("cfg") or {}),
                "score": None,
            }
            # Mutate types
            if _r.random() < mut_rate and wnames:
                # flip one wanderer plugin
                choice = _r.choice(wnames)
                if choice in child["w"]:
                    child["w"].remove(choice)
                else:
                    child["w"].append(choice)
            if _r.random() < mut_rate and nnames:
                choice = _r.choice(nnames)
                if choice in child["n"]:
                    child["n"].remove(choice)
                else:
                    child["n"].append(choice)
            if _r.random() < mut_rate and pnames:
                choice = _r.choice(pnames)
                if choice in child["p"]:
                    child["p"].remove(choice)
                else:
                    child["p"].append(choice)
        # Mutate generic numeric cfg entries over a broad key bank
        keys = set((child.get("cfg") or {}).keys())
        try:
            keys |= set(getattr(wanderer, "_neuro_cfg", {}).keys())
        except Exception:
            pass
        try:
            kb = st.get("key_bank")
            if kb:
                keys |= set(kb)
        except Exception:
            pass
        # Seed known useful numeric keys into the search space without excluding any others
        keys.add("lr_override")
        keys = [k for k in set(keys) if isinstance(k, str)]
        if keys and _r.random() < mut_rate:
            k = _r.choice(keys)
            try:
                base = float((child.get("cfg", {}).get(k) if k in (child.get("cfg") or {}) else getattr(wanderer, "_neuro_cfg", {}).get(k, 0.0)))
                scale = 1.0 + ((_r.random() * 2.0 - 1.0) * (0.5 + 0.5 * (mut_rate - 0.05)))
                child["cfg"][k] = base * scale
            except Exception:
                pass
            new_pop.append(child)
        st["pop"] = new_pop
        st["gen"] = st.get("gen", 0) + 1
        st["idx"] = 0

    # ---------------- Evolution helpers (no cloning, full rollback) ----------------
    def _snapshot_graph(self, brain: "Brain") -> Dict[str, Any]:
        try:
            neurons = set(getattr(brain, "neurons", {}).keys())
        except Exception:
            neurons = set()
        try:
            syns = list(getattr(brain, "synapses", []) or [])
            syn_ids = set(id(s) for s in syns)
            syn_weights = {id(s): float(getattr(s, "weight", 1.0)) for s in syns}
        except Exception:
            syn_ids = set()
            syn_weights = {}
        return {"neurons": neurons, "syn_ids": syn_ids, "syn_w": syn_weights}

    def _restore_graph(self, brain: "Brain", snap: Dict[str, Any]) -> None:
        try:
            # Remove newly added synapses
            current_syns = list(getattr(brain, "synapses", []) or [])
            before_ids = snap.get("syn_ids", set())
            for s in current_syns:
                if id(s) not in before_ids:
                    try:
                        brain.remove_synapse(s)
                    except Exception:
                        pass
            # Remove newly added neurons
            before_neurons = snap.get("neurons", set())
            for pos, n in list(getattr(brain, "neurons", {}).items()):
                if pos not in before_neurons:
                    try:
                        brain.remove_neuron(n)
                    except Exception:
                        pass
            # Restore original synapse weights
            for s in list(getattr(brain, "synapses", []) or []):
                sid = id(s)
                if sid in snap.get("syn_w", {}):
                    try:
                        s.weight = float(snap["syn_w"][sid])
                    except Exception:
                        pass
        except Exception:
            pass

    def _apply_genome_temp(self, wanderer: "Wanderer", genome: Dict[str, Any]) -> Dict[str, Any]:
        # Apply stacks and paradigms temporarily; merge cfg; return a handle for rollback
        handle = push_temporary_plugins(wanderer, wanderer_types=genome.get("w"), neuro_types=genome.get("n"))
        prev_cfg = dict(getattr(wanderer, "_neuro_cfg", {}) or {})
        toggled = []
        try:
            for nm in genome.get("p", []) or []:
                wanderer.brain.enable_paradigm(nm, enabled=True)
                toggled.append(nm)
        except Exception:
            pass
        try:
            for k, v in (genome.get("cfg") or {}).items():
                wanderer._neuro_cfg[k] = v
        except Exception:
            pass
        return {"handle": handle, "cfg_prev": prev_cfg, "paradigms": toggled}

    def _rollback_temp(self, wanderer: "Wanderer", temp: Dict[str, Any]) -> None:
        try:
            if temp.get("handle") is not None:
                pop_temporary_plugins(wanderer, temp["handle"])
        except Exception:
            pass
        try:
            for nm in temp.get("paradigms", []) or []:
                wanderer.brain.enable_paradigm(nm, enabled=False)
        except Exception:
            pass
        try:
            wanderer._neuro_cfg = temp.get("cfg_prev", {})
        except Exception:
            pass

    def _short_eval(self, wanderer: "Wanderer", start: "Neuron", max_steps: int) -> Dict[str, float]:
        # Run a short walk with lr=0 to measure loss and speed proxy; rollback handled by caller
        stats = {"loss": float("inf"), "speed": 0.0}
        try:
            res = wanderer.walk(max_steps=max(1, int(max_steps)), start=start, lr=0.0)
            # Compute speed proxy from res.step_metrics
            dts = res.get("step_metrics", []) or []
            deltas = [float(m.get("delta", 0.0)) for m in dts]
            speed = max(0.0, - (sum(deltas) / max(1, len(deltas)))) if deltas else 0.0
            stats = {"loss": float(res.get("loss", 0.0)), "speed": float(speed)}
        except Exception:
            pass
        return stats

    def _mutate(self, genome: Dict[str, Any], wanderer: "Wanderer", rate: float) -> Dict[str, Any]:
        import random as _r
        g = {"w": list(genome.get("w", [])), "n": list(genome.get("n", [])), "p": list(genome.get("p", [])), "cfg": dict(genome.get("cfg", {}))}
        wnames = list(_WANDERER_TYPES.keys())
        nnames = list(_NEURO_TYPES.keys())
        pnames = list(_PARADIGM_TYPES.keys()) if '_PARADIGM_TYPES' in globals() else []
        # With rate, add/remove one plugin from each family (no specific selection bias)
        if wnames and _r.random() < rate:
            nm = _r.choice(wnames)
            if nm in g["w"] and _r.random() < 0.5:
                g["w"].remove(nm)
            else:
                if nm not in g["w"]:
                    g["w"].append(nm)
        if nnames and _r.random() < rate:
            nm = _r.choice(nnames)
            if nm in g["n"] and _r.random() < 0.5:
                g["n"].remove(nm)
            else:
                if nm not in g["n"]:
                    g["n"].append(nm)
        if pnames and _r.random() < rate:
            nm = _r.choice(pnames)
            if nm in g["p"] and _r.random() < 0.5:
                g["p"].remove(nm)
            else:
                if nm not in g["p"]:
                    g["p"].append(nm)
        # Numeric param tweak: pick an observed key or create a neutral one
        keys = set(g["cfg"].keys())
        try:
            keys |= set((getattr(wanderer, "_neuro_cfg", {}) or {}).keys())
        except Exception:
            pass
        if keys and _r.random() < rate:
            k = _r.choice(list(keys))
            try:
                base = float(g["cfg"].get(k, getattr(wanderer, "_neuro_cfg", {}).get(k, 0.0)))
                scale = 1.0 + ((_r.random() * 2.0 - 1.0) * 0.5)
                g["cfg"][k] = base * scale
            except Exception:
                pass
        return g

    def _apply_best_persistently(self, wanderer: "Wanderer") -> None:
        st = wanderer._plugin_state.get("hyper", {})
        best = st.get("best")
        if not best:
            return
        try:
            # Replace stacks
            wanderer._wplugins = []
            for nm in best.get("w", []) or []:
                plug = _WANDERER_TYPES.get(str(nm))
                if plug is not None:
                    wanderer._wplugins.append(plug)
            wanderer._neuro_plugins = []
            for nm in best.get("n", []) or []:
                nplug = _NEURO_TYPES.get(str(nm))
                if nplug is not None:
                    wanderer._neuro_plugins.append(nplug)
            # Enable paradigms
            for nm in best.get("p", []) or []:
                try:
                    wanderer.brain.enable_paradigm(nm, enabled=True)
                except Exception:
                    pass
            # Merge cfg
            for k, v in (best.get("cfg") or {}).items():
                wanderer._neuro_cfg[k] = v
        except Exception:
            pass

    def _run_evolution(self, wanderer: "Wanderer", start: "Neuron", steps: int) -> None:
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        eval_steps = int(cfg.get("hyper_eval_steps", 3))
        mut_rate = float(cfg.get("hyper_mut", 0.3))
        st = wanderer._plugin_state.get("hyper", {})
        # Initialize best genome from current stacks and cfg
        base = {"w": [], "n": [], "p": [], "cfg": dict(cfg)}
        # Evaluate base
        base_snap = self._snapshot_graph(wanderer.brain)
        temp = self._apply_genome_temp(wanderer, base)
        base_stats = self._short_eval(wanderer, start, eval_steps)
        self._rollback_temp(wanderer, temp)
        self._restore_graph(wanderer.brain, base_snap)
        best = base
        best_loss = float(base_stats.get("loss", float("inf")))
        best_speed = float(base_stats.get("speed", 0.0))
        for _ in range(max(1, int(steps))):
            cand = self._mutate(best, wanderer, rate=mut_rate)
            snap = self._snapshot_graph(wanderer.brain)
            temp = self._apply_genome_temp(wanderer, cand)
            stats = self._short_eval(wanderer, start, eval_steps)
            self._rollback_temp(wanderer, temp)
            self._restore_graph(wanderer.brain, snap)
            cand_loss = float(stats.get("loss", float("inf")))
            cand_speed = float(stats.get("speed", 0.0))
            # Accept only if BOTH metrics improve (strict dominance) per user objective
            if cand_loss < best_loss and cand_speed > best_speed:
                best = cand
                best_loss = cand_loss
                best_speed = cand_speed
        # Persist best in plugin state
        try:
            st["best"] = best
            st["best_loss"] = best_loss
            st["best_speed"] = best_speed
        except Exception:
            pass


try:
    register_wanderer_type("hyperEvolution", HyperEvolutionPlugin())
except Exception:
    pass


# Neuroplasticity plugin registry (moved). Import registry and registrar.
from .wanderer import register_neuroplasticity_type  # re-exported below
from .wanderer import NEURO_TYPES_REGISTRY as _NEURO_TYPES


class BaseNeuroplasticityPlugin:
    def on_init(self, wanderer: "Wanderer") -> None:
        try:
            report("neuroplasticity", "init", {"type": "base"}, "events")
        except Exception:
            pass

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        try:
            report("neuroplasticity", "step", {"dir": direction, "step": step_index}, "events")
        except Exception:
            pass
        # Optional growth during walk when stuck
        try:
            cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
            grow_on_step = bool(cfg.get("grow_on_step_when_stuck", False))
            max_new = int(cfg.get("max_new_per_walk", 1))
            if not grow_on_step:
                return
            if getattr(current, "outgoing", None) and len(current.outgoing) > 0:
                return
            cur_new = int(wanderer._plugin_state.get("neuro_new_added", 0))
            if cur_new >= max_new:
                return
            brain = wanderer.brain
            avail = []
            try:
                avail = brain.available_indices()
            except Exception:
                pass
            if not avail:
                return
            last_pos = getattr(current, "position", None)
            chosen = None
            for cand in avail:
                try:
                    if cand != last_pos and brain.get_neuron(cand) is None:
                        chosen = cand
                        break
                except Exception:
                    continue
            if chosen is None:
                chosen = avail[0]
            try:
                brain.add_neuron(chosen, tensor=0.0)
            except Exception:
                pass
            try:
                brain.connect(getattr(current, "position"), chosen, direction="uni")
            except Exception:
                return
            wanderer._plugin_state["neuro_new_added"] = cur_new + 1
            report("neuroplasticity", "grow_step", {"from": getattr(current, "position", None), "to": chosen}, "events")
        except Exception:
            pass

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        try:
            visited = getattr(wanderer, "_visited", [])
            if not visited:
                return
            last = visited[-1]
            cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
            grow_if_no_out = bool(cfg.get("grow_if_no_outgoing", True))
            max_new = int(cfg.get("max_new_per_walk", 1))
            enable_prune = bool(cfg.get("enable_prune", False))
            prune_gt = cfg.get("prune_if_outgoing_gt") if "prune_if_outgoing_gt" in cfg else None
            adjust_bias = cfg.get("adjust_bias_on_loss")

            # Adjust bias slightly based on loss sign if requested
            if adjust_bias is not None:
                try:
                    delta = -float(stats.get("loss", 0.0))
                    last.bias = float(last.bias) + (float(adjust_bias) * (1.0 if delta < 0 else -1.0))
                except Exception:
                    pass

            # Prune if too many outgoing
            if enable_prune and prune_gt is not None:
                try:
                    if len(getattr(last, "outgoing", [])) > int(prune_gt):
                        syn = last.outgoing[-1]
                        wanderer.brain.remove_synapse(syn)
                        report("neuroplasticity", "prune", {"from": getattr(last, "position", None)}, "events")
                except Exception:
                    pass

            if getattr(last, "outgoing", None) and len(last.outgoing) > 0:
                return
            if not grow_if_no_out:
                return
            cur_new = int(wanderer._plugin_state.get("neuro_new_added", 0))
            if cur_new >= max_new:
                return
            brain = wanderer.brain
            try:
                avail = brain.available_indices()
            except Exception:
                avail = []
            if not avail:
                return
            # Choose an unused index different from the last neuron's position
            last_pos = getattr(last, "position", None)
            chosen = None
            try:
                for cand in avail:
                    if cand != last_pos and brain.get_neuron(cand) is None:
                        chosen = cand
                        break
            except Exception:
                chosen = None
            if chosen is None:
                # Fallback to the first available even if used; handle errors gracefully
                chosen = avail[0]
            try:
                brain.add_neuron(chosen, tensor=0.0)
            except Exception:
                # If already exists, continue by connecting only if possible
                pass
            idx = chosen
            brain.connect(getattr(last, "position"), idx, direction="uni")
            wanderer._plugin_state["neuro_new_added"] = cur_new + 1
            report("neuroplasticity", "grow", {"from": getattr(last, "position", None), "to": idx}, "events")
        except Exception:
            pass


# Register base plugin by default
register_neuroplasticity_type("base", BaseNeuroplasticityPlugin())

from .wanderer import Wanderer


__all__ += [
    "Wanderer",
    "register_wanderer_type",
    "register_neuroplasticity_type",
]


# -----------------------------
# SelfAttention (moved to its own module)
# -----------------------------
from .selfattention import SelfAttention, register_selfattention_type, attach_selfattention

__all__ += ["SelfAttention", "register_selfattention_type", "attach_selfattention"]


# -----------------------------
# SelfAttention Routine: Conv1D random insertion with rollback
# -----------------------------

class Conv1DRandomInsertionRoutine:
    """Insert a conv1d neuron every N steps; remove if loss not improved.

    - period: check insertion every `period` global steps (from Reporter).
    - eval_after: number of steps to wait after insertion before evaluation.
    - kernel: default kernel for conv1d parameter neuron.

    Routines operate via SelfAttention.after_step hook and only use
    the provided Wanderer/Brain APIs (no imports), as per architecture.
    """

    def __init__(self, period: int = 20, eval_after: int = 10, kernel: Optional[List[float]] = None, max_data_sources: int = 1) -> None:
        self.period = max(1, int(period))
        self.eval_after = max(1, int(eval_after))
        self.kernel = list(kernel) if isinstance(kernel, list) else [1.0, 0.0, -1.0]
        self._active: Optional[Dict[str, Any]] = None
        self.max_data_sources = max(0, int(max_data_sources))

    def _global_step(self, sa: "SelfAttention") -> int:
        try:
            return len(sa.history())
        except Exception:
            return 0

    def _mean_loss(self, sa: "SelfAttention", start: int, end: Optional[int]) -> float:
        try:
            hist = sa.history()
            if end is None or end > len(hist):
                end = len(hist)
            vals: List[float] = []
            for rec in hist[start:end]:
                v = rec.get("current_loss")
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return (sum(vals) / max(1, len(vals))) if vals else float("inf")
        except Exception:
            return float("inf")

    def _random_free_index(self, brain: "Brain"):
        try:
            candidates = list(brain.available_indices())
        except Exception:
            candidates = []
        random.shuffle(candidates)
        for idx in candidates:
            try:
                if brain.get_neuron(idx) is None:
                    return idx
            except Exception:
                continue
        return candidates[0] if candidates else (0,) * int(getattr(brain, "n", 1))

    def _pick_param_sources(self, brain: "Brain", exclude: List["Neuron"]) -> List["Neuron"]:
        # Deterministically select 5 existing neurons not in exclude
        ex = set(id(n) for n in exclude)
        candidates = []
        try:
            for n in getattr(brain, "neurons", {}).values():
                if id(n) not in ex:
                    candidates.append(n)
        except Exception:
            candidates = []
        # Sort deterministically by position tuple then id for stability
        def keyfn(n):
            pos = getattr(n, "position", None)
            return (0, tuple(pos)) if isinstance(pos, tuple) else (1, id(n))
        candidates.sort(key=keyfn)
        return candidates[:5]

    def _splice_conv(self, brain: "Brain", syn: Optional["Synapse"]) -> Dict[str, Any]:
        if syn is not None:
            src = syn.source
            dst = syn.target
        else:
            neurons = list(getattr(brain, "neurons", {}).values())
            if len(neurons) >= 2:
                src, dst = random.sample(neurons, 2)
            else:
                # Not enough existing neurons to define a destination; abort
                raise RuntimeError("Conv1D insertion skipped: not enough existing neurons for destination")

        # Collect 5 pre-existing parameter source neurons
        params = self._pick_param_sources(brain, exclude=[dst])
        if len(params) < 5:
            raise RuntimeError("Conv1D insertion skipped: need 5 pre-existing parameter neurons")

        # Create conv as a base neuron first (no plugin yet), then wire strictly: 5 params -> conv, conv -> dst
        conv = brain.add_neuron(self._random_free_index(brain), tensor=[0.0])
        created_syns: List["Synapse"] = []
        # wire parameters (5 incoming) using existing neurons; mark as params
        for pn in params:
            created_syns.append(brain.connect(getattr(pn, "position"), getattr(conv, "position"), direction="uni", type_name="param"))
        # Optional: attach one or more data sources if available (pre-existing only)
        try:
            all_neurons = [n for n in getattr(brain, "neurons", {}).values()]
        except Exception:
            all_neurons = []
        used = set(id(x) for x in ([conv, dst] + list(params)))
        data_sources = []
        for n in all_neurons:
            if id(n) in used:
                continue
            data_sources.append(n)
            if len(data_sources) >= self.max_data_sources:
                break
        for dn in data_sources:
            created_syns.append(brain.connect(getattr(dn, "position"), getattr(conv, "position"), direction="uni", type_name="data"))

        # one outgoing
        created_syns.append(brain.connect(getattr(conv, "position"), getattr(dst, "position"), direction="uni"))

        # Strictly promote to conv1d now; if validation fails, rollback
        try:
            conv.type_name = "conv1d"
            plugin = _NEURON_TYPES.get("conv1d")
            if plugin is not None and hasattr(plugin, "on_init"):
                plugin.on_init(conv)  # type: ignore[attr-defined]
        except Exception:
            for syn_created in list(created_syns):
                try:
                    brain.remove_synapse(syn_created)
                except Exception:
                    pass
            try:
                brain.remove_neuron(conv)
            except Exception:
                pass
            raise

        try:
            info = SelfAttention().validate_neuron_wiring(conv)
            report("selfattention", "conv1d_splice", {"ok": bool(info.get("ok", True))}, "builder")
        except Exception:
            pass

        return {
            "conv": conv,
            "params": params,
            "created_synapses": created_syns,
            "replaced_synapse": syn,
            "src": src,
            "dst": dst,
        }

    def after_step(self, sa: "SelfAttention", ro, wanderer: "Wanderer", step_idx: int, ctx: Dict[str, Any]):
        brain = getattr(wanderer, "brain", None)
        if brain is None:
            return None
        gstep = self._global_step(sa)

        # If evaluating an active insertion
        if self._active is not None:
            inserted = int(self._active.get("insert_gstep", gstep))
            if (gstep - inserted) >= self.eval_after:
                mean_after = self._mean_loss(sa, inserted, inserted + self.eval_after)
                baseline = float(self._active.get("baseline_current_loss", float("inf")))
                improved = mean_after < baseline
                try:
                    report("selfattention", "conv1d_eval", {"baseline": baseline, "mean_after": mean_after, "improved": improved}, "builder")
                except Exception:
                    pass
                if not improved:
                    # Remove created structures and restore replaced synapse
                    try:
                        for syn in list(self._active.get("created_synapses", [])):
                            try:
                                brain.remove_synapse(syn)
                            except Exception:
                                pass
                        try:
                            brain.remove_neuron(self._active["conv"])  # type: ignore[index]
                        except Exception:
                            pass
                        for pn in self._active.get("params", []):
                            try:
                                brain.remove_neuron(pn)
                            except Exception:
                                pass
                        rs = self._active.get("replaced_synapse")
                        if rs is not None:
                            try:
                                brain.connect(getattr(self._active["src"], "position"), getattr(self._active["dst"], "position"), direction=getattr(rs, "direction", "uni"), age=getattr(rs, "age", 0), type_name=getattr(rs, "type_name", None))
                            except Exception:
                                pass
                        report("selfattention", "conv1d_removed", {"reason": "no_improvement"}, "builder")
                    except Exception:
                        pass
                else:
                    try:
                        report("selfattention", "conv1d_kept", {"improved": True}, "builder")
                    except Exception:
                        pass
                self._active = None
            return None

        # Trigger insertion periodically
        if gstep > 0 and (gstep % self.period) == 0:
            try:
                syns = [s for s in getattr(brain, "synapses", []) if getattr(s, "direction", "uni") == "uni"]
            except Exception:
                syns = []
            chosen = random.choice(syns) if syns else None
            try:
                structures = self._splice_conv(brain, chosen)
                # Baseline current per-step loss from last step
                baseline = float("inf")
                try:
                    last = sa.history(1)
                    if last:
                        v = last[-1].get("current_loss")
                        if v is not None:
                            baseline = float(v)
                except Exception:
                    pass
                self._active = {**structures, "insert_gstep": gstep, "baseline_current_loss": baseline}
                try:
                    report("selfattention", "conv1d_inserted", {"gstep": gstep}, "builder")
                except Exception:
                    pass
            except Exception:
                # Ignore failures this cycle
                pass
        return None


# Register routine type for easy wiring via SelfAttention(type_name=...)
try:
    register_selfattention_type("conv1d_random_inserter", Conv1DRandomInsertionRoutine())
    __all__ += ["Conv1DRandomInsertionRoutine"]
except Exception:
    pass

# -----------------------------
# High-level Helpers
# -----------------------------

def run_wanderer_training(
    brain: "Brain",
    *,
    num_walks: int = 10,
    max_steps: int = 10,
    lr: float = 1e-2,
    start_selector: Optional[Callable[["Brain"], Optional["Neuron"]]] = None,
    wanderer_type: Optional[str] = None,
    seed: Optional[int] = None,
    loss: Optional[Union[str, Callable[..., Any], Any]] = None,
    target_provider: Optional[Callable[[Any], Any]] = None,
    callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run multiple wanderer walks as a simple training loop.

    - brain: target Brain
    - num_walks: number of walks/episodes
    - max_steps: maximum steps per walk
    - lr: learning rate per walk
    - start_selector: optional callable to choose a starting neuron per walk
    - wanderer_type: optional plugin type name for Wanderer
    - seed: RNG seed for Wanderer
    - loss: None, callable, or string like 'nn.MSELoss'
    - target_provider: optional target builder for built-in nn losses
    - callback: optional hook called as callback(i, stats) per walk

    Returns dict with 'history' list of walk stats and aggregate 'final_loss'.
    """
    w = Wanderer(brain, type_name=wanderer_type, seed=seed, loss=loss, target_provider=target_provider)
    history: List[Dict[str, Any]] = []
    for i in range(num_walks):
        start = start_selector(brain) if start_selector is not None else None
        stats = w.walk(max_steps=max_steps, start=start, lr=lr)
        history.append(stats)
        try:
            report("training", f"walk_{i}", {"loss": stats.get("loss", 0.0), "steps": stats.get("steps", 0)}, "wanderer")
        except Exception:
            pass
        if callback is not None:
            try:
                callback(i, stats)
            except Exception:
                pass
    final_loss = history[-1]["loss"] if history else 0.0
    out = {"history": history, "final_loss": final_loss}
    try:
        report("training", "summary", {"num_walks": num_walks, "final_loss": final_loss}, "wanderer")
    except Exception:
        pass
    return out


from .training import run_wanderer_training
__all__ += ["run_wanderer_training"]


from .reporter import (
    Reporter,
    REPORTER,
    report,
    report_group,
    report_dir,
    clear_report_group,
    export_wanderer_steps_to_jsonl,
)

__all__ += [
    "Reporter",
    "REPORTER",
    "report",
    "report_group",
    "report_dir",
    "clear_report_group",
    "export_wanderer_steps_to_jsonl",
]


def get_last_walk_summary() -> Optional[Dict[str, Any]]:
    """Return the most recent walk summary recorded under training/walks.

    Reads the global REPORTER's ``training/walks`` group and selects the
    entry with the highest numeric suffix. Returns ``None`` when the group
    is empty or missing.
    """

    try:
        walks = REPORTER.group("training", "walks")
    except Exception:
        return None
    if not walks:
        return None

    def _idx(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return -1

    last_key = max(walks.keys(), key=_idx)
    return walks.get(last_key)


__all__ += ["get_last_walk_summary"]

# High-level training helpers (moved)
from .training import (
    run_training_with_datapairs,
    run_wanderer_epochs_with_datapairs,
    run_wanderers_parallel,
    make_default_codec,
    quick_train_on_pairs,
)

__all__ += [
    "run_training_with_datapairs",
    "run_wanderer_epochs_with_datapairs",
    "run_wanderers_parallel",
    "make_default_codec",
    "quick_train_on_pairs",
]


# -----------------------------
# Training with DataPairs
# -----------------------------

from .training import create_start_neuron

def create_start_neuron_old(brain: "Brain", encoded_input: Union[TensorLike, Sequence[float], float, int]) -> "Neuron":
    """Create and return a start neuron in the brain and inject encoded input.

    - Picks the first available index from `brain.available_indices()` or a fallback index (0,...).
    - Creates a neuron with a dummy tensor, then calls `receive(encoded_input)` to set data.
    """
    try:
        avail = brain.available_indices()
        idx = avail[0] if avail else (0,) * int(getattr(brain, "n", 1))
    except Exception:
        idx = (0,)
    n = brain.add_neuron(idx, tensor=0.0)
    n.receive(encoded_input)
    try:
        report("training", "create_start_neuron", {"position": getattr(n, "position", None)}, "datapair")
    except Exception:
        pass
    return n

def run_training_with_datapairs(
    brain: "Brain",
    datapairs: Iterable[Union[DataPair, Tuple[Any, Any], Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]]]],
    codec: "UniversalTensorCodec",
    *,
    steps_per_pair: int = 5,
    lr: float = 1e-2,
    wanderer_type: Optional[str] = None,
    seed: Optional[int] = None,
    loss: Optional[Union[str, Callable[..., Any], Any]] = "nn.MSELoss",
    left_to_start: Optional[Callable[[Any, "Brain"], Optional["Neuron"]]] = None,
    neuro_config: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[int, Dict[str, Any], DataPair], None]] = None,
    gradient_clip: Optional[Dict[str, Any]] = None,
    selfattention: Optional["SelfAttention"] = None,
) -> Dict[str, Any]:
    """Train over a sequence of DataPairs and return aggregate stats.

    - datapairs: elements can be DataPair, raw (left,right) objects, or encoded
      ((enc_left, enc_right)) as produced by DataPair.encode(codec).
    - codec: UniversalTensorCodec used for encoding/decoding when needed.
    - left_to_start: optional function mapping left object -> starting Neuron.
    - loss: same semantics as in Wanderer; default uses nn.MSELoss.

    Returns: {"history": [per-pair stats], "final_loss": float, "count": int}.
    """
    history: List[Dict[str, Any]] = []
    count = 0

    def _normalize_pair(p: Union[DataPair, Tuple[Any, Any], Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]]]) -> DataPair:
        if isinstance(p, DataPair):
            return p
        if isinstance(p, tuple) and len(p) == 2:
            a, b = p
            # Heuristic: if elements look like token sequences/tensors (ints), try decode
            try:
                need_decode = False
                for side in (a, b):
                    if isinstance(side, (list, tuple)) and (len(side) == 0 or isinstance(side[0], int)):
                        need_decode = True
                    elif hasattr(side, "dtype") and hasattr(side, "numel"):
                        need_decode = True
                if need_decode:
                    dp = DataPair.decode((a, b), codec)
                else:
                    dp = DataPair(a, b)
                return dp
            except Exception:
                # Fallback to raw
                return DataPair(a, b)
        # Fallback, wrap as-is
        return DataPair(p, None)  # type: ignore[arg-type]

    # Compute dataset signature for consistency enforcement
    def _dataset_sig(iterable) -> str:
        h = hashlib.sha256()
        max_items = 64
        for idx, it in enumerate(iterable):
            if idx >= max_items:
                break
            dp = _normalize_pair(it)
            enc_l, enc_r = dp.encode(codec)
            # Use lengths and first 32 tokens for signature
            def sig_chunk(x):
                try:
                    if hasattr(x, "tolist"):
                        lst = x.view(-1).tolist()
                    else:
                        lst = list(x)
                except Exception:
                    lst = []
                return bytes(int(v) & 0xFF for v in (lst[:32] if isinstance(lst, list) else []))
            h.update(sig_chunk(enc_l))
            h.update(sig_chunk(enc_r))
        return h.hexdigest()

    dataset_list = list(datapairs)
    sig = _dataset_sig(dataset_list)
    if getattr(brain, "_dataset_signature", None) is None:
        brain._dataset_signature = sig  # type: ignore[attr-defined]
    else:
        if brain._dataset_signature != sig and not getattr(brain, "allow_dissimilar_datasets_in_wanderers", False):
            raise ValueError("Dataset mismatch across wanderers on the same brain; set allow_dissimilar_datasets_in_wanderers=True to override")

    # Build a single Wanderer instance for all pairs; target set per-pair
    _current_target: Dict[str, Any] = {"val": None}

    def _target_provider(_y: Any) -> Any:
        return _current_target["val"]

    w = Wanderer(
        brain,
        type_name=wanderer_type,
        seed=seed,
        loss=loss,
        target_provider=_target_provider,
        neuro_config=neuro_config,
        gradient_clip=gradient_clip,
    )
    if selfattention is not None:
        attach_selfattention(w, selfattention)
    # Allow any enabled learning paradigms on the brain to configure the Wanderer
    try:
        apply_paradigms_to_wanderer(brain, w)
    except Exception:
        pass

    for i, item in enumerate(dataset_list):
        dp = _normalize_pair(item)

        # Encode BOTH parts strictly before running the wanderer
        enc_left, enc_right = dp.encode(codec)

        # Build or choose a start neuron and inject the encoded left once
        start: Optional[Neuron]
        if left_to_start is not None:
            # By convention pass the encoded left to selector
            start = left_to_start(enc_left, brain)  # type: ignore[arg-type]
        else:
            # Choose an existing neuron if available; else create one
            try:
                start = next(iter(brain.neurons.values())) if getattr(brain, "neurons", None) else None  # type: ignore[attr-defined]
            except Exception:
                start = None
            if start is None:
                try:
                    avail = brain.available_indices()
                    if avail:
                        idx = avail[0]
                        start = brain.add_neuron(idx, tensor=0.0)
                except Exception:
                    start = None
        if start is not None:
            start.receive(enc_left)  # inject encoded input into the first neuron
        else:
            start = create_start_neuron(brain, enc_left)

        # Set per-pair target for the shared Wanderer
        _current_target["val"] = enc_right
        stats = w.walk(max_steps=steps_per_pair, start=start, lr=lr)
        history.append(stats)
        count += 1
        try:
            report(
                "training",
                f"datapair_{i}",
                {
                    "loss": stats.get("loss", 0.0),
                    "steps": stats.get("steps", 0),
                    "left_type": type(dp.left).__name__,
                    "right_type": type(dp.right).__name__,
                },
                "datapair",
            )
        except Exception:
            pass
        if callback is not None:
            try:
                callback(i, stats, dp)
            except Exception:
                pass

    final_loss = history[-1]["loss"] if history else 0.0
    out = {"history": history, "final_loss": final_loss, "count": count}
    try:
        report("training", "datapair_summary", {"count": count, "final_loss": final_loss}, "datapair")
    except Exception:
        pass
    return out


__all__ += ["run_training_with_datapairs"]


def run_wanderer_epochs_with_datapairs(
    brain: "Brain",
    datapairs: Iterable[Union[DataPair, Tuple[Any, Any], Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]]]],
    codec: "UniversalTensorCodec",
    *,
    num_epochs: int = 1,
    steps_per_pair: int = 5,
    lr: float = 1e-2,
    wanderer_type: Optional[str] = None,
    seed: Optional[int] = None,
    loss: Optional[Union[str, Callable[..., Any], Any]] = "nn.MSELoss",
    left_to_start: Optional[Callable[[Any, "Brain"], Optional["Neuron"]]] = None,
    callback: Optional[Callable[[int, int, Dict[str, Any], DataPair], None]] = None,
) -> Dict[str, Any]:
    """Run multiple epochs; each epoch runs all datapairs once through the Wanderer.

    Returns: {epochs: [{history, final_loss, delta_vs_prev}], final_loss}.
    Logs per-epoch and summary via REPORTER under group "training/epochs".
    """
    dataset: List[DataPair] = []
    for item in datapairs:
        if isinstance(item, DataPair):
            dataset.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            dataset.append(DataPair(item[0], item[1]))
        else:
            dataset.append(DataPair(item, None))  # type: ignore[arg-type]

    prev_final = None
    epochs: List[Dict[str, Any]] = []
    for e in range(num_epochs):
        # Reuse high-level helper for one pass over dataset
        res = run_training_with_datapairs(
            brain,
            dataset,
            codec,
            steps_per_pair=steps_per_pair,
            lr=lr,
            wanderer_type=wanderer_type,
            seed=seed,
            loss=loss,
            left_to_start=left_to_start,
            callback=(lambda i, stats, dp: callback(e, i, stats, dp)) if callback is not None else None,
        )
        final_loss = res.get("final_loss", 0.0)
        delta = None if prev_final is None else (final_loss - prev_final)
        prev_final = final_loss
        entry = {"history": res.get("history", []), "final_loss": final_loss, "delta_vs_prev": delta}
        epochs.append(entry)
        try:
            report("training", f"epoch_{e}", {"final_loss": final_loss, "delta": delta}, "epochs")
        except Exception:
            pass
    out = {"epochs": epochs, "final_loss": prev_final if prev_final is not None else 0.0}
    try:
        report("training", "epochs_summary", {"num_epochs": num_epochs, "final_loss": out["final_loss"]}, "epochs")
    except Exception:
        pass
    return out


__all__ += ["run_wanderer_epochs_with_datapairs", "create_start_neuron"]


# -----------------------------
# Multi-Wanderer Orchestration
# -----------------------------

def run_wanderers_parallel(
    brain: "Brain",
    datasets: List[Iterable[Union[DataPair, Tuple[Any, Any], Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]]]]],
    codec: "UniversalTensorCodec",
    *,
    mode: str = "thread",  # "thread" or "process"
    steps_per_pair: int = 5,
    lr: float = 1e-2,
    wanderer_type: Optional[str] = None,
    seeds: Optional[List[Optional[int]]] = None,
    loss: Optional[Union[str, Callable[..., Any], Any]] = "nn.MSELoss",
    left_to_start: Optional[Callable[[Any, "Brain"], Optional["Neuron"]]] = None,
    neuro_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run multiple Wanderers on the same brain sequentially or concurrently.

    - datasets: list of iterable datasets (one per wanderer). All must match in
      signature unless brain.allow_dissimilar_datasets_in_wanderers is True.
    - mode: "thread" (concurrent threads) or "process" (real processes).

    Returns list of results per wanderer (same format as run_training_with_datapairs).
    """
    # Compute dataset signatures and enforce similarity
    sigs: List[str] = []
    normed_lists: List[List[Any]] = []
    for ds in datasets:
        lst = list(ds)
        normed_lists.append(lst)
        h = hashlib.sha256()
        # Sample hash via encoding first few pairs
        c = 0
        for item in lst:
            if c >= 64:
                break
            dp = item if isinstance(item, DataPair) else DataPair(item[0], item[1])  # type: ignore[index]
            enc_l, enc_r = dp.encode(codec)
            def chunk(x):
                try:
                    if hasattr(x, "tolist"):
                        return bytes(int(v) & 0xFF for v in x.view(-1).tolist()[:32])
                    return bytes(int(v) & 0xFF for v in list(x)[:32])
                except Exception:
                    return b""
            h.update(chunk(enc_l)); h.update(chunk(enc_r))
            c += 1
        sigs.append(h.hexdigest())
    base_sig = sigs[0] if sigs else ""
    for s in sigs[1:]:
        if s != base_sig and not getattr(brain, "allow_dissimilar_datasets_in_wanderers", False):
            raise ValueError("All wanderers must use the same dataset unless allow_dissimilar_datasets_in_wanderers=True")

    results: List[Optional[Dict[str, Any]]] = [None] * len(normed_lists)

    if mode == "thread":
        import threading

        def runner(idx: int):
            seed = None if seeds is None or idx >= len(seeds) else seeds[idx]
            res = run_training_with_datapairs(
                brain,
                normed_lists[idx],
                codec,
                steps_per_pair=steps_per_pair,
                lr=lr,
                wanderer_type=wanderer_type,
                seed=seed,
                loss=loss,
                left_to_start=left_to_start,
            )
            results[idx] = res

        threads = []
        for i in range(len(normed_lists)):
            t = threading.Thread(target=runner, args=(i,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return [r for r in results if r is not None]
    elif mode == "process":
        # Placeholder orchestration; full shared-brain IPC is substantial and will be added next.
        # For now, raise to avoid silent fallback that would violate requirements.
        raise NotImplementedError("process mode requires Brain host/IPC; request implementation to enable real multi-process wanderers")
    else:
        raise ValueError("mode must be 'thread' or 'process'")


__all__ += ["run_wanderers_parallel"]


# -----------------------------
# Dataset Convenience: Wine (scikit-learn)
# -----------------------------

def _try_load_wine():
    try:
        sklearn = importlib.import_module("sklearn")
        datasets = importlib.import_module("sklearn.datasets")
        return datasets.load_wine()
    except Exception as e:
        raise ImportError(
            "scikit-learn is required to load the wine dataset. Install with 'py -3 -m pip install scikit-learn'."
        ) from e


def run_wine_hello_world(
    *,
    log_path: str = "wanderer_steps.jsonl",
    grid_size: Tuple[int, int] = (8, 8),
    num_pairs: Optional[int] = 50,
    steps_per_pair: int = 3,
    lr: float = 5e-3,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Hello-world training on scikit-learn's Wine dataset with per-step logging.

    - Loads Wine dataset (features -> left, class index -> right).
    - Trains using `run_training_with_datapairs` with neuroplasticity active (default base plugin).
    - Logs per-wanderer-step metrics (time, dt, current/prev/mean loss, neuron/synapse counts) to REPORTER and writes JSONL to `log_path`.

    Returns: training result dict with history and final_loss.
    """
    data = _try_load_wine()
    X = data["data"] if isinstance(data, dict) else data.data
    y = data["target"] if isinstance(data, dict) else data.target
    # Build datapairs; left as list of floats, right as int label
    pairs: List[DataPair] = []
    n = len(X)
    limit = n if num_pairs is None else min(n, int(num_pairs))
    for i in range(limit):
        left = [float(v) for v in X[i]]
        right = int(y[i])
        pairs.append(make_datapair(left, right))

    brain = Brain(2, size=grid_size)
    codec = UniversalTensorCodec()

    res = run_training_with_datapairs(
        brain,
        pairs,
        codec,
        steps_per_pair=steps_per_pair,
        lr=lr,
        seed=seed,
        loss="nn.MSELoss",
        neuro_config={
            "grow_on_step_when_stuck": True,
            "max_new_per_walk": 5,
        },
        gradient_clip={
            "method": "norm",
            "max_norm": 1.0,
            "norm_type": 2.0,
        },
    )

    export_wanderer_steps_to_jsonl(log_path)
    try:
        report("training", "wine_hello_world", {"final_loss": res.get("final_loss", 0.0), "logged_path": log_path}, "examples")
    except Exception:
        pass
    return res

__all__ += ["run_wine_hello_world"]


# -----------------------------
# Convenience Helpers
# -----------------------------

def make_default_codec() -> "UniversalTensorCodec":
    """Return a default `UniversalTensorCodec` instance.

    This helper centralizes codec creation for callers that want a simple,
    one-liner without importing the class name explicitly.
    """
    try:
        report("codec", "make_default_codec", {"ok": True}, "helpers")
    except Exception:
        pass
    return UniversalTensorCodec()


def quick_train_on_pairs(
    pairs: Iterable[Union[DataPair, Tuple[Any, Any], Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]]]],
    *,
    grid_size: Tuple[int, int] = (4, 4),
    steps_per_pair: int = 3,
    lr: float = 1e-2,
    seed: Optional[int] = None,
    wanderer_type: Optional[str] = None,
    codec: Optional["UniversalTensorCodec"] = None,
    neuro_config: Optional[Dict[str, Any]] = None,
    gradient_clip: Optional[Dict[str, Any]] = None,
    selfattention: Optional["SelfAttention"] = None,
) -> Dict[str, Any]:
    """High-level convenience to train quickly on a small 2D brain.

    - Builds a 2D `Brain` of size `grid_size` and a default codec if none provided.
    - Runs `run_training_with_datapairs` with provided knobs and returns the result dict.
    - Logs a concise summary via `REPORTER` under `training/quick`.
    """
    brain = Brain(2, size=grid_size)
    cdc = codec if codec is not None else UniversalTensorCodec()
    res = run_training_with_datapairs(
        brain,
        pairs,
        cdc,
        steps_per_pair=steps_per_pair,
        lr=lr,
        wanderer_type=wanderer_type,
        seed=seed,
        neuro_config=neuro_config,
        gradient_clip=gradient_clip,
        selfattention=selfattention,
    )
    try:
        report(
            "training",
            "quick_train_on_pairs",
            {
                "final_loss": res.get("final_loss"),
                "count": res.get("count"),
                "grid_size": list(grid_size),
                "steps_per_pair": steps_per_pair,
                "lr": lr,
                "wanderer_type": wanderer_type,
            },
            "quick",
        )
    except Exception:
        pass
    return res

__all__ += ["make_default_codec", "quick_train_on_pairs"]
# GUI: PyQt6 Modern App (lazy-import)
# -----------------------------

def launch_gui(config: Optional[Dict[str, Any]] = None) -> None:
    """Launch a PyQt6 GUI for Marble with a modern theme.

    Notes:
    - Imports PyQt6 lazily inside this function so tests importing this module
      do not require PyQt6.
    - Uses existing high-level helpers (`run_training_with_datapairs`) and the
      global `REPORTER` for live logs.
    - Prefers CUDA tensors automatically via existing device helpers.
    - This function blocks until the window is closed.
    """
    # Lazy imports to keep module import light and test-friendly
    try:
        from PyQt6.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QPushButton,
            QLabel,
            QTextEdit,
            QLineEdit,
            QListWidget,
            QStackedWidget,
            QSplitter,
            QTreeWidget,
            QTreeWidgetItem,
            QFormLayout,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
            QFileDialog,
            QMessageBox,
            QTableWidget,
            QTableWidgetItem,
            QStatusBar,
            QCheckBox,
        )
        from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
        from PyQt6.QtGui import QPalette, QColor, QAction
    except Exception as e:
        raise ImportError(
            "PyQt6 is required for the GUI. Install with 'py -3 -m pip install pyqt6' (CPU-only)."
        ) from e

    class TrainingThread(QThread):
        finished_stats = pyqtSignal(dict)
        error = pyqtSignal(str)
        log = pyqtSignal(str)

        def __init__(
            self,
            pairs_text: str,
            grid_w: int,
            grid_h: int,
            steps_per_pair: int,
            lr: float,
            seed: Optional[int],
            wanderer_type: Optional[str],
        ) -> None:
            super().__init__()
            self.pairs_text = pairs_text
            self.grid_w = int(grid_w)
            self.grid_h = int(grid_h)
            self.steps_per_pair = int(steps_per_pair)
            self.lr = float(lr)
            self.seed = seed
            self.wanderer_type = wanderer_type
            self.brain = None  # will be set after run
            self.codec = None

        def _parse_pairs(self) -> List[DataPair]:
            pairs: List[DataPair] = []
            txt = self.pairs_text.strip()
            if not txt:
                # Provide a tiny demo dataset if user left it empty
                demo = [("foo", "bar"), ("baz", "qux"), ("lorem", "ipsum")]
                for a, b in demo:
                    pairs.append(make_datapair(a, b))
                return pairs
            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Allow formats: left -> right, or JSON list like [1,2,3] -> 4
                if "->" in line:
                    left_raw, right_raw = line.split("->", 1)
                    left_raw = left_raw.strip()
                    right_raw = right_raw.strip()
                else:
                    # Single token line becomes (line, line)
                    left_raw = right_raw = line
                def parse_side(s: str) -> Any:
                    try:
                        # Try JSON first
                        return json.loads(s)
                    except Exception:
                        return s
                left = parse_side(left_raw)
                right = parse_side(right_raw)
                pairs.append(make_datapair(left, right))
            return pairs

        def run(self) -> None:
            try:
                # Build brain + codec, then run training with live REPORTER logs
                self.brain = Brain(2, size=(self.grid_w, self.grid_h))
                self.codec = UniversalTensorCodec()
                pairs = self._parse_pairs()
                self.log.emit(f"Starting training on {len(pairs)} pairs...")
                res = run_training_with_datapairs(
                    self.brain,
                    pairs,
                    self.codec,
                    steps_per_pair=self.steps_per_pair,
                    lr=self.lr,
                    wanderer_type=self.wanderer_type,
                    seed=self.seed,
                )
                self.finished_stats.emit(res)
            except Exception as exc:
                self.error.emit(str(exc))

    def apply_modern_theme(app: QApplication, *, dark: bool = True) -> None:
        app.setStyle("Fusion")
        pal = QPalette()
        if dark:
            pal.setColor(QPalette.ColorRole.Window, QColor(37, 37, 38))
            pal.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
            pal.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
            pal.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 48))
            pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(37, 37, 38))
            pal.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
            pal.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
            pal.setColor(QPalette.ColorRole.Button, QColor(45, 45, 48))
            pal.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
            pal.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
            pal.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
            pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        else:
            pal = app.palette()  # use default light palette
        app.setPalette(pal)

    class ReporterPanel(QWidget):
        def __init__(self) -> None:
            super().__init__()
            layout = QHBoxLayout(self)
            self.tree = QTreeWidget()
            self.tree.setHeaderLabels(["Group", "Items"])
            layout.addWidget(self.tree, 1)
            self.detail = QTextEdit()
            self.detail.setReadOnly(True)
            layout.addWidget(self.detail, 2)
            self.tree.itemSelectionChanged.connect(self._on_select)
            self.refresh()

        def _populate(self, node: Dict[str, Any], parent: Optional[QTreeWidgetItem] = None, name: str = "root") -> None:
            items = node.get("items", []) if isinstance(node, dict) else []
            subs = node.get("subgroups", {}) if isinstance(node, dict) else {}
            label = name
            qitem = QTreeWidgetItem([label, ", ".join(items)])
            if parent is None:
                self.tree.addTopLevelItem(qitem)
            else:
                parent.addChild(qitem)
            for subname, subnode in subs.items():
                self._populate(subnode, qitem, subname)

        def refresh(self) -> None:
            self.tree.clear()
            try:
                dirt = report_dir()
            except Exception:
                dirt = {}
            for gname, node in dirt.items():
                self._populate(node, None, gname)

        def _on_select(self) -> None:
            items = self.tree.selectedItems()
            if not items:
                self.detail.clear()
                return
            path = []
            cur = items[0]
            while cur is not None:
                path.insert(0, cur.text(0))
                cur = cur.parent()
            try:
                # Show items dict at that group path
                grp = REPORTER.group(*path) if path else {}
            except Exception:
                grp = {}
            try:
                self.detail.setPlainText(json.dumps(grp, indent=2, default=str))
            except Exception:
                self.detail.setPlainText(str(grp))

    class DashboardPanel(QWidget):
        def __init__(self) -> None:
            super().__init__()
            lay = QVBoxLayout(self)
            self.info = QLabel("Marble GUI — ready.")
            self.info.setWordWrap(True)
            lay.addWidget(self.info)
            self.stats = QTableWidget(0, 2)
            self.stats.setHorizontalHeaderLabels(["Metric", "Value"])
            lay.addWidget(self.stats)

        def set_stats(self, stats: Dict[str, Any]) -> None:
            rows = list(stats.items())
            self.stats.setRowCount(len(rows))
            for i, (k, v) in enumerate(rows):
                self.stats.setItem(i, 0, QTableWidgetItem(str(k)))
                self.stats.setItem(i, 1, QTableWidgetItem(str(v)))

    class TrainingPanel(QWidget):
        def __init__(self, on_started) -> None:
            super().__init__()
            self._thread: Optional[TrainingThread] = None
            self._brain = None
            self._on_started = on_started

            outer = QVBoxLayout(self)
            form = QFormLayout()
            self.grid_w = QSpinBox(); self.grid_w.setRange(1, 256); self.grid_w.setValue(8)
            self.grid_h = QSpinBox(); self.grid_h.setRange(1, 256); self.grid_h.setValue(8)
            self.steps = QSpinBox(); self.steps.setRange(1, 1000); self.steps.setValue(5)
            self.lr = QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(1e-6, 1.0); self.lr.setSingleStep(1e-3); self.lr.setValue(1e-2)
            self.seed = QLineEdit(); self.seed.setPlaceholderText("optional integer seed")
            self.wtype = QLineEdit(); self.wtype.setPlaceholderText("wanderer plugins, e.g. bestlosspath,wanderalongsynapseweights")
            form.addRow("Grid Width", self.grid_w)
            form.addRow("Grid Height", self.grid_h)
            form.addRow("Steps/Pair", self.steps)
            form.addRow("Learning Rate", self.lr)
            form.addRow("Seed", self.seed)
            form.addRow("Wanderer Type", self.wtype)
            outer.addLayout(form)

            self.pairs = QTextEdit()
            self.pairs.setPlaceholderText("Enter datapairs, one per line:\nleft -> right\n[1,2,3] -> 4\n# or leave empty for a demo dataset")
            outer.addWidget(self.pairs)

            btns = QHBoxLayout()
            self.run = QPushButton("Run Training")
            self.export = QPushButton("Export Step Logs (JSONL)")
            self.export.setEnabled(False)
            btns.addWidget(self.run)
            btns.addWidget(self.export)
            outer.addLayout(btns)

            self.out = QTextEdit(); self.out.setReadOnly(True)
            outer.addWidget(self.out)

            self.run.clicked.connect(self._start)
            self.export.clicked.connect(self._export_logs)

        def _start(self) -> None:
            if self._thread is not None and self._thread.isRunning():
                return
            seed_val: Optional[int]
            s = self.seed.text().strip()
            seed_val = int(s) if s.isdigit() else None
            wt = self.wtype.text().strip() or None
            self._thread = TrainingThread(
                self.pairs.toPlainText(),
                self.grid_w.value(),
                self.grid_h.value(),
                self.steps.value(),
                float(self.lr.value()),
                seed_val,
                wt,
            )
            self._thread.log.connect(lambda m: self._append(m))
            self._thread.error.connect(lambda e: self._append(f"Error: {e}"))
            def on_done(res: dict):
                self._brain = self._thread.brain
                self._append(f"Finished. final_loss={res.get('final_loss')}, count={res.get('count')}")
                self.export.setEnabled(True)
                if callable(self._on_started):
                    try:
                        self._on_started(res, self._brain)
                    except Exception:
                        pass
            self._thread.finished_stats.connect(on_done)
            self._append("Queued training job...")
            self._thread.start()

        def _append(self, msg: str) -> None:
            self.out.append(msg)

        def _export_logs(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Logs", "wanderer_steps.jsonl", "JSONL (*.jsonl)")
            if not path:
                return
            try:
                export_wanderer_steps_to_jsonl(path)
                QMessageBox.information(self, "Export", f"Exported step logs to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

        def brain(self):
            return self._brain

    class GraphPanel(QWidget):
        def __init__(self) -> None:
            super().__init__()
            lay = QVBoxLayout(self)
            self.info = QLabel("Graph summary will appear after training.")
            self.info.setWordWrap(True)
            lay.addWidget(self.info)
            self.text = QTextEdit(); self.text.setReadOnly(True)
            lay.addWidget(self.text)

        def set_brain(self, brain: Optional[Brain]) -> None:  # type: ignore[name-defined]
            if brain is None:
                self.text.setPlainText("")
                return
            try:
                neurons = getattr(brain, "neurons", {})
                syns = getattr(brain, "synapses", [])
                lines = [
                    f"Neurons: {len(neurons)}",
                    f"Synapses: {len(syns)}",
                    "Sample neurons (up to 10):",
                ]
                for i, (pos, n) in enumerate(neurons.items()):
                    if i >= 10:
                        break
                    tlen = 0
                    try:
                        t = getattr(n, "tensor", None)
                        if hasattr(t, "numel"):
                            tlen = int(t.numel())
                        elif isinstance(t, list):
                            tlen = len(t)
                        else:
                            tlen = 1
                    except Exception:
                        tlen = 0
                    lines.append(f"  - {pos}, type={getattr(n, 'type_name', None)}, tensor_elems={tlen}")
                self.text.setPlainText("\n".join(lines))
            except Exception as e:
                self.text.setPlainText(f"Error summarizing brain: {e}")

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Marble GUI")
            self.resize(1100, 700)

            self.status = QStatusBar(); self.setStatusBar(self.status)

            # Navigation
            self.nav = QListWidget()
            self.nav.addItems(["Dashboard", "Training", "Reporter", "Graph", "Settings"])

            # Pages
            self.pages = QStackedWidget()
            self.dashboard = DashboardPanel()
            self.training = TrainingPanel(self._on_training_done)
            self.reporter = ReporterPanel()
            self.graph = GraphPanel()
            self.settings = self._build_settings()
            for w in [self.dashboard, self.training, self.reporter, self.graph, self.settings]:
                self.pages.addWidget(w)

            split = QSplitter()
            split.addWidget(self.nav); split.addWidget(self.pages)
            split.setStretchFactor(1, 1)
            self.setCentralWidget(split)

            self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
            self.nav.setCurrentRow(0)

            # Menu
            m_file = self.menuBar().addMenu("File")
            m_view = self.menuBar().addMenu("View")
            act_export = QAction("Export Step Logs", self)
            act_export.triggered.connect(lambda: self.training._export_logs())
            m_file.addAction(act_export)
            self._dark = True
            act_theme = QAction("Toggle Dark Theme", self)
            def _toggle():
                self._dark = not self._dark
                apply_modern_theme(QApplication.instance(), dark=self._dark)
            act_theme.triggered.connect(_toggle)
            m_view.addAction(act_theme)

            # Live reporter refresh
            self._timer = QTimer(self)
            self._timer.timeout.connect(self.reporter.refresh)
            self._timer.start(1000)

        def _build_settings(self) -> QWidget:
            w = QWidget()
            lay = QFormLayout(w)
            torch_device = "cpu"
            try:
                # Reuse device logic via helper hidden on various classes
                helper = _DeviceHelper()  # type: ignore[name-defined]
                torch_device = getattr(helper, "_device", "cpu")
            except Exception:
                pass
            self.lbl_device = QLabel(torch_device)
            self.chk_autorefresh = QCheckBox("Auto-refresh Reporter (1s)")
            self.chk_autorefresh.setChecked(True)
            self.chk_autorefresh.stateChanged.connect(lambda s: self._timer.start(1000) if s == Qt.CheckState.Checked.value else self._timer.stop())
            lay.addRow("Device", self.lbl_device)
            lay.addRow(self.chk_autorefresh)
            return w

        def _on_training_done(self, stats: Dict[str, Any], brain: Optional[Brain]) -> None:  # type: ignore[name-defined]
            # Update dashboard + graph
            self.dashboard.set_stats({
                "final_loss": stats.get("final_loss"),
                "count": stats.get("count"),
            })
            self.graph.set_brain(brain)

    app = QApplication([])
    apply_modern_theme(app, dark=True)
    win = MainWindow()
    win.show()
    try:
        report("gui", "launch", {"ok": True}, "events")
    except Exception:
        pass
    app.exec()


__all__ += ["launch_gui"]
