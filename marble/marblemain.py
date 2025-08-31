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
import itertools
import gc
from .learnable_param import LearnableParam
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
from .buildingblock import (
    register_buildingblock_type,
    get_buildingblock_type,
    BuildingBlock,
)

from .lobe import Lobe

__all__ += [
    "_DeviceHelper",
    "_NEURON_TYPES",
    "_SYNAPSE_TYPES",
    "register_neuron_type",
    "register_synapse_type",
    "Neuron",
    "Synapse",
    "register_buildingblock_type",
    "get_buildingblock_type",
    "BuildingBlock",
    "Lobe",
]


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


# -----------------------------
# Learning Paradigm Plugins (Brain-level orchestration) (Brain-level orchestration)
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


def _auto_register_local_paradigms() -> None:
    import inspect

    def _camel_to_snake(name: str) -> str:
        out: List[str] = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0 and (name[i - 1].islower() or (i + 1 < len(name) and name[i + 1].islower())):
                out.append("_")
            out.append(ch.lower())
        return "".join(out)

    for nm, obj in list(globals().items()):
        if inspect.isclass(obj) and nm.endswith("Paradigm"):
            register_learning_paradigm_type(_camel_to_snake(nm[:-8]), obj)


_auto_register_local_paradigms()

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
    - kuzu_path: optional filename for a Kuzu graph database mirroring the brain
    - learn_all_numeric_parameters: when True, numeric defaults in subsequently
      imported modules are auto-exposed as learnable parameters

    The brain maintains a discrete occupancy grid; neurons/synapses must be placed
    at indices that are inside this shape.
    """

    def __init__(
        self,
        n: int,
        *,
        size: Optional[Union[int, Sequence[int]]] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        formula: Optional[str] = None,
        max_iters: int = 50,
        escape_radius: float = 2.0,
        mode: str = "grid",
        sparse_bounds: Optional[Sequence[Union[Tuple[float, float], Tuple[float, None], Tuple[float]]]] = None,
        allow_dissimilar_datasets_in_wanderers: bool = False,
        store_snapshots: bool = False,
        snapshot_path: Optional[str] = None,
        snapshot_freq: Optional[int] = None,
        snapshot_keep: Optional[int] = None,
        kuzu_path: Optional[str] = None,
        learn_all_numeric_parameters: bool = False,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self.mode = str(mode)
        self._dataset_signature: Optional[str] = None
        self.allow_dissimilar_datasets_in_wanderers = bool(allow_dissimilar_datasets_in_wanderers)
        self._lock_dir = os.path.join(tempfile.gettempdir(), f"marble_brainlocks_{os.getpid()}_{id(self)}")
        self.learn_all_numeric_parameters = bool(learn_all_numeric_parameters)
        try:
            os.makedirs(self._lock_dir, exist_ok=True)
        except Exception:
            pass

        if self.mode not in ("grid", "sparse"):
            raise ValueError("mode must be 'grid' or 'sparse'")

        # Shared storage for both modes
        self.synapses: List[Synapse] = []

        if self.mode == "grid":
            self.dynamic = size is None
            if not self.dynamic:
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
            else:
                self.size = None
                self.bounds = None  # type: ignore
                self.formula = None
                self.max_iters = int(max_iters)
                self.escape_radius = float(escape_radius)
                self.occupancy: Dict[Tuple[int, ...], bool] = {}
            self.neurons: Dict[Tuple[int, ...], Neuron] = {}
            self._dyn_min: List[int] = [0] * self.n
            self._dyn_max: List[int] = [-1] * self.n
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

        # Snapshot configuration
        self.store_snapshots = bool(store_snapshots)
        self.snapshot_path = snapshot_path
        self.snapshot_freq = int(snapshot_freq) if snapshot_freq is not None else None
        self.snapshot_keep = int(snapshot_keep) if snapshot_keep is not None else None
        if self.store_snapshots:
            if not self.snapshot_path:
                raise ValueError("snapshot_path must be provided when store_snapshots is True")
            if self.snapshot_freq is None:
                raise ValueError("snapshot_freq must be provided when store_snapshots is True")
            try:
                os.makedirs(self.snapshot_path, exist_ok=True)
            except Exception:
                pass

        # Training progress counters for reporting
        self._progress_epoch = 0
        self._progress_total_epochs = 1
        self._progress_walk = 0
        self._progress_total_walks = 1

        # Global learnable parameters for brain-level plugins
        self._learnables: Dict[str, LearnableParam] = {}

        # Named lobes (subgraphs of neurons and synapses)
        self.lobes: Dict[str, Lobe] = {}

        # Lock to guard training operations and ensure thread safety
        self._train_lock = threading.Lock()

        # Optional Kuzu graph mirroring
        self._kuzu_conn = None
        if kuzu_path:
            try:
                import kuzu  # type: ignore

                self._kuzu_db = kuzu.Database(kuzu_path)
                self._kuzu_conn = kuzu.Connection(self._kuzu_db)
                self._kuzu_conn.execute(
                    "CREATE NODE TABLE IF NOT EXISTS Neuron(key STRING, coords STRING, PRIMARY KEY(key))"
                )
                self._kuzu_conn.execute(
                    "CREATE REL TABLE IF NOT EXISTS Synapse(FROM Neuron TO Neuron, direction STRING)"
                )
            except Exception:
                self._kuzu_conn = None

    def _kuzu_key(self, coords: Sequence[float]) -> str:
        return ",".join(str(float(c)) for c in coords)

    def _kuzu_add_neuron(self, coords: Sequence[float]) -> None:
        if self._kuzu_conn is None:
            return
        try:
            self._kuzu_conn.execute(
                "CREATE (:Neuron {key:$key, coords:$coords})",
                {"key": self._kuzu_key(coords), "coords": json.dumps(list(coords))},
            )
            self._kuzu_conn.execute("CHECKPOINT")
        except Exception:
            pass

    def _kuzu_remove_neuron(self, coords: Sequence[float]) -> None:
        if self._kuzu_conn is None:
            return
        try:
            self._kuzu_conn.execute(
                "MATCH (n:Neuron {key:$key}) DETACH DELETE n",
                {"key": self._kuzu_key(coords)},
            )
        except Exception:
            pass

    def _kuzu_add_synapse(
        self, src_coords: Sequence[float], dst_coords: Sequence[float], direction: str
    ) -> None:
        if self._kuzu_conn is None:
            return
        try:
            self._kuzu_conn.execute(
                "MATCH (a:Neuron {key:$src}),(b:Neuron {key:$dst}) CREATE (a)-[:Synapse {direction:$dir}]->(b)",
                {
                    "src": self._kuzu_key(src_coords),
                    "dst": self._kuzu_key(dst_coords),
                    "dir": direction,
                },
            )
            self._kuzu_conn.execute("CHECKPOINT")
        except Exception:
            pass

    def _kuzu_rebuild_all(self) -> None:
        if self._kuzu_conn is None:
            return
        try:
            self._kuzu_conn.execute("MATCH (n) DETACH DELETE n")
            for pos, neuron in self.neurons.items():
                coords = self.world_coords(pos) if self.mode == "grid" else pos
                self._kuzu_conn.execute(
                    "CREATE (:Neuron {key:$key, coords:$coords})",
                    {"key": self._kuzu_key(coords), "coords": json.dumps(list(coords))},
                )
            for syn in self.synapses:
                if isinstance(getattr(syn, "source", None), Neuron) and isinstance(
                    getattr(syn, "target", None), Neuron
                ):
                    s_pos = getattr(syn.source, "position", None)
                    d_pos = getattr(syn.target, "position", None)
                    if s_pos is None or d_pos is None:
                        continue
                    s_coords = self.world_coords(s_pos) if self.mode == "grid" else s_pos
                    d_coords = self.world_coords(d_pos) if self.mode == "grid" else d_pos
                    self._kuzu_conn.execute(
                        "MATCH (a:Neuron {key:$src}),(b:Neuron {key:$dst}) CREATE (a)-[:Synapse {direction:$dir}]->(b)",
                        {
                            "src": self._kuzu_key(s_coords),
                            "dst": self._kuzu_key(d_coords),
                            "dir": getattr(syn, "direction", "uni"),
                        },
                    )
            self._kuzu_conn.execute("CHECKPOINT")
        except Exception:
            pass

    # Prevent accidental copying which could break training immutability
    def __copy__(self):
        raise TypeError("Brain instances are immutable and cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("Brain instances are immutable and cannot be deep-copied")

    # --- Public API ---
    def is_inside(self, index: Sequence[int]) -> bool:
        if self.mode == "grid":
            if getattr(self, "dynamic", False):
                return True
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
            if getattr(self, "dynamic", False):
                return tuple(float(i) for i in idx)
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
            if not getattr(self, "dynamic", False):
                if not self.is_inside(idx):
                    raise ValueError("Neuron index is outside the brain shape")
                if idx in self.neurons:
                    raise ValueError("Neuron already exists at this index")
            neuron = Neuron(tensor, **kwargs)
            setattr(neuron, "position", idx)
            self.neurons[idx] = neuron
            if getattr(self, "dynamic", False):
                for d, v in enumerate(idx):
                    if self._dyn_max[d] < self._dyn_min[d]:
                        self._dyn_min[d] = self._dyn_max[d] = v
                    else:
                        if v < self._dyn_min[d]:
                            self._dyn_min[d] = v
                        if v > self._dyn_max[d]:
                            self._dyn_max[d] = v
            try:
                report("brain", "add_neuron", {"position": idx}, "events")
            except Exception:
                pass
            try:
                self._kuzu_add_neuron(self.world_coords(idx))
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
            try:
                self._kuzu_add_neuron(coords)
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
        if isinstance(src, Neuron) and isinstance(dst, Neuron):
            try:
                s_pos = getattr(src, "position", None)
                d_pos = getattr(dst, "position", None)
                if s_pos is not None and d_pos is not None:
                    s_coords = self.world_coords(s_pos) if self.mode == "grid" else s_pos
                    d_coords = self.world_coords(d_pos) if self.mode == "grid" else d_pos
                    self._kuzu_add_synapse(s_coords, d_coords, direction)
            except Exception:
                pass
        return syn

    def define_lobe(
        self,
        name: str,
        neurons: Sequence[Neuron],
        synapses: Optional[Sequence[Synapse]] = None,
        *,
        inherit_plugins: bool = True,
        plugin_types: Optional[Union[str, Sequence[str]]] = None,
        neuro_config: Optional[Dict[str, Any]] = None,
    ) -> Lobe:
        """Register and return a named lobe built from given neurons/synapses."""
        lobe = Lobe(
            neurons,
            synapses,
            plugin_types=plugin_types,
            neuro_config=neuro_config,
            inherit_plugins=inherit_plugins,
        )
        self.lobes[name] = lobe
        return lobe

    def get_lobe(self, name: str) -> Optional[Lobe]:
        return self.lobes.get(name)

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

    def add_dimension(self) -> None:
        self.n += 1
        if self.mode == "grid":
            self._dyn_min.append(0)
            self._dyn_max.append(-1)
            new_map: Dict[Tuple[int, ...], Neuron] = {}
            for pos, n in list(self.neurons.items()):
                new_pos = tuple(list(pos) + [0])
                setattr(n, "position", new_pos)
                new_map[new_pos] = n
            self.neurons = new_map
        else:
            self.sparse_bounds = self.sparse_bounds + ((0.0, None),)
            new_map: Dict[Tuple[float, ...], Neuron] = {}
            for pos, n in list(self.neurons.items()):
                new_pos = tuple(list(pos) + [0.0])
                setattr(n, "position", new_pos)
                new_map[new_pos] = n
            self.neurons = new_map

    def remove_last_dimension(self) -> None:
        if self.n <= 1:
            return
        if self.mode == "grid":
            if getattr(self, "_dyn_min", None):
                self._dyn_min.pop()
            if getattr(self, "_dyn_max", None):
                self._dyn_max.pop()
        else:
            self.sparse_bounds = self.sparse_bounds[:-1]
        to_remove = [n for n in list(self.neurons.values()) if getattr(n, "position", (0,))[-1] != (0 if self.mode == "grid" else 0.0)]
        for n in to_remove:
            self.remove_neuron(n)
        new_map: Dict[Tuple[Any, ...], Neuron] = {}
        for pos, n in list(self.neurons.items()):
            new_pos = tuple(pos[:-1])
            setattr(n, "position", new_pos)
            new_map[new_pos] = n
        self.neurons = new_map
        self.n -= 1

    # Remove a synapse and clean references
    def remove_synapse(self, synapse: "Synapse") -> None:
        if synapse in self.synapses:
            self.synapses.remove(synapse)
        try:
            src = synapse.source
            if isinstance(src, Synapse):
                if synapse in src.outgoing_synapses:
                    src.outgoing_synapses.remove(synapse)
                if synapse in src.incoming_synapses:
                    src.incoming_synapses.remove(synapse)
            else:
                if synapse in src.outgoing:
                    src.outgoing.remove(synapse)
                if synapse in src.incoming:
                    src.incoming.remove(synapse)
            dst = synapse.target
            if isinstance(dst, Synapse):
                if synapse in dst.outgoing_synapses:
                    dst.outgoing_synapses.remove(synapse)
                if synapse in dst.incoming_synapses:
                    dst.incoming_synapses.remove(synapse)
            else:
                if synapse in dst.outgoing:
                    dst.outgoing.remove(synapse)
                if synapse in dst.incoming:
                    dst.incoming.remove(synapse)
        except Exception:
            pass
        try:
            self._kuzu_rebuild_all()
        except Exception:
            pass
        try:
            report("brain", "remove_synapse", {"direction": synapse.direction}, "events")
        except Exception:
            pass

    # Remove a neuron and bridge its synapses to avoid gaps
    def remove_neuron(self, neuron: "Neuron") -> None:
        pos = getattr(neuron, "position", None)
        incomings = list(getattr(neuron, "incoming", []) or [])
        outgoings = list(getattr(neuron, "outgoing", []) or [])
        try:
            if incomings and outgoings:
                for ins in incomings:
                    for outs in outgoings:
                        if isinstance(ins.target, Neuron):
                            try:
                                ins.target.incoming.remove(ins)
                            except Exception:
                                pass
                        if isinstance(outs.source, Neuron):
                            try:
                                outs.source.outgoing.remove(outs)
                            except Exception:
                                pass
                        ins.target = outs
                        outs.source = ins
                        if outs not in ins.outgoing_synapses:
                            ins.outgoing_synapses.append(outs)
                        if ins not in outs.incoming_synapses:
                            outs.incoming_synapses.append(ins)
            else:
                for syn in incomings + outgoings:
                    if syn in self.synapses:
                        self.remove_synapse(syn)
        except Exception:
            pass
        try:
            pos = getattr(neuron, "position", None)
            if pos is not None and pos in self.neurons:
                try:
                    del self.neurons[pos]
                except Exception:
                    self.neurons = {k: v for k, v in self.neurons.items() if v is not neuron}
            report("brain", "remove_neuron", {"position": pos}, "events")
        except Exception:
            pass
        try:
            self._kuzu_rebuild_all()
        except Exception:
            pass
    # --- Learnable parameter management (global) ---
    def ensure_learnable_param(
        self,
        name: str,
        init_value: Any,
        *,
        requires_grad: bool = True,
        lr: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Any:
        if name in self._learnables:
            return self._learnables[name].tensor
        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        if torch is not None:
            try:
                t = torch.tensor(init_value, dtype=torch.float32, device=device, requires_grad=requires_grad)
            except Exception:
                t = torch.tensor([float(init_value)], dtype=torch.float32, device=device, requires_grad=requires_grad)
        else:
            t = init_value
        self._learnables[name] = LearnableParam(
            tensor=t,
            orig_type=type(init_value),
            lr=lr,
            min_value=min_value,
            max_value=max_value,
        )
        return t

    def set_param_optimization(self, name: str, *, enabled: bool = True, lr: Optional[float] = None) -> None:
        ent = self._learnables.get(name)
        if ent is None:
            return
        ent.opt = bool(enabled)
        if lr is not None:
            ent.lr = float(lr)

    def get_learnable_param_tensor(self, name: str) -> Any:
        ent = self._learnables.get(name)
        return None if ent is None else ent.tensor

    def _collect_enabled_params(self) -> List[LearnableParam]:
        out: List[LearnableParam] = []
        for lp in self._learnables.values():
            if lp.opt:
                out.append(lp)
        return out

    def _update_learnables(self) -> None:
        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore
        if torch is None:
            return
        params = self._collect_enabled_params()
        groups = []
        for lp in params:
            t = lp.tensor
            lr = float(lp.lr if lp.lr is not None else 1e-2)
            if hasattr(t, "grad") and t.grad is not None:
                groups.append({"params": [t], "lr": lr})
        if not groups:
            return
        opt = torch.optim.Adam(groups)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for lp in params:
            lp.apply_constraints()

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
            if getattr(self, "dynamic", False):
                if not self.neurons:
                    out = [(0,) * self.n]
                else:
                    nxt0 = self._dyn_max[0] + 1
                    out = [(nxt0,) + (0,) * (self.n - 1)]
                try:
                    report("brain", "available_indices", {"count": len(out)}, "metrics")
                except Exception:
                    pass
                return out
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

    def size_stats(self) -> Tuple[int, Optional[int]]:
        """Return current neuron count and maximum capacity if fixed."""
        current = len(self.neurons)
        if getattr(self, "dynamic", False) or getattr(self, "size", None) is None:
            return current, None
        cap = 1
        for v in self.size:
            try:
                cap *= int(v)
            except Exception:
                cap = 0
        return current, cap

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

    # --- Snapshot persistence ---
    def save_snapshot(self, path: Optional[str] = None) -> str:
        """Persist full brain state to a single `.marble` file.

        If *path* is None, uses configured ``snapshot_path`` and auto-generates a
        filename ``snapshot_<timestamp>.marble``. Returns the path written.
        """
        target = path
        if target is None:
            if not self.snapshot_path:
                raise ValueError("snapshot_path is not configured")
            ts = int(time.time())
            target = os.path.join(self.snapshot_path, f"snapshot_{ts}.marble")
        if not str(target).endswith(".marble"):
            target = str(target) + ".marble"

        def tensor_to_list(t: Any) -> List[float]:
            try:
                if hasattr(t, "detach") and hasattr(t, "tolist"):
                    return [float(x) for x in t.detach().to("cpu").view(-1).tolist()]
            except Exception:
                pass
            return [float(x) for x in (t if isinstance(t, (list, tuple)) else [t])]

        data: Dict[str, Any] = {
            "version": 1,
            "n": self.n,
            "mode": self.mode,
            "size": list(getattr(self, "size", [])),
            "bounds": [list(b) for b in getattr(self, "bounds", [])],
            "formula": getattr(self, "formula", None),
            "max_iters": getattr(self, "max_iters", None),
            "escape_radius": getattr(self, "escape_radius", None),
            "sparse_bounds": [list(b) for b in getattr(self, "sparse_bounds", [])],
            "neurons": [],
            "synapses": [],
        }
        for pos, neuron in self.neurons.items():  # type: ignore[union-attr]
            data["neurons"].append(
                {
                    "position": list(pos),
                    "weight": neuron.weight,
                    "bias": neuron.bias,
                    "age": neuron.age,
                    "type_name": neuron.type_name,
                    "tensor": tensor_to_list(neuron.tensor),
                }
            )
        for syn in self.synapses:
            src = getattr(syn.source, "position", None)
            dst = getattr(syn.target, "position", None)
            data["synapses"].append(
                {
                    "source": list(src),
                    "target": list(dst),
                    "direction": syn.direction,
                    "age": syn.age,
                    "type_name": syn.type_name,
                    "weight": syn.weight,
                }
            )
        with open(target, "wb") as f:
            pickle.dump(data, f)
        # Retention: keep only the newest N snapshots if configured
        if getattr(self, "snapshot_keep", None) is not None:
            try:
                files = [
                    os.path.join(self.snapshot_path, p)
                    for p in os.listdir(self.snapshot_path)
                    if p.endswith(".marble")
                ]
                files.sort(key=os.path.getmtime, reverse=True)
                for old in files[int(self.snapshot_keep) :]:
                    try:
                        os.remove(old)
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            report("brain", "snapshot_saved", {"path": target}, "io")
        except Exception:
            pass
        return target

    @classmethod
    def load_snapshot(cls, path: str) -> "Brain":
        """Load a brain snapshot previously saved with ``save_snapshot``."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid brain snapshot")
        mode = data.get("mode", "grid")
        if mode == "sparse":
            brain = cls(
                int(data.get("n", 1)),
                mode="sparse",
                sparse_bounds=tuple(tuple(b) for b in data.get("sparse_bounds", [])),
                store_snapshots=False,
            )
        else:
            brain = cls(
                int(data.get("n", 1)),
                size=tuple(data.get("size", [])),
                bounds=tuple(tuple(b) for b in data.get("bounds", [])),
                formula=data.get("formula"),
                max_iters=int(data.get("max_iters", 50)),
                escape_radius=float(data.get("escape_radius", 2.0)),
                store_snapshots=False,
            )
        for item in data.get("neurons", []):
            brain.add_neuron(
                item.get("position", []),
                tensor=item.get("tensor", 0.0),
                weight=item.get("weight", 1.0),
                bias=item.get("bias", 0.0),
                age=item.get("age", 0),
                type_name=item.get("type_name"),
            )
        for syn in data.get("synapses", []):
            brain.connect(
                syn.get("source", []),
                syn.get("target", []),
                direction=syn.get("direction", "uni"),
                age=syn.get("age", 0),
                type_name=syn.get("type_name"),
                weight=syn.get("weight", 1.0),
            )
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


def save_brain_snapshot(brain: "Brain", path: Optional[str] = None) -> str:
    """Convenience wrapper around ``Brain.save_snapshot``."""
    return brain.save_snapshot(path)


def load_brain_snapshot(path: str) -> "Brain":
    """Convenience wrapper around ``Brain.load_snapshot``."""
    return Brain.load_snapshot(path)


__all__ += ["save_brain_snapshot", "load_brain_snapshot"]


# -----------------------------
# Brain Training Plugins + Method
# -----------------------------

_BRAIN_TRAIN_TYPES: Dict[str, Any] = {}


def register_brain_train_type(name: str, plugin: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Brain train type name must be a non-empty string")
    _BRAIN_TRAIN_TYPES[name] = plugin




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


def _after_walk(plugin: Optional[Any], brain: "Brain", wanderer: "Wanderer", i: int, stats: Dict[str, Any]) -> bool:
    if plugin is not None and hasattr(plugin, "after_walk"):
        res = _call_safely(getattr(plugin, "after_walk"), brain, wanderer, i, stats)
        if isinstance(res, dict) and res.get("stop"):
            return True
    return False


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
        stop = False
        for p in plugins:
            if _after_walk(p, brain, wanderer, i, stats):
                stop = True
        _maybe_report("training", f"brain_walk_{i}", {"loss": stats.get("loss", 0.0), "steps": stats.get("steps", 0)}, "brain")
        if callback is not None:
            _call_safely(callback, i, stats)
        if stop:
            break
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

# Import all plugin modules so they self-register with their registries.
from . import plugins as _plugins  # noqa: F401

# Plugin registry for Wanderer (moved). Import registry and registrar.
from .wanderer import register_wanderer_type  # re-exported below
from .wanderer import WANDERER_TYPES_REGISTRY as _WANDERER_TYPES
try:
    # Ensure built-in Wanderer plugins that self-register are loaded on import
    from .plugins.wanderer_weights import WanderAlongSynapseWeightsPlugin as _WPL  # noqa: F401
    from .plugins.wanderer_epsgreedy import EpsilonGreedyChooserPlugin as _EPL  # noqa: F401
except Exception:
    pass


# Neuroplasticity plugin registry (moved). Import registry and registrar.
from .wanderer import register_neuroplasticity_type  # re-exported below
from .wanderer import NEURO_TYPES_REGISTRY as _NEURO_TYPES


try:
    from .plugins.neuroplasticity_base import BaseNeuroplasticityPlugin
    __all__ += ["BaseNeuroplasticityPlugin"]
except Exception:
    pass

from .wanderer import Wanderer, expose_learnable_params


__all__ += [
    "Wanderer",
    "register_wanderer_type",
    "register_neuroplasticity_type",
    "expose_learnable_params",
]


# -----------------------------
# SelfAttention (moved to its own module)
# -----------------------------
from .selfattention import SelfAttention, register_selfattention_type, attach_selfattention
from .plugins.selfattention_conv1d_inserter import Conv1DRandomInsertionRoutine

__all__ += ["SelfAttention", "register_selfattention_type", "attach_selfattention"]


# Trigger automatic plugin discovery
from . import plugins as _plugins  # noqa: F401
from .plugins.wanderer_contrastive_infonce import ContrastiveInfoNCEPlugin
from .plugins.wanderer_td_qlearning import TDQLearningPlugin
from .plugins.wanderer_distillation import DistillationPlugin

__all__ += [
    "ContrastiveInfoNCEPlugin",
    "TDQLearningPlugin",
    "DistillationPlugin",
]

# -----------------------------
# High-level Helpers
# -----------------------------

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
    mixedprecision: bool = True,
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
    w = Wanderer(
        brain,
        type_name=wanderer_type,
        seed=seed,
        loss=loss,
        target_provider=target_provider,
        mixedprecision=mixedprecision,
    )
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
    train_type: Optional[Union[str, Sequence[str]]] = None,
    seed: Optional[int] = None,
    loss: Optional[Union[str, Callable[..., Any], Any]] = "nn.MSELoss",
    left_to_start: Optional[Callable[[Any, "Brain"], Optional["Neuron"]]] = None,
    neuro_config: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[int, Dict[str, Any], DataPair], None]] = None,
      gradient_clip: Optional[Dict[str, Any]] = None,
      selfattention: Optional["SelfAttention"] = None,
      streaming: bool = True,
      batch_size: Optional[int] = None,
      lobe: Optional[Lobe] = None,
      mixedprecision: bool = True,
  ) -> Dict[str, Any]:
    """Train over a sequence of DataPairs and return aggregate stats.

    - datapairs: elements can be DataPair, raw (left,right) objects, or encoded
      ((enc_left, enc_right)) as produced by DataPair.encode(codec).
    - codec: UniversalTensorCodec used for encoding/decoding when needed.
    - left_to_start: optional function mapping left object -> starting Neuron.
    - loss: same semantics as in Wanderer; default uses nn.MSELoss.
    - train_type: optional Brain-train plugin name(s) applied across datapairs.

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

    if streaming:
        dp_iter, sig_iter = itertools.tee(datapairs)
        sig = _dataset_sig(sig_iter)
        data_iter = dp_iter
    else:
        dataset_list = list(datapairs)
        sig = _dataset_sig(dataset_list)
        data_iter = iter(dataset_list)
    if getattr(brain, "_dataset_signature", None) is None:
        brain._dataset_signature = sig  # type: ignore[attr-defined]
    else:
        if brain._dataset_signature != sig and not getattr(brain, "allow_dissimilar_datasets_in_wanderers", False):
            raise ValueError(
                "Dataset mismatch across wanderers on the same brain; set allow_dissimilar_datasets_in_wanderers=True to override"
            )

    if not getattr(brain, "store_snapshots", False):
        try:
            brain.store_snapshots = True  # type: ignore[attr-defined]
            if not getattr(brain, "snapshot_path", None):
                brain.snapshot_path = os.path.join(tempfile.gettempdir(), "marble_snapshots")  # type: ignore[attr-defined]
            if getattr(brain, "snapshot_freq", None) is None:
                brain.snapshot_freq = 1  # type: ignore[attr-defined]
            os.makedirs(brain.snapshot_path, exist_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

    # Build a single Wanderer instance for all pairs; target set per-pair
    _current_target: Dict[str, Any] = {"val": None}

    def _target_provider(_y: Any) -> Any:
        return _current_target["val"]

    cfg = neuro_config
    wtype = wanderer_type
    if lobe is not None and not getattr(lobe, "inherit_plugins", True):
        wtype = lobe.plugin_types
        cfg = lobe.neuro_config
    w = Wanderer(
        brain,
        type_name=wtype,
        seed=seed,
        loss=loss,
        target_provider=_target_provider,
        neuro_config=cfg,
        gradient_clip=gradient_clip,
        mixedprecision=mixedprecision,
    )
    batch_sz = int(batch_size if batch_size is not None else getattr(w, "_batch_size", 1))
    if batch_sz > 1 and (wanderer_type is None or "batchtrainer" not in str(wanderer_type)):
        raise ValueError("batch_size>1 requires 'batchtrainer' wanderer plugin")
    if selfattention is not None:
        attach_selfattention(w, selfattention)
    # Allow any enabled learning paradigms on the brain to configure the Wanderer
    try:
        apply_paradigms_to_wanderer(brain, w)
    except Exception:
        pass

    # Resolve Brain-train plugins (comma-separated or list) and initialize
    train_plugins: List[Any] = []
    if isinstance(train_type, str):
        names = [s.strip() for s in train_type.split(",") if s.strip()]
        for nm in names:
            p = _BRAIN_TRAIN_TYPES.get(nm)
            if p is not None:
                train_plugins.append(p)
    elif isinstance(train_type, (list, tuple)):
        for nm in train_type:
            p = _BRAIN_TRAIN_TYPES.get(str(nm))
            if p is not None:
                train_plugins.append(p)

    cfg = {
        "num_walks": None,
        "max_steps": steps_per_pair,
        "lr": lr,
        "type_name": train_type,
    }
    for p in train_plugins:
        _on_init_train(p, brain, w, cfg)

    batch: List[DataPair] = []
    batch_index = 0
    def _process_batch(items: List[DataPair], idx: int) -> None:
        nonlocal count
        enc_left_list: List[Any] = []
        enc_right_list: List[Any] = []
        torch = w._torch  # type: ignore[attr-defined]
        for dp in items:
            l_raw, r_raw = dp.left, dp.right
            l_simple = (isinstance(l_raw, (int, float)) or (isinstance(l_raw, (list, tuple)) and len(l_raw) == 1 and isinstance(l_raw[0], (int, float))))
            r_simple = (isinstance(r_raw, (int, float)) or (isinstance(r_raw, (list, tuple)) and len(r_raw) == 1 and isinstance(r_raw[0], (int, float))))
            if l_simple and r_simple:
                l_val = float(l_raw[0] if isinstance(l_raw, (list, tuple)) else l_raw)
                r_val = float(r_raw[0] if isinstance(r_raw, (list, tuple)) else r_raw)
                if torch is not None:
                    enc_left_list.append(torch.tensor([l_val], dtype=torch.float32, device=w._device))
                    enc_right_list.append(torch.tensor([r_val], dtype=torch.float32, device=w._device))
                else:
                    enc_left_list.append([l_val])
                    enc_right_list.append([r_val])
            else:
                el, er = dp.encode(codec)
                enc_left_list.append(el)
                enc_right_list.append(er)
        if torch is not None:
            enc_left = torch.stack([torch.tensor(x, device=w._device, dtype=torch.float32) if not hasattr(x, "shape") else x.float() for x in enc_left_list])
            enc_right = torch.stack([torch.tensor(x, device=w._device, dtype=torch.float32) if not hasattr(x, "shape") else x.float() for x in enc_right_list])
        else:
            enc_left = [list(x if isinstance(x, (list, tuple)) else [x]) for x in enc_left_list]
            enc_right = [list(x if isinstance(x, (list, tuple)) else [x]) for x in enc_right_list]

        start: Optional[Neuron]
        first_left = enc_left_list[0]
        if left_to_start is not None:
            start = left_to_start(first_left, brain)  # type: ignore[arg-type]
        else:
            try:
                start = next(iter(brain.neurons.values())) if getattr(brain, "neurons", None) else None  # type: ignore[attr-defined]
            except Exception:
                start = None
            if start is None:
                try:
                    avail = brain.available_indices()
                    if avail:
                        idxn = avail[0]
                        start = brain.add_neuron(idxn, tensor=0.0)
                except Exception:
                    start = None
        for p in train_plugins:
            s = _select_start(brain, w, idx, p, None)
            if s is not None:
                start = s
        if start is not None:
            start.receive(enc_left)
        else:
            start = create_start_neuron(brain, enc_left)

        overrides: Dict[str, Any] = {}
        for p in train_plugins:
            ovr = _before_walk_overrides(p, brain, w, idx)
            overrides.update(ovr)
        ms = int(overrides.get("max_steps", steps_per_pair))
        lr_i = float(overrides.get("lr", lr))
        if batch_sz > 1:
            lr_i /= batch_sz
        _current_target["val"] = enc_right
        stats = w.walk(max_steps=ms, start=start, lr=lr_i, lobe=lobe)
        stats["plugins"] = [p.__class__.__name__ for p in getattr(w, "_wplugins", []) or []]
        history.append(stats)
        for p in train_plugins:
            _after_walk(p, brain, w, idx, stats)
        count += len(items)
        try:
            report(
                "training",
                f"batch_{idx}",
                {"loss": stats.get("loss", 0.0), "steps": stats.get("steps", 0)},
                "datapair",
            )
        except Exception:
            pass
        if callback is not None and items:
            try:
                callback(idx, stats, items[0])
            except Exception:
                pass
        del items, enc_left, enc_right
        gc.collect()

    for i, item in enumerate(data_iter):
        dp = _normalize_pair(item)
        batch.append(dp)
        if len(batch) >= batch_sz:
            _process_batch(batch, batch_index)
            batch = []
            batch_index += 1
    if batch:
        _process_batch(batch, batch_index)

    final_loss = history[-1]["loss"] if history else 0.0
    out = {"history": history, "final_loss": final_loss, "count": count}
    for p in train_plugins:
        extra = _on_end_train(p, brain, w, history)
        out = _merge_dict_safe(out, extra)
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
    mixedprecision: bool = True,
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
            mixedprecision=mixedprecision,
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
    mixedprecision: bool = True,
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

        lock = threading.Lock()

        def runner(idx: int) -> None:
            seed = None if seeds is None or idx >= len(seeds) else seeds[idx]
            with lock:
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
                    mixedprecision=mixedprecision,
                )
                results[idx] = res

        threads: List[threading.Thread] = []
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
    mixedprecision: bool = True,
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
        mixedprecision=mixedprecision,
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

# Re-import training helpers to ensure lock-based implementations are used
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
