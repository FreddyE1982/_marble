from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .reporter import REPORTER, report
from .graph import _NEURON_TYPES


_SELFA_TYPES: Dict[str, Any] = {}


def register_selfattention_type(name: str, plugin: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("SelfAttention type name must be a non-empty string")
    _SELFA_TYPES[name] = plugin


class _ReporterReadOnlyView:
    def __init__(self, reporter: "Reporter") -> None:
        self._r = reporter

    def item(self, itemname: str, groupname: str, *subgroups: str):
        return self._r.item((itemname,) + (groupname,) + tuple(subgroups))

    def group(self, groupname: str, *subgroups: str) -> Dict[str, Any]:
        return self._r.group(groupname, *subgroups)

    def dirtree(self, groupname: Optional[str] = None, *subgroups: str) -> Dict[str, Any]:
        return self._r.dirtree(groupname, *subgroups)


class SelfAttention:
    """Manage self-attention routines that can adjust Wanderer settings step-wise.

    - Routines are plugins with optional hooks:
        * on_init(selfattention)
        * after_step(selfattention, reporter_ro, wanderer, step_index, ctx) -> dict|None
          Returned dict is treated as parameter updates applied at the next step.
    - Provides get/set access to Wanderer settings via queued updates to apply next step.
    - Routines receive a read-only reporter view; they cannot write to REPORTER.
    """

    def __init__(self, *, type_name: Optional[str] = None, routines: Optional[List[Any]] = None, history_size: int = 1000) -> None:
        self.type_name = type_name
        self._routines: List[Any] = []
        base = _SELFA_TYPES.get(self.type_name) if self.type_name else None
        if base is not None:
            self._routines.append(base)
        if isinstance(routines, list):
            for r in routines:
                self._routines.append(r)
        self._reporter_ro = _ReporterReadOnlyView(REPORTER)
        self._owner = None  # type: ignore[assignment]
        self._history_size = max(1, int(history_size))  # retained for API; history is sourced from REPORTER
        self._last_applied: Optional[Dict[str, Any]] = None
        # Change stack for rollback support (latest change on top)
        # Each entry: {tag: str|None, created_neurons: [Neuron], created_synapses: [Synapse],
        #              removed_synapses: [snapshot], removed_neurons: [snapshot]}
        self._change_stack: List[Dict[str, Any]] = []
        # Per-neuron learnable parameter registry managed by SelfAttention.
        # Structure: { id(neuron): { name: { 'tensor': torch.Tensor, 'opt': bool, 'lr': Optional[float] } } }
        self._learnables: Dict[int, Dict[str, Dict[str, Any]]] = {}
        # Global learnable parameters for SelfAttention routines
        self._global_learnables: Dict[str, Dict[str, Any]] = {}

    # API exposed to routines
    def get_param(self, name: str) -> Any:
        owner = getattr(self, "_owner", None)
        if owner is None:
            return None
        if not isinstance(name, str) or not name or name.startswith("_"):
            return None
        try:
            return getattr(owner, name)
        except Exception:
            return None

    def set_param(self, name: str, value: Any) -> None:
        owner = getattr(self, "_owner", None)
        if owner is None:
            return
        if not isinstance(name, str) or not name or name.startswith("_"):
            return
        try:
            # Queue for application at next step
            pending = getattr(owner, "_pending_settings")  # type: ignore[attr-defined]
            pending[name] = value
        except Exception:
            pass

    # Internal wiring
    def _bind(self, wanderer: "Wanderer") -> None:
        self._owner = wanderer
        for r in self._routines:
            try:
                if hasattr(r, "on_init"):
                    r.on_init(self)  # type: ignore[attr-defined]
            except Exception:
                pass

    # --- Learnable parameter management (per-neuron) ---
    def ensure_learnable_param(self, neuron: "Neuron", name: str, init_value: Any, *, requires_grad: bool = True, lr: Optional[float] = None) -> Any:
        """Register (or fetch existing) per-neuron learnable parameter accessible to plugins.

        - Stores tensor under neuron._plugin_state['learnable_params'][name]
        - Managed in this SelfAttention's registry for optional optimization.
        - Returns the tensor.
        """
        owner = getattr(self, "_owner", None)
        torch = getattr(owner, "_torch", None) if owner is not None else None
        device = getattr(owner, "_device", "cpu") if owner is not None else "cpu"
        nid = id(neuron)
        if nid not in self._learnables:
            self._learnables[nid] = {}
        entry = self._learnables[nid].get(name)
        if entry is not None:
            return entry.get("tensor")
        # Build tensor from init_value
        if torch is not None:
            try:
                t = torch.tensor(init_value, dtype=torch.float32, device=device, requires_grad=requires_grad)
            except Exception:
                # Fallback scalar zero
                t = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=requires_grad)
        else:
            # Without torch, store as plain list/float for plugin access; no autograd
            t = init_value
        self._learnables[nid][name] = {"tensor": t, "opt": False, "lr": lr}
        try:
            # Also expose in neuron's plugin_state so plugins prefer it in forward paths
            lstore = getattr(neuron, "_plugin_state", {}).setdefault("learnable_params", {})
            lstore[name] = t
        except Exception:
            pass
        try:
            report("selfattention", "ensure_learnable", {"neuron": nid, "name": name}, "builder")
        except Exception:
            pass
        return t

    def ensure_global_learnable_param(self, name: str, init_value: Any, *, requires_grad: bool = True,
                                      lr: Optional[float] = None) -> Any:
        owner = getattr(self, "_owner", None)
        torch = getattr(owner, "_torch", None) if owner is not None else None
        device = getattr(owner, "_device", "cpu") if owner is not None else "cpu"
        if name in self._global_learnables:
            return self._global_learnables[name]["tensor"]
        if torch is not None:
            try:
                t = torch.tensor(init_value, dtype=torch.float32, device=device, requires_grad=requires_grad)
            except Exception:
                t = torch.tensor([init_value], dtype=torch.float32, device=device, requires_grad=requires_grad)
        else:
            t = init_value
        self._global_learnables[name] = {"tensor": t, "opt": False, "lr": lr}
        return t

    def set_param_optimization(self, neuron: "Neuron", name: str, *, enabled: bool = True, lr: Optional[float] = None) -> None:
        nid = id(neuron)
        if nid not in self._learnables or name not in self._learnables[nid]:
            # Not registered yet; nothing to toggle
            return
        self._learnables[nid][name]["opt"] = bool(enabled)
        if lr is not None:
            self._learnables[nid][name]["lr"] = float(lr)

    def set_global_param_optimization(self, name: str, *, enabled: bool = True, lr: Optional[float] = None) -> None:
        ent = self._global_learnables.get(name)
        if ent is None:
            return
        ent["opt"] = bool(enabled)
        if lr is not None:
            ent["lr"] = float(lr)

    def get_neuron_param_tensor(self, neuron: "Neuron", name: str) -> Any:
        nid = id(neuron)
        ent = self._learnables.get(nid, {}).get(name)
        return None if ent is None else ent.get("tensor")

    def get_global_param_tensor(self, name: str) -> Any:
        ent = self._global_learnables.get(name)
        return None if ent is None else ent.get("tensor")

    def list_neuron_learnables(self, neuron: "Neuron") -> List[str]:
        return list(self._learnables.get(id(neuron), {}).keys())

    def _collect_enabled_params(self, wanderer: "Wanderer") -> List[Tuple[Any, float]]:
        out: List[Tuple[Any, float]] = []
        torch = getattr(wanderer, "_torch", None)
        default_lr = float(getattr(wanderer, "current_lr", 0.0) or 0.0) or 1e-2
        for nid, params in self._learnables.items():
            for name, cfg in params.items():
                if not cfg.get("opt"):
                    continue
                t = cfg.get("tensor")
                if torch is not None and hasattr(t, "requires_grad"):
                    out.append((t, float(cfg.get("lr", default_lr))))
        for name, cfg in self._global_learnables.items():
            if not cfg.get("opt"):
                continue
            t = cfg.get("tensor")
            if torch is not None and hasattr(t, "requires_grad"):
                out.append((t, float(cfg.get("lr", default_lr))))
        return out

    def _update_learnables(self, wanderer: "Wanderer") -> None:
        """Apply SGD updates to enabled learnable params after loss.backward()."""
        torch = getattr(wanderer, "_torch", None)
        if torch is None:
            return
        pairs = self._collect_enabled_params(wanderer)
        for t, lr in pairs:
            try:
                if hasattr(t, "grad") and t.grad is not None:
                    with torch.no_grad():
                        t -= lr * t.grad
                    # Clear grad for next step
                    if hasattr(t, "grad"):
                        t.grad = None
            except Exception:
                pass

    def _after_step(self, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]) -> None:
        # Invoke routines with read-only reporter and context; they may return updates
        for r in self._routines:
            try:
                if hasattr(r, "after_step"):
                    upd = r.after_step(self, self._reporter_ro, wanderer, step_index, dict(ctx))  # type: ignore[attr-defined]
                    if isinstance(upd, dict):
                        for k, v in upd.items():
                            self.set_param(str(k), v)
            except Exception:
                pass

    def _notify_applied(self, applied: Dict[str, Any]) -> None:
        # Called by Wanderer when queued settings are applied at the start of a step
        try:
            self._last_applied = dict(applied)
        except Exception:
            self._last_applied = None

    # --- Change tracking & rollback API for routines ---
    def start_change(self, tag: Optional[str] = None) -> int:
        rec = {
            "tag": tag,
            "created_neurons": [],
            "created_synapses": [],
            "removed_synapses": [],
            "removed_neurons": [],
        }
        self._change_stack.append(rec)
        try:
            report("selfattention", "start_change", {"tag": tag, "depth": len(self._change_stack)}, "builder")
        except Exception:
            pass
        return len(self._change_stack) - 1

    def record_created_neuron(self, neuron: "Neuron") -> None:
        if not self._change_stack:
            self.start_change("implicit")
        try:
            self._change_stack[-1]["created_neurons"].append(neuron)
        except Exception:
            pass

    def record_created_synapse(self, synapse: "Synapse") -> None:
        if not self._change_stack:
            self.start_change("implicit")
        try:
            self._change_stack[-1]["created_synapses"].append(synapse)
        except Exception:
            pass

    def record_removed_synapse(self, synapse: "Synapse") -> None:
        if not self._change_stack:
            self.start_change("implicit")
        snap = {}
        try:
            snap = {
                "source": getattr(getattr(synapse, "source", None), "position", None),
                "target": getattr(getattr(synapse, "target", None), "position", None),
                "direction": getattr(synapse, "direction", "uni"),
                "age": int(getattr(synapse, "age", 0)),
                "type_name": getattr(synapse, "type_name", None),
                "weight": float(getattr(synapse, "weight", 1.0)),
            }
        except Exception:
            pass
        try:
            self._change_stack[-1]["removed_synapses"].append(snap)
        except Exception:
            pass

    def record_removed_neuron(self, neuron: "Neuron") -> None:
        if not self._change_stack:
            self.start_change("implicit")
        snap = {}
        try:
            # Capture core attributes and a CPU list version of the tensor
            t = getattr(neuron, "tensor", None)
            if hasattr(t, "detach") and hasattr(t, "to"):
                t_list = t.detach().to("cpu").view(-1).tolist()
            elif isinstance(t, list):
                t_list = list(t)
            else:
                try:
                    t_list = list(t)
                except Exception:
                    t_list = []
            pos = getattr(neuron, "position", None)
            snap = {
                "position": pos,
                "tensor": t_list,
                "weight": float(getattr(neuron, "weight", 1.0)),
                "bias": float(getattr(neuron, "bias", 0.0)),
                "age": int(getattr(neuron, "age", 0)),
                "type_name": getattr(neuron, "type_name", None),
                # Incident synapses snapshots (so we can rebuild topology)
                "incoming": [],
                "outgoing": [],
            }
            for s in getattr(neuron, "incoming", []) or []:
                try:
                    snap["incoming"].append({
                        "source": getattr(getattr(s, "source", None), "position", None),
                        "target": getattr(getattr(s, "target", None), "position", None),
                        "direction": getattr(s, "direction", "uni"),
                        "age": int(getattr(s, "age", 0)),
                        "type_name": getattr(s, "type_name", None),
                        "weight": float(getattr(s, "weight", 1.0)),
                    })
                except Exception:
                    pass
            for s in getattr(neuron, "outgoing", []) or []:
                try:
                    snap["outgoing"].append({
                        "source": getattr(getattr(s, "source", None), "position", None),
                        "target": getattr(getattr(s, "target", None), "position", None),
                        "direction": getattr(s, "direction", "uni"),
                        "age": int(getattr(s, "age", 0)),
                        "type_name": getattr(s, "type_name", None),
                        "weight": float(getattr(s, "weight", 1.0)),
                    })
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._change_stack[-1]["removed_neurons"].append(snap)
        except Exception:
            pass

    def commit_change(self) -> None:
        # Keep the record for potential rollback (latest change)
        try:
            report("selfattention", "commit_change", {"depth": len(self._change_stack)}, "builder")
        except Exception:
            pass

    def rollback_last_change(self) -> bool:
        if not self._change_stack:
            return False
        rec = self._change_stack.pop()
        w = getattr(self, "_owner", None)
        brain = getattr(w, "brain", None) if w is not None else None
        if brain is None:
            return False
        ok = True
        # Remove created synapses
        for s in list(rec.get("created_synapses", [])):
            try:
                brain.remove_synapse(s)
            except Exception:
                ok = False
        # Remove created neurons
        for n in list(rec.get("created_neurons", [])):
            try:
                brain.remove_neuron(n)
            except Exception:
                ok = False
        # Restore removed neurons (then their incident synapses)
        for ns in list(rec.get("removed_neurons", [])):
            try:
                pos = ns.get("position")
                if pos is None:
                    continue
                nn = brain.add_neuron(pos, tensor=ns.get("tensor", []),
                                      weight=ns.get("weight", 1.0), bias=ns.get("bias", 0.0),
                                      age=ns.get("age", 0), type_name=ns.get("type_name", None))
                # Recreate incoming/outgoing synapses
                for sd in ns.get("incoming", []):
                    try:
                        brain.connect(sd.get("source"), sd.get("target"), direction=sd.get("direction", "uni"),
                                      age=sd.get("age", 0), type_name=sd.get("type_name", None), weight=sd.get("weight", 1.0))
                    except Exception:
                        ok = False
                for sd in ns.get("outgoing", []):
                    try:
                        brain.connect(sd.get("source"), sd.get("target"), direction=sd.get("direction", "uni"),
                                      age=sd.get("age", 0), type_name=sd.get("type_name", None), weight=sd.get("weight", 1.0))
                    except Exception:
                        ok = False
            except Exception:
                ok = False
        # Restore removed synapses
        for ss in list(rec.get("removed_synapses", [])):
            try:
                brain.connect(ss.get("source"), ss.get("target"), direction=ss.get("direction", "uni"),
                              age=ss.get("age", 0), type_name=ss.get("type_name", None), weight=ss.get("weight", 1.0))
            except Exception:
                ok = False
        try:
            report("selfattention", "rollback", {"tag": rec.get("tag"), "ok": ok}, "builder")
        except Exception:
            pass
        return ok

    # Public analysis helpers for routines/users, sourced from REPORTER
    def history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            items = self._reporter_ro.group("wanderer_steps", "logs")
        except Exception:
            items = {}
        # Sort keys like step_123 numerically
        def keyfn(k: str) -> int:
            try:
                return int(str(k).split("_")[-1])
            except Exception:
                return 1 << 30
        keys = sorted(items.keys(), key=keyfn)
        recs = [items[k] for k in keys]
        if last_n is None:
            return recs
        try:
            return recs[-int(last_n):]
        except Exception:
            return recs

    # --- Neuron type discovery and wiring validation ---
    def list_neuron_types(self) -> List[str]:
        """Return available neuron types known to the system (including base)."""
        types = ["base"]
        try:
            types += list(_NEURON_TYPES.keys())
        except Exception:
            pass
        # Deduplicate and sort
        uniq = sorted(set(t for t in types if isinstance(t, str) and t))
        return uniq

    def validate_conv1d(self, neuron: "Neuron") -> Dict[str, Any]:
        """Validate conv1d wiring: exactly 5 incoming, exactly 1 outgoing.

        The self-attention routines are responsible for wiring; this method only
        checks that the required number of inputs/outputs are connected.
        """
        ok = False
        reason = None
        try:
            if getattr(neuron, "type_name", None) != "conv1d":
                ok, reason = False, "not a conv1d neuron"
            else:
                inc = getattr(neuron, "incoming", []) or []
                out = getattr(neuron, "outgoing", []) or []
                if len(inc) != 5:
                    ok, reason = False, f"conv1d requires 5 incoming, found {len(inc)}"
                elif len(out) != 1:
                    ok, reason = False, f"conv1d requires 1 outgoing, found {len(out)}"
                else:
                    ok, reason = True, None
        except Exception as e:
            ok, reason = False, f"validation error: {e}"
        # Log validation result
        try:
            report("selfattention", "validate_conv1d", {"ok": ok, "reason": reason}, "builder")
        except Exception:
            pass
        return {"ok": ok, "reason": reason}

    def validate_neuron_wiring(self, neuron: "Neuron") -> Dict[str, Any]:
        """Validate wiring for known special neuron types; base type always ok.

        Returns {ok: bool, reason: Optional[str]}.
        """
        t = getattr(neuron, "type_name", None)
        if t in (None, "", "base"):
            return {"ok": True, "reason": None}
        if t == "conv1d":
            return self.validate_conv1d(neuron)
        if t in ("conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"):
            inc = getattr(neuron, "incoming", []) or []
            out = getattr(neuron, "outgoing", []) or []
            def is_param(s):
                tt = getattr(s, "type_name", None)
                return isinstance(tt, str) and tt.startswith("param")
            param_cnt = len([s for s in inc if is_param(s)])
            if param_cnt != 5:
                return {"ok": False, "reason": f"{t} requires 5 incoming PARAM synapses, found {param_cnt}"}
            if len(out) != 1:
                return {"ok": False, "reason": f"{t} requires 1 outgoing, found {len(out)}"}
            return {"ok": True, "reason": None}
        if t in ("maxpool1d", "maxpool2d", "maxpool3d"):
            inc = getattr(neuron, "incoming", []) or []
            out = getattr(neuron, "outgoing", []) or []
            def is_param2(s):
                tt = getattr(s, "type_name", None)
                return isinstance(tt, str) and tt.startswith("param")
            param_cnt = len([s for s in inc if is_param2(s)])
            if param_cnt != 3:
                return {"ok": False, "reason": f"{t} requires 3 incoming PARAM synapses, found {param_cnt}"}
            if len(out) != 1:
                return {"ok": False, "reason": f"{t} requires 1 outgoing, found {len(out)}"}
            return {"ok": True, "reason": None}
        # Unknown special type: consider ok but inform
        try:
            report("selfattention", "validate_unknown", {"type": t}, "builder")
        except Exception:
            pass
        return {"ok": True, "reason": None}


def attach_selfattention(wanderer: "Wanderer", selfattention: "SelfAttention") -> "Wanderer":
    """Attach a SelfAttention instance to the Wanderer (supports stacking)."""
    try:
        if not hasattr(wanderer, "_selfattentions") or getattr(wanderer, "_selfattentions") is None:
            setattr(wanderer, "_selfattentions", [])
        lst = getattr(wanderer, "_selfattentions")
        lst.append(selfattention)
        selfattention._bind(wanderer)
    except Exception:
        pass
    return wanderer


__all__ = [
    "SelfAttention",
    "register_selfattention_type",
    "attach_selfattention",
]

