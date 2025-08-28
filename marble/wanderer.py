from __future__ import annotations

# Thin aggregation layer for Wanderer-related APIs.
# Implementation currently resides in marble.marblemain; this module provides
# a stable boundary to continue refactoring without breaking public imports.

from typing import Any, Dict, Optional, List, Tuple, Sequence, Union, Callable
import time
import random
import contextlib

from .graph import _DeviceHelper, Neuron, Synapse
from .reporter import report

# Core registries for Wanderer and Neuroplasticity plugins
WANDERER_TYPES_REGISTRY: Dict[str, Any] = {}
NEURO_TYPES_REGISTRY: Dict[str, Any] = {}


def register_wanderer_type(name: str, plugin: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Wanderer type name must be a non-empty string")
    WANDERER_TYPES_REGISTRY[name] = plugin


def register_neuroplasticity_type(name: str, plugin: Any) -> None:
    NEURO_TYPES_REGISTRY[str(name)] = plugin


# Local aliases for registry use inside the class (keeps original logic intact)
_WANDERER_TYPES = WANDERER_TYPES_REGISTRY
_NEURO_TYPES = NEURO_TYPES_REGISTRY


def _tqdm_factory():
    try:
        from IPython import get_ipython  # type: ignore
        if get_ipython() is not None:  # pragma: no cover - runtime check
            from tqdm.notebook import tqdm  # type: ignore
        else:  # pragma: no cover
            from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        from tqdm import tqdm  # type: ignore
    return tqdm


class Wanderer(_DeviceHelper):
    """Autograd-driven wanderer that traverses the Brain via neurons/synapses.

    Behavior:
    - Starts from a specific neuron or a random neuron within the brain.
    - At each step, chooses a connected synapse and follows its allowed direction.
      For bidirectional synapses, direction is chosen randomly.
    - Uses torch autograd to build a computation graph over the path and performs a
      backward pass to update visited neurons' weights/biases via simple SGD.

    Plugins:
    - A wanderer plugin can override path selection and/or loss computation by
      implementing optional hooks on a provided object:
        * on_init(wanderer)
        * choose_next(wanderer, current_neuron, choices) -> (synapse, direction)
          where choices is a list of (synapse, direction_str) with direction_str in
          {"forward","backward"}.
        * loss(wanderer, outputs) -> torch scalar tensor
    """

    def __init__(
        self,
        brain: "Brain",
        *,
        type_name: Optional[str] = None,
        seed: Optional[int] = None,
        loss: Optional[Union[str, Callable[..., Any], Any]] = None,
        target_provider: Optional[Callable[[Any], Any]] = None,
        neuroplasticity_type: Optional[Union[str, Sequence[str]]] = "base",
        neuro_config: Optional[Dict[str, Any]] = None,
        gradient_clip: Optional[Dict[str, Any]] = None,
        mixedprecision: bool = True,
    ) -> None:
        super().__init__()
        # Mandatory autograd requirement
        if self._torch is None:
            raise RuntimeError(
                "torch is required for Wanderer autograd. Please install CPU torch (or GPU if available) and retry."
            )
        self.brain = brain
        if mixedprecision:
            if type_name:
                names = [s.strip() for s in str(type_name).split(",") if s.strip()]
                if "mixedprecision" not in names:
                    type_name = ",".join(names + ["mixedprecision"])
            else:
                type_name = "mixedprecision"
        self.type_name = type_name
        self.rng = random.Random(seed)
        self._plugin_state: Dict[str, Any] = {}
        self._plugin_state["learnable_params"] = {}
        self.learnable_params: Dict[str, Any] = self._plugin_state["learnable_params"]
        for i in range(10):
            self.learnable_params[f"param_{i}"] = self._torch.tensor(
                0.0,
                dtype=self._torch.float32,
                device=self._device,
                requires_grad=True,
            )
        self._visited: List[Neuron] = []
        self._param_map: Dict[int, Tuple[Any, Any]] = {}  # id(neuron) -> (w_param, b_param)
        self._loss_spec = loss
        self._loss_module = None  # torch.nn.* instance if applicable
        self._target_provider = target_provider
        self._neuro_type = neuroplasticity_type
        self._neuro_cfg: Dict[str, Any] = dict(neuro_config or {})
        # Gradient clipping configuration (additive; optional)
        # Keys:
        #   method: "norm" | "value"
        #   max_norm: float (for norm)
        #   norm_type: float (for norm; default 2.0)
        #   clip_value: float (for value)
        self._grad_clip: Dict[str, Any] = dict(gradient_clip or {})

        try:
            report("wanderer", "init", {"plugin": self.type_name}, "events")
        except Exception:
            pass

        # Walk-control state
        self._last_walk_loss: Optional[float] = None
        self._finish_requested: bool = False
        self._finish_stats: Optional[Dict[str, Any]] = None
        self._walk_ctx: Dict[str, Any] = {}
        self._last_step_time: Optional[float] = None
        self._walk_loss_sum: float = 0.0
        self._walk_step_count: int = 0
        self._global_step_counter: int = 0
        self._last_walk_mean_loss: Optional[float] = None
        self._walk_counter: int = 0
        self._pending_settings: Dict[str, Any] = {}
        self._selfattentions: List[Any] = []
        self.lr_override: Optional[float] = None
        self.current_lr: Optional[float] = None
        self._last_applied_settings: Optional[Dict[str, Any]] = None

        # Resolve stacked wanderer plugins (comma-separated str or list)
        self._wplugins: List[Any] = []
        if isinstance(self.type_name, str):
            names = [s.strip() for s in self.type_name.split(",") if s.strip()]
            for nm in names:
                plug = _WANDERER_TYPES.get(nm)
                if plug is not None:
                    self._wplugins.append(plug)
        elif isinstance(self.type_name, (list, tuple)):
            for nm in self.type_name:
                plug = _WANDERER_TYPES.get(str(nm))
                if plug is not None:
                    self._wplugins.append(plug)
        # on_init for all wanderer plugins
        for plug in self._wplugins:
            try:
                if hasattr(plug, "on_init"):
                    plug.on_init(self)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Resolve stacked neuroplasticity plugins
        self._neuro_type = neuroplasticity_type
        self._neuro_plugins: List[Any] = []
        if isinstance(neuroplasticity_type, str):
            nnames = [s.strip() for s in neuroplasticity_type.split(",") if s.strip()]
        else:
            nnames = [str(x) for x in (neuroplasticity_type or [])]
        for nm in nnames:
            nplug = _NEURO_TYPES.get(nm)
            if nplug is not None:
                self._neuro_plugins.append(nplug)
        for nplug in self._neuro_plugins:
            try:
                if hasattr(nplug, "on_init"):
                    nplug.on_init(self)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Learning paradigms may configure Wanderer and receive init hook
        for paradigm in (self.brain.active_paradigms() if hasattr(self.brain, "active_paradigms") else (getattr(self.brain, "_paradigms", []) or [])):
            try:
                if hasattr(paradigm, "on_wanderer"):
                    paradigm.on_wanderer(self)
            except Exception:
                pass
            try:
                if hasattr(paradigm, "on_init"):
                    paradigm.on_init(self)
            except Exception:
                pass

    def walk(
        self,
        *,
        max_steps: int = 10,
        start: Optional[Neuron] = None,
        lr: float = 1e-2,
        loss_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> Dict[str, Any]:
        torch = self._torch  # type: ignore[assignment]

        plug = None
        for p in getattr(self, "_wplugins", []) or []:
            if hasattr(p, "walk"):
                plug = p
                break
        if plug is not None:
            try:
                res = plug.walk(self, max_steps=max_steps, start=start, lr=lr, loss_fn=loss_fn)  # type: ignore[attr-defined]
                try:
                    if isinstance(res, dict) and "loss" in res and "steps" in res and "visited" in res:
                        self._walk_counter = getattr(self, "_walk_counter", 0) + 1
                        report(
                            "training",
                            f"walk_{self._walk_counter}",
                            {
                                "final_loss": float(res.get("loss", 0.0)),
                                "mean_step_loss": res.get("mean_step_loss"),
                                "steps": int(res.get("steps", 0)),
                                "visited": int(res.get("visited", 0)),
                                "timestamp": time.time(),
                            },
                            "walks",
                        )
                        if getattr(self.brain, "store_snapshots", False):
                            freq = getattr(self.brain, "snapshot_freq", None)
                            if freq and self._walk_counter % int(freq) == 0:
                                try:
                                    self.brain.save_snapshot()
                                except Exception:
                                    pass
                except Exception:
                    pass
                return res if isinstance(res, dict) else {"loss": 0.0, "steps": 0, "visited": 0}
            except Exception:
                pass

        current = start if start is not None else self._random_start()
        if current is None:
            return {"loss": 0.0, "steps": 0, "visited": 0}

        outputs: List[Any] = []
        self._visited = []
        self._param_map = {}
        step_metrics: List[Dict[str, float]] = []
        self._finish_requested = False
        self._finish_stats = None
        self._walk_ctx = {}
        self._last_step_time = time.time()
        self._walk_loss_sum = 0.0
        self._walk_step_count = 0
        amp_enabled = bool(getattr(self, '_use_mixed_precision', False))
        amp_device = 'cuda' if torch.cuda.is_available() and str(self._device).startswith('cuda') else 'cpu'
        amp_dtype = torch.float16 if amp_device == 'cuda' else torch.bfloat16
        scaler = getattr(self, '_amp_scaler', None) if amp_enabled else None
        if amp_enabled and scaler is None:
            try:
                scaler = torch.cuda.amp.GradScaler()
            except Exception:
                scaler = None
            self._amp_scaler = scaler

        tqdm_cls = _tqdm_factory()
        pbar = tqdm_cls(total=max_steps, leave=False)
        total_dt = 0.0
        prev_mean = None

        try:
            for n in self.brain.neurons.values():  # type: ignore[union-attr]
                lock_ctx = None
                try:
                    if hasattr(self.brain, "lock_neuron"):
                        lock_ctx = self.brain.lock_neuron(n, timeout=0.5)
                except Exception:
                    lock_ctx = None
                if lock_ctx is not None:
                    with lock_ctx:
                        t = getattr(n, "tensor", None)
                        if hasattr(t, "detach") and hasattr(t, "to"):
                            n.tensor = t.detach().to(self._device)
                else:
                    t = getattr(n, "tensor", None)
                    if hasattr(t, "detach") and hasattr(t, "to"):
                        n.tensor = t.detach().to(self._device)
        except Exception:
            pass

        def params_for(n: Neuron) -> Tuple[Any, Any]:
            key = id(n)
            if key in self._param_map:
                return self._param_map[key]
            try:
                wt_val = (
                    float(n.weight.detach().to("cpu").item())
                    if hasattr(getattr(n, "weight", None), "detach")
                    else float(getattr(n, "weight", 0.0))
                )
            except Exception:
                wt_val = float(getattr(n, "weight", 0.0))
            w = torch.tensor(wt_val, dtype=torch.float32, device=self._device, requires_grad=True)
            try:
                bs_val = (
                    float(n.bias.detach().to("cpu").item())
                    if hasattr(getattr(n, "bias", None), "detach")
                    else float(getattr(n, "bias", 0.0))
                )
            except Exception:
                bs_val = float(getattr(n, "bias", 0.0))
            b = torch.tensor(bs_val, dtype=torch.float32, device=self._device, requires_grad=True)
            self._param_map[key] = (w, b)
            return w, b

        try:
            for p in getattr(self, "_wplugins", []) or []:
                if hasattr(p, "before_walk"):
                    p.before_walk(self, current)
        except Exception:
            pass
        try:
            for nplug in getattr(self, "_neuro_plugins", []) or []:
                if hasattr(nplug, "before_walk"):
                    nplug.before_walk(self, current)
        except Exception:
            pass
        try:
            for paradigm in (self.brain.active_paradigms() if hasattr(self.brain, "active_paradigms") else (getattr(self.brain, "_paradigms", []) or [])):
                if hasattr(paradigm, "before_walk"):
                    paradigm.before_walk(self, current)
        except Exception:
            pass

        steps = 0
        carried_value: Optional[Any] = None
        moved_last = False
        while steps < max_steps and current is not None:
            try:
                if self._pending_settings:
                    applied_now = dict(self._pending_settings)
                    for _k, _v in list(self._pending_settings.items()):
                        try:
                            setattr(self, _k, _v)
                        except Exception:
                            pass
                    self._pending_settings.clear()
                    self._last_applied_settings = applied_now
                    try:
                        for sa in getattr(self, "_selfattentions", []) or []:
                            if hasattr(sa, "_notify_applied"):
                                sa._notify_applied(applied_now)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass
            self._visited.append(current)
            w_param, b_param = params_for(current)

            original_w = current.weight
            original_b = current.bias
            current.weight = w_param  # type: ignore[assignment]
            current.bias = b_param  # type: ignore[assignment]
            try:
                with (torch.autocast(device_type=amp_device, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()):
                    out = current.forward(carried_value)
            finally:
                pass

            outputs.append(out)

            with (torch.autocast(device_type=amp_device, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()):
                step_loss_t = self._compute_loss([out], override_loss=loss_fn)
            cur_loss = float(step_loss_t.detach().to("cpu").item())
            prev_loss = step_metrics[-1]["loss"] if step_metrics else None
            delta = cur_loss - prev_loss if prev_loss is not None else 0.0

            now = time.time()
            dt = None
            try:
                dt = float(now - (self._last_step_time if self._last_step_time is not None else now))
            except Exception:
                dt = None
            self._last_step_time = now
            step_metrics.append({"loss": cur_loss, "delta": delta, "dt": dt, "mean_loss": self._walk_loss_sum / max(1, self._walk_step_count + 1)})
            self._walk_loss_sum += cur_loss
            self._walk_step_count += 1
            mean_loss = self._walk_loss_sum / max(1, self._walk_step_count)
            total_dt += dt if dt else 0.0

            try:
                cum_loss_t = self._compute_loss(outputs, override_loss=loss_fn)
                cur_walk_loss = float(cum_loss_t.detach().to("cpu").item())
            except Exception:
                cur_walk_loss = float("nan")

            mean_speed = self._walk_step_count / total_dt if total_dt > 0 else 0.0
            loss_speed = -(delta / dt) if dt and delta is not None and dt != 0 else 0.0
            mean_delta = mean_loss - (prev_mean if prev_mean is not None else mean_loss)
            mean_loss_speed = -(mean_delta / dt) if dt and dt != 0 else 0.0
            cur_size, cap = (0, None)
            try:
                cur_size, cap = self.brain.size_stats()  # type: ignore[attr-defined]
            except Exception:
                cur_size = len(getattr(self.brain, "neurons", {}))
                cap = None
            desc = f"{getattr(self.brain, '_progress_epoch', 0)+1}/{getattr(self.brain, '_progress_total_epochs', 1)} epochs "
            desc += f"{getattr(self.brain, '_progress_walk', 0)+1}/{getattr(self.brain, '_progress_total_walks', 1)} walks"
            pbar.set_description(desc)
            try:
                pbar.set_postfix(
                    neurons=cur_size,
                    synapses=len(getattr(self.brain, "synapses", [])),
                    brain=f"{cur_size}/{cap if cap is not None else '-'}",
                    speed=f"{mean_speed:.2f}",
                    paths=len(getattr(self.brain, "synapses", [])),
                    loss=f"{cur_loss:.4f}",
                    mean_loss=f"{mean_loss:.4f}",
                    loss_speed=f"{loss_speed:.4f}",
                    mean_loss_speed=f"{mean_loss_speed:.4f}",
                )
            except Exception:
                pass
            pbar.update(1)
            prev_mean = mean_loss

            self._global_step_counter += 1
            itemname = f"step_{self._global_step_counter}"
            try:
                report(
                    "wanderer_steps",
                    itemname,
                    {
                        "time": now,
                        "dt_since_last": dt,
                        "current_loss": cur_loss,
                        "current_walk_loss": cur_walk_loss,
                        "last_walk_loss": (float(self._last_walk_loss) if self._last_walk_loss is not None else None),
                        "last_walk_mean_loss": (float(self._last_walk_mean_loss) if self._last_walk_mean_loss is not None else None),
                        "applied_settings": self._last_applied_settings,
                        "current_lr": (float(self.current_lr) if isinstance(self.current_lr, (int, float)) or self.current_lr is not None else None),
                        "previous_loss": (float(prev_loss) if prev_loss is not None else None),
                        "mean_loss": float(mean_loss),
                        "neuron_count": int(len(getattr(self.brain, "neurons", {}))),
                        "synapse_count": int(len(getattr(self.brain, "synapses", []))),
                        "walk_step_index": int(self._walk_step_count - 1),
                    },
                    "logs",
                )
            except Exception:
                pass

            self._walk_ctx = {
                "current": current,
                "outputs": outputs,
                "steps": steps,
                "cur_loss_tensor": step_loss_t,
            }

            choices = self._gather_choices(current)
            if not choices:
                try:
                    for nplug in getattr(self, "_neuro_plugins", []) or []:
                        if hasattr(nplug, "on_step"):
                            nplug.on_step(self, current, None, "none", steps, out)  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    for paradigm in (self.brain.active_paradigms() if hasattr(self.brain, "active_paradigms") else (getattr(self.brain, "_paradigms", []) or [])):
                        if hasattr(paradigm, "on_step"):
                            paradigm.on_step(self, current, None, "none", steps, out)  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    for p in getattr(self, "_wplugins", []) or []:
                        if hasattr(p, "on_step"):
                            p.on_step(self, current, None, "none", steps, out)  # type: ignore[attr-defined]
                except Exception:
                    pass
                choices = self._gather_choices(current)
                if not choices:
                    break

            next_syn, dir_str = self._random_choice(choices)
            for plugin in getattr(self, "_wplugins", []) or []:
                try:
                    if hasattr(plugin, "choose_next"):
                        cand_syn, cand_dir = plugin.choose_next(self, current, choices)  # type: ignore[attr-defined]
                        if cand_syn is not None and cand_dir in ("forward", "backward"):
                            next_syn, dir_str = cand_syn, cand_dir
                except Exception:
                    pass

            lock_ctx = None
            try:
                if hasattr(self.brain, "lock_synapse"):
                    lock_ctx = self.brain.lock_synapse(next_syn, timeout=1.0)
            except Exception:
                lock_ctx = None
            if lock_ctx is not None:
                with lock_ctx:
                    if dir_str == "forward":
                        next_neuron = next_syn.transmit(out, direction="forward")
                    else:
                        next_neuron = next_syn.transmit(out, direction="backward")
            else:
                if dir_str == "forward":
                    next_neuron = next_syn.transmit(out, direction="forward")
                else:
                    next_neuron = next_syn.transmit(out, direction="backward")

            for nplug in getattr(self, "_neuro_plugins", []) or []:
                try:
                    if hasattr(nplug, "on_step"):
                        nplug.on_step(self, current, next_syn, dir_str, steps, out)  # type: ignore[attr-defined]
                except Exception:
                    pass

            for paradigm in (self.brain.active_paradigms() if hasattr(self.brain, "active_paradigms") else (getattr(self.brain, "_paradigms", []) or [])):
                try:
                    if hasattr(paradigm, "on_step"):
                        paradigm.on_step(self, current, next_syn, dir_str, steps, out)  # type: ignore[attr-defined]
                except Exception:
                    pass

            try:
                for sa in getattr(self, "_selfattentions", []) or []:
                    if hasattr(sa, "_after_step"):
                        sa._after_step(self, steps, dict(self._walk_ctx))  # type: ignore[attr-defined]
            except Exception:
                pass

            if self._finish_requested:
                break

            try:
                report("wanderer", "step", {"dir": dir_str, "choices": len(choices)}, "events")
            except Exception:
                pass

            current = next_neuron
            carried_value = out
            steps += 1
            moved_last = True
        pbar.close()

        try:
            if moved_last and current is not None:
                out_tail = current.forward(None)
                outputs.append(out_tail)
        except Exception:
            pass
        with (torch.autocast(device_type=amp_device, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()):
            if self._finish_stats is not None and "loss_tensor" in self._finish_stats:
                loss = self._finish_stats["loss_tensor"]
            else:
                loss = self._compute_loss(outputs, override_loss=loss_fn)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scale = scaler.get_scale()
            params = []
            for n in self._visited:
                w_param, b_param = self._param_map[id(n)]
                params.append(w_param); params.append(b_param)
            params.extend(self.learnable_params.values())
            for p in params:
                if p.grad is not None:
                    p.grad.data = p.grad.data / scale
            scaler.update()
        else:
            loss.backward()

        try:
            gc = getattr(self, "_grad_clip", {}) or {}
            method = str(gc.get("method", "")).lower()
            params = []
            for n in self._visited:
                w_param, b_param = self._param_map[id(n)]
                if hasattr(w_param, "grad") or hasattr(b_param, "grad"):
                    params.append(w_param)
                    params.append(b_param)
            params.extend(self.learnable_params.values())
            if params and method:
                torch = self._torch  # type: ignore[assignment]
                if method == "norm":
                    max_norm = float(gc.get("max_norm", 1.0))
                    norm_type = float(gc.get("norm_type", 2.0))
                    try:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                elif method == "value":
                    clip_value = float(gc.get("clip_value", 1.0))
                    try:
                        torch.nn.utils.clip_grad_value_(params, clip_value=clip_value)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            lr_eff = float(self.lr_override) if getattr(self, "lr_override", None) is not None else float(lr)
        except Exception:
            lr_eff = lr  # type: ignore[assignment]
        try:
            self.current_lr = float(lr_eff)
        except Exception:
            self.current_lr = lr_eff  # type: ignore[assignment]

        for n in self._visited:
            w_param, b_param = self._param_map[id(n)]
            with torch.no_grad():
                w_new = w_param - lr_eff * (w_param.grad if w_param.grad is not None else 0.0)
                b_new = b_param - lr_eff * (b_param.grad if b_param.grad is not None else 0.0)
            try:
                n.weight = float(w_new.item())  # type: ignore[assignment]
            except Exception:
                n.weight = float(w_new)  # type: ignore[assignment]
            try:
                n.bias = float(b_new.item())  # type: ignore[assignment]
            except Exception:
                n.bias = float(b_new)  # type: ignore[assignment]

        with torch.no_grad():
            for p in self.learnable_params.values():
                grad = p.grad if getattr(p, "grad", None) is not None else 0.0
                p.sub_(lr_eff * grad)
                if hasattr(p, "grad"):
                    p.grad = None

        try:
            for sa in getattr(self, "_selfattentions", []) or []:
                if hasattr(sa, "_update_learnables"):
                    sa._update_learnables(self)  # type: ignore[attr-defined]
        except Exception:
            pass

        final_loss_val = float(loss.detach().to("cpu").item())
        self._last_walk_loss = final_loss_val
        try:
            if step_metrics:
                self._last_walk_mean_loss = float(sum(m["loss"] for m in step_metrics) / max(1, len(step_metrics)))
            else:
                self._last_walk_mean_loss = 0.0
        except Exception:
            self._last_walk_mean_loss = None

        try:
            self._walk_counter += 1
            report(
                "training",
                f"walk_{self._walk_counter}",
                {
                    "final_loss": final_loss_val,
                    "mean_step_loss": (self._last_walk_mean_loss if self._last_walk_mean_loss is not None else None),
                    "steps": int(steps),
                    "visited": int(len(self._visited)),
                    "timestamp": time.time(),
                },
                "walks",
            )
            if getattr(self.brain, "store_snapshots", False):
                freq = getattr(self.brain, "snapshot_freq", None)
                if freq and self._walk_counter % int(freq) == 0:
                    try:
                        self.brain.save_snapshot()
                    except Exception:
                        pass
        except Exception:
            pass

        for nplug in getattr(self, "_neuro_plugins", []) or []:
            try:
                if hasattr(nplug, "on_walk_end"):
                    nplug.on_walk_end(self, res)  # type: ignore[attr-defined]
            except Exception:
                pass
        for paradigm in (self.brain.active_paradigms() if hasattr(self.brain, "active_paradigms") else (getattr(self.brain, "_paradigms", []) or [])):
            try:
                if hasattr(paradigm, "on_walk_end"):
                    paradigm.on_walk_end(self, res)  # type: ignore[attr-defined]
            except Exception:
                pass
        for plug in getattr(self, "_wplugins", []) or []:
            try:
                if hasattr(plug, "on_walk_end"):
                    plug.on_walk_end(self, res)  # type: ignore[attr-defined]
            except Exception:
                pass

        res = {
            "loss": final_loss_val,
            "steps": int(steps),
            "visited": int(len(self._visited)),
            "step_metrics": step_metrics,
        }
        try:
            report("wanderer", "walk", res, "metrics")
        except Exception:
            pass
        return res

    def _random_start(self) -> Optional[Neuron]:
        if not self.brain.neurons:
            return None
        try:
            idx = self.rng.randrange(0, len(self.brain.neurons))
            return list(self.brain.neurons.values())[idx]  # type: ignore[call-arg]
        except Exception:
            for n in self.brain.neurons.values():  # type: ignore[call-arg]
                return n
            return None

    def _gather_choices(self, n: Neuron) -> List[Tuple[Synapse, str]]:
        choices: List[Tuple[Synapse, str]] = []
        for s in n.outgoing:
            if s.direction in ("uni", "bi"):
                choices.append((s, "forward"))
        for s in n.incoming:
            if s.direction == "bi":
                choices.append((s, "backward"))
        return choices

    def _random_choice(self, choices: List[Tuple[Synapse, str]]) -> Tuple[Synapse, str]:
        return self.rng.choice(choices)

    def _compute_loss(self, outputs: List[Any], *, override_loss: Optional[Callable[[List[Any]], Any]] = None):
        torch = self._torch  # type: ignore[assignment]
        losses = []
        for plugin in getattr(self, "_wplugins", []) or []:
            try:
                if hasattr(plugin, "loss"):
                    losses.append(plugin.loss(self, outputs))  # type: ignore[attr-defined]
            except Exception:
                pass
        if losses:
            try:
                acc = None
                for t in losses:
                    acc = t if acc is None else (acc + t)
                return acc if acc is not None else torch.tensor(0.0, device=self._device)
            except Exception:
                pass
        if override_loss is not None:
            return override_loss(outputs)
        if isinstance(self._loss_spec, str) and self._loss_spec.startswith("nn."):
            if self._loss_module is None:
                try:
                    nn = getattr(torch, "nn")
                    cls_name = self._loss_spec.split(".", 1)[1]
                    LossCls = getattr(nn, cls_name)
                    self._loss_module = LossCls()
                except Exception as e:
                    raise ValueError(f"Could not resolve loss spec {self._loss_spec}: {e}")
            loss_mod = self._loss_module
            loss_name = type(loss_mod).__name__ if hasattr(loss_mod, "__class__") else ""
            terms = []
            for y in outputs:
                if hasattr(y, "detach") and hasattr(y, "to"):
                    yt = y.float()
                else:
                    yt = torch.tensor([float(v) for v in (y if isinstance(y, (list, tuple)) else [y])],
                                      dtype=torch.float32, device=self._device)
                if self._target_provider is not None:
                    tgt = self._target_provider(yt)
                    if not (hasattr(tgt, "to")):
                        cls_losses_long = {"CrossEntropyLoss", "NLLLoss", "MultiMarginLoss", "MultiLabelMarginLoss"}
                        if loss_name in cls_losses_long:
                            tgt = torch.tensor(tgt, dtype=torch.long, device=self._device)
                        else:
                            tgt = torch.tensor(tgt, dtype=yt.dtype, device=self._device)
                else:
                    tgt = torch.zeros_like(yt)
                try:
                    if hasattr(tgt, "to"):
                        cls_losses_long = {"CrossEntropyLoss", "NLLLoss", "MultiMarginLoss", "MultiLabelMarginLoss"}
                        if loss_name in cls_losses_long:
                            tgt = tgt.to(device=self._device, dtype=torch.long)
                        else:
                            tgt = tgt.to(device=self._device, dtype=yt.dtype)
                except Exception:
                    pass
                try:
                    if hasattr(yt, "view") and hasattr(tgt, "view"):
                        yv = yt.view(-1)
                        tv = tgt if hasattr(tgt, "view") else tgt
                        if hasattr(tv, "view"):
                            tv = tv.view(-1)
                        if tv.numel() == 1:
                            aligned = tgt
                        elif yv.numel() == tv.numel():
                            aligned = tv.view_as(yv)
                        elif tv.numel() < yv.numel():
                            pad_len = int(yv.numel() - tv.numel())
                            pad_vals = torch.zeros(pad_len, dtype=tv.dtype if hasattr(tv, "dtype") else yt.dtype, device=self._device)
                            aligned = torch.cat([tv, pad_vals], dim=0)
                        else:
                            aligned = tv[: yv.numel()]
                        tgt = aligned
                        try:
                            if hasattr(yt, "shape") and hasattr(aligned, "shape") and yt.shape != aligned.shape:
                                yt = yt.view_as(aligned)
                        except Exception:
                            pass
                except Exception:
                    pass
                terms.append(loss_mod(yt, tgt))
            return sum(terms) if terms else torch.tensor(0.0, device=self._device)
        elif callable(self._loss_spec):
            return self._loss_spec(outputs)
        else:
            terms = []
            for y in outputs:
                if hasattr(y, "detach") and hasattr(y, "to") and hasattr(y, "float"):
                    yt = y.float()
                    terms.append((yt.view(-1) ** 2).mean())
                else:
                    t = torch.tensor([float(v) for v in (y if isinstance(y, (list, tuple)) else [y])],
                                     dtype=torch.float32, device=self._device)
                    terms.append((t.view(-1) ** 2).mean())
            return sum(terms) if terms else torch.tensor(0.0, device=self._device)

    def walkfinish(self) -> Tuple[float, Optional[float]]:
        if not self._walk_ctx:
            return (0.0, None)
        outputs: List[Any] = self._walk_ctx.get("outputs", [])
        current: Optional[Neuron] = self._walk_ctx.get("current")
        loss_t = self._compute_loss(outputs)
        loss_v = float(loss_t.detach().to("cpu").item())
        delta = None if self._last_walk_loss is None else (loss_v - self._last_walk_loss)
        out_pos = getattr(current, "position", None) if current is not None else None
        self._finish_stats = {"loss_tensor": loss_t, "loss": loss_v, "delta": delta, "output_neuron": out_pos}
        self._finish_requested = True
        try:
            report("wanderer", "walkfinish", {"loss": loss_v, "delta_vs_prev": delta, "output_neuron_pos": out_pos}, "events")
        except Exception:
            pass
        return (loss_v, delta)


def push_temporary_plugins(
    wanderer: "Wanderer",
    *,
    wanderer_types: Optional[Sequence[str]] = None,
    neuro_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Temporarily add stacks of Wanderer and Neuroplasticity plugins. Returns a handle for restoration.

    Usage:
        handle = push_temporary_plugins(w, wanderer_types=["epsilongreedy"], neuro_types=["base"])  # add
        ... do work ...
        pop_temporary_plugins(w, handle)  # restore previous stacks
    """
    handle: Dict[str, Any] = {}
    handle["wplugins_prev"] = list(getattr(wanderer, "_wplugins", []) or [])
    handle["nplugins_prev"] = list(getattr(wanderer, "_neuro_plugins", []) or [])
    if wanderer_types:
        for nm in wanderer_types:
            plug = _WANDERER_TYPES.get(str(nm))
            if plug is not None:
                getattr(wanderer, "_wplugins").append(plug)
    if neuro_types:
        for nm in neuro_types:
            nplug = _NEURO_TYPES.get(str(nm))
            if nplug is not None:
                getattr(wanderer, "_neuro_plugins").append(nplug)
    return handle


def pop_temporary_plugins(wanderer: "Wanderer", handle: Dict[str, Any]) -> None:
    """Restore Wanderer plugin stacks to the state captured by the given handle."""
    try:
        wprev = handle.get("wplugins_prev")
        nprev = handle.get("nplugins_prev")
        if wprev is not None:
            wanderer._wplugins = list(wprev)
        if nprev is not None:
            wanderer._neuro_plugins = list(nprev)
    except Exception:
        pass


# Core registries for Wanderer and Neuroplasticity plugins
__all__ = [
    "register_wanderer_type",
    "register_neuroplasticity_type",
    "WANDERER_TYPES_REGISTRY",
    "NEURO_TYPES_REGISTRY",
    "Wanderer",
    "push_temporary_plugins",
    "pop_temporary_plugins",
]
