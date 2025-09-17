from __future__ import annotations

import hashlib
import math
import random
import time
import threading
import gc
from threading import Lock

LockType = type(Lock())
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .codec import UniversalTensorCodec, TensorLike
from .datapair import DataPair
from .reporter import report
from .lobe import Lobe
from .learnables_yaml import updatelearnablesyaml


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
    neuro_config: Optional[Dict[str, Any]] = None,
    lobe: Optional[Lobe] = None,
    optimizer: Optional[Union[str, Any]] = None,
    mixedprecision: bool = True,
    dashboard: bool = False,
) -> Dict[str, Any]:
    from .marblemain import Wanderer  # lazy to avoid import cycle
    updatelearnablesyaml()
    def _inner() -> Dict[str, Any]:
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
            target_provider=target_provider,
            neuro_config=cfg,
            optimizer=optimizer,
            mixedprecision=mixedprecision,
        )
        try:
            setattr(w, "pbar_leave", True)
        except Exception:
            pass
        history: List[Dict[str, Any]] = []
        brain._progress_total_epochs = 1  # type: ignore[attr-defined]
        brain._progress_epoch = 0  # type: ignore[attr-defined]
        brain._progress_total_walks = num_walks  # type: ignore[attr-defined]
        for i in range(num_walks):
            brain._progress_walk = i  # type: ignore[attr-defined]
            start = start_selector(brain) if start_selector is not None else None
            stats = w.walk(max_steps=max_steps, start=start, lr=lr, lobe=lobe)
            stats["plugins"] = [p.__class__.__name__ for p in getattr(w, "_wplugins", []) or []]
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

    lock = getattr(brain, "_train_lock", None)
    if not isinstance(lock, LockType):
        lock = Lock()
        setattr(brain, "_train_lock", lock)
    if dashboard:
        try:
            from .dashboard import start_dashboard
            start_dashboard()
        except Exception:
            pass
    with lock:
        return _inner()


def create_start_neuron(brain: "Brain", encoded_input: Union[TensorLike, Sequence[float], float, int]) -> "Neuron":
    try:
        avail = brain.available_indices()
        idx = avail[0] if avail else (0,) * int(getattr(brain, "n", 1))
    except Exception:
        idx = (0,)
    if idx in getattr(brain, "neurons", {}):
        n = brain.neurons[idx]
    else:
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
    codec: UniversalTensorCodec,
    *,
    steps_per_pair: Optional[int] = None,
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
    lobe: Optional[Lobe] = None,
    batch_size: Optional[int] = None,
    streaming: bool = True,
    optimizer: Optional[Union[str, Any]] = None,
    mixedprecision: bool = True,
    dashboard: bool = False,
    auto_max_steps_every: Optional[int] = None,
) -> Dict[str, Any]:
    from .marblemain import Wanderer  # lazy import
    updatelearnablesyaml()
    def _inner() -> Dict[str, Any]:
        nonlocal auto_max_steps_every
        effective_batch_size = 1
        if batch_size is not None:
            try:
                effective_batch_size = max(1, int(batch_size))
            except Exception:
                effective_batch_size = 1
        elif isinstance(neuro_config, dict) and "batch_size" in neuro_config:
            try:
                effective_batch_size = max(1, int(neuro_config.get("batch_size", 1)))
            except Exception:
                effective_batch_size = 1
        use_enhanced = (
            effective_batch_size > 1
            and optimizer is None
            and not dashboard
        )
        if use_enhanced:
            try:
                from . import marblemain as _mm

                impl = getattr(_mm, "_RUN_TRAINING_WITH_DATAPAIRS_IMPL", None)
            except Exception:
                impl = None
            if callable(impl):
                kwargs_impl = {
                    "steps_per_pair": steps_per_pair,
                    "lr": lr,
                    "wanderer_type": wanderer_type,
                    "train_type": train_type,
                    "seed": seed,
                    "loss": loss,
                    "left_to_start": left_to_start,
                    "neuro_config": neuro_config,
                    "callback": callback,
                    "gradient_clip": gradient_clip,
                    "selfattention": selfattention,
                    "streaming": streaming,
                    "batch_size": effective_batch_size,
                    "lobe": lobe,
                    "mixedprecision": mixedprecision,
                    "auto_max_steps_every": auto_max_steps_every,
                }
                return impl(brain, datapairs, codec, **kwargs_impl)
        from .plugins import wanderer_resource_allocator as resource_allocator
        history: List[Dict[str, Any]] = []
        count = 0
        try:
            brain._progress_total_walks = len(datapairs)  # type: ignore[attr-defined]
        except Exception:
            brain._progress_total_walks = 0  # type: ignore[attr-defined]

        def _normalize_pair(p: Union[DataPair, Tuple[Any, Any], Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]]]) -> DataPair:
            if isinstance(p, DataPair):
                return p
            if isinstance(p, tuple) and len(p) == 2:
                a, b = p
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
                    return DataPair(a, b)
            return DataPair(p, None)  # type: ignore[arg-type]

        def _dataset_sig(iterable) -> str:
            h = hashlib.sha256()
            max_items = 64
            for idx, it in enumerate(iterable):
                if idx >= max_items:
                    break
                dp = _normalize_pair(it)
                enc_l, enc_r = dp.encode(codec)
                def chunk(x):
                    try:
                        if hasattr(x, "tolist"):
                            return bytes(int(v) & 0xFF for v in x.view(-1).tolist()[:32])
                        return bytes(int(v) & 0xFF for v in list(x)[:32])
                    except Exception:
                        return b""
                h.update(chunk(enc_l)); h.update(chunk(enc_r))
            return h.hexdigest()

        if not streaming:
            if getattr(brain, "_dataset_signature", None) is None:
                try:
                    brain._dataset_signature = _dataset_sig(datapairs)
                except Exception:
                    brain._dataset_signature = None
            else:
                if not getattr(brain, "allow_dissimilar_datasets_in_wanderers", False):
                    try:
                        if brain._dataset_signature != _dataset_sig(datapairs):
                            raise ValueError("Dataset signature differs from the first-run signature for this Brain")
                    except Exception:
                        pass

        cfg = neuro_config
        wtype = wanderer_type
        if lobe is not None and not getattr(lobe, "inherit_plugins", True):
            wtype = lobe.plugin_types
            cfg = lobe.neuro_config
        cfg = dict(cfg or {})
        if batch_size is not None:
            cfg["batch_size"] = batch_size
        cfg["streaming"] = streaming
        w = Wanderer(
            brain,
            type_name=wtype,
            seed=seed,
            loss=loss,
            neuro_config=cfg,
            gradient_clip=gradient_clip,
            optimizer=optimizer,
            mixedprecision=mixedprecision,
        )
        # Ensure progress bars remain visible for the datapair helper
        try:
            setattr(w, "pbar_leave", True)
        except Exception:
            pass
        if selfattention is not None:
            try:
                from .selfattention import attach_selfattention
                attach_selfattention(w, selfattention)
            except Exception:
                pass
        # Allow any enabled learning paradigms to configure the Wanderer
        try:
            from .marblemain import apply_paradigms_to_wanderer  # lazy to avoid cycles
            apply_paradigms_to_wanderer(brain, w)
        except Exception:
            pass

        # Resolve Brain-train plugins (comma-separated or list) and initialize
        train_plugins: List[Any] = []
        try:
            from .marblemain import _BRAIN_TRAIN_TYPES, _on_init_train, _on_end_train, _merge_dict_safe
            from .marblemain import _auto_max_steps_interval
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
        except Exception:
            # If registry not available, continue without brain-train plugins
            _BRAIN_TRAIN_TYPES = {}
            def _on_init_train(*args, **kwargs):
                return None
            def _on_end_train(*args, **kwargs):
                return None
            def _merge_dict_safe(base, extra):
                return base

        if auto_max_steps_every is None:
            auto_max_steps_every = _auto_max_steps_interval()
        if steps_per_pair is None or steps_per_pair <= 0:
            current_max_steps = max(1, brain.longest_path_steps())
        else:
            current_max_steps = int(steps_per_pair)
        init_cfg = {
            "num_walks": None,
            "max_steps": current_max_steps,
            "lr": lr,
            "type_name": train_type,
        }
        for p in train_plugins:
            try:
                _on_init_train(p, brain, w, init_cfg)
            except Exception:
                pass

        for item in datapairs:
            brain._progress_walk = count  # type: ignore[attr-defined]
            dp = _normalize_pair(item)
            class _Holder(dict):
                pass
            holder: Dict[str, Any] = _Holder()
            with resource_allocator.track_tensor(holder, "enc_l"):
                with resource_allocator.track_tensor(holder, "enc_r"):
                    holder["enc_l"], holder["enc_r"] = dp.encode(codec)
            enc_l, enc_r = holder["enc_l"], holder["enc_r"]

            if enc_r is not None:
                def _tp(_y, target=enc_r):
                    return target
                w._target_provider = _tp
            else:
                w._target_provider = None

            start = left_to_start(dp.left, brain) if left_to_start is not None else create_start_neuron(brain, enc_l)
            stats = w.walk(max_steps=current_max_steps, start=start, lr=lr, lobe=lobe)
            with resource_allocator.track_tensor(stats, "loss"):
                with resource_allocator.track_tensor(stats, "delta"):
                    finish_loss, delta = w.walkfinish()
                    stats["loss"] = finish_loss
                    stats["delta"] = delta
            stats["plugins"] = [p.__class__.__name__ for p in getattr(w, "_wplugins", []) or []]
            history.append(stats)
            count += 1
            try:
                report("training", f"pair_{count}", {"loss": stats.get("loss", 0.0), "steps": stats.get("steps", 0)}, "datapair")
            except Exception:
                pass
            if callback is not None:
                try:
                    callback(count - 1, stats, dp)
                except Exception:
                    pass
            if (steps_per_pair is None or steps_per_pair <= 0) and auto_max_steps_every and count % auto_max_steps_every == 0:
                current_max_steps = max(1, brain.longest_path_steps())
            if streaming:
                # Drop references to processed datapairs to free memory immediately
                del dp, enc_l, enc_r, start
                gc.collect()

        final_loss = history[-1]["loss"] if history else 0.0
        out = {"history": history, "final_loss": final_loss, "count": count}
        # Allow train plugins to contribute summary fields
        for p in train_plugins:
            try:
                extra = _on_end_train(p, brain, w, history)
                out = _merge_dict_safe(out, extra)
            except Exception:
                pass
        status = getattr(brain, "status", lambda: {})()
        out["status"] = status
        try:
            report("training", "datapair_summary", {"count": count, "final_loss": final_loss}, "datapair")
        except Exception:
            pass
        try:
            report("training", "status", status, "datapair")
        except Exception:
            pass
        return out

    lock = getattr(brain, "_train_lock", None)
    if not isinstance(lock, LockType):
        lock = Lock()
        setattr(brain, "_train_lock", lock)
    if dashboard:
        try:
            from .dashboard import start_dashboard
            start_dashboard()
        except Exception:
            pass
    with lock:
        return _inner()


def run_wanderer_epochs_with_datapairs(
    brain: "Brain",
    datapairs: Iterable[Union[DataPair, Tuple[Any, Any], Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]]]],
    codec: UniversalTensorCodec,
    *,
    num_epochs: int = 1,
    steps_per_pair: int = 5,
    lr: float = 1e-2,
    wanderer_type: Optional[str] = None,
    seed: Optional[int] = None,
    loss: Optional[Union[str, Callable[..., Any], Any]] = "nn.MSELoss",
    left_to_start: Optional[Callable[[Any, "Brain"], Optional["Neuron"]]] = None,
    callback: Optional[Callable[[int, int, Dict[str, Any], DataPair], None]] = None,
    batch_size: Optional[int] = None,
    streaming: bool = True,
    optimizer: Optional[Union[str, Any]] = None,
    mixedprecision: bool = True,
    dashboard: bool = False,
) -> Dict[str, Any]:
    def _inner() -> Dict[str, Any]:
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
        brain._progress_total_epochs = num_epochs  # type: ignore[attr-defined]
        for e in range(num_epochs):
            brain._progress_epoch = e  # type: ignore[attr-defined]
            brain._progress_total_walks = len(dataset)  # type: ignore[attr-defined]
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
                batch_size=batch_size,
                streaming=streaming,
                optimizer=optimizer,
                mixedprecision=mixedprecision,
                dashboard=dashboard,
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

    lock = getattr(brain, "_train_lock", None)
    if not isinstance(lock, LockType):
        lock = Lock()
        setattr(brain, "_train_lock", lock)
    if dashboard:
        try:
            from .dashboard import start_dashboard
            start_dashboard()
        except Exception:
            pass
    with lock:
        return _inner()


def run_wanderers_parallel(
    brain: "Brain",
    datasets: List[Iterable[Union[DataPair, Tuple[Any, Any], Tuple[Union[TensorLike, Sequence[int]], Union[TensorLike, Sequence[int]]]]]],
    codec: UniversalTensorCodec,
    *,
    mode: str = "thread",
    steps_per_pair: int = 5,
    lr: float = 1e-2,
    wanderer_type: Optional[str] = None,
    seeds: Optional[List[Optional[int]]] = None,
    loss: Optional[Union[str, Callable[..., Any], Any]] = "nn.MSELoss",
    left_to_start: Optional[Callable[[Any, "Brain"], Optional["Neuron"]]] = None,
    neuro_config: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    streaming: bool = True,
    optimizer: Optional[Union[str, Any]] = None,
    mixedprecision: bool = True,
) -> List[Dict[str, Any]]:
    sigs: List[str] = []
    normed_lists: List[List[Any]] = []
    for ds in datasets:
        lst = list(ds)
        normed_lists.append(lst)
        h = hashlib.sha256()
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
    if not getattr(brain, "allow_dissimilar_datasets_in_wanderers", False):
        if len(set(sigs)) > 1:
            raise ValueError("All datasets must have similar signatures unless brain.allow_dissimilar_datasets_in_wanderers=True")

    results: List[Dict[str, Any]] = []
    if mode == "thread":
        def worker(idx: int) -> None:
            seed = seeds[idx] if seeds is not None and idx < len(seeds) else None
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
                neuro_config=neuro_config,
                batch_size=batch_size,
                streaming=streaming,
                optimizer=optimizer,
                mixedprecision=mixedprecision,
            )
            results.append(res)

        threads: List[threading.Thread] = []
        for i in range(len(normed_lists)):
            t = threading.Thread(target=worker, args=(i,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    elif mode == "process":
        raise NotImplementedError("process mode not implemented")
    else:
        raise ValueError("mode must be 'thread' or 'process'")
    return results


def make_default_codec() -> UniversalTensorCodec:
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
    codec: Optional[UniversalTensorCodec] = None,
    neuro_config: Optional[Dict[str, Any]] = None,
    gradient_clip: Optional[Dict[str, Any]] = None,
    selfattention: Optional["SelfAttention"] = None,
    batch_size: Optional[int] = None,
    streaming: bool = True,
    optimizer: Optional[Union[str, Any]] = None,
    mixedprecision: bool = True,
) -> Dict[str, Any]:
    from .marblemain import Brain  # lazy import

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
        batch_size=batch_size,
        streaming=streaming,
        optimizer=optimizer,
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


__all__ = [
    "run_wanderer_training",
    "create_start_neuron",
    "run_training_with_datapairs",
    "run_wanderer_epochs_with_datapairs",
    "run_wanderers_parallel",
    "make_default_codec",
    "quick_train_on_pairs",
]
