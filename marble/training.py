from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .codec import UniversalTensorCodec, TensorLike
from .datapair import DataPair
from .reporter import report


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
    from .marblemain import Wanderer  # lazy to avoid import cycle

    w = Wanderer(brain, type_name=wanderer_type, seed=seed, loss=loss, target_provider=target_provider)
    history: List[Dict[str, Any]] = []
    brain._progress_total_epochs = 1  # type: ignore[attr-defined]
    brain._progress_epoch = 0  # type: ignore[attr-defined]
    brain._progress_total_walks = num_walks  # type: ignore[attr-defined]
    for i in range(num_walks):
        brain._progress_walk = i  # type: ignore[attr-defined]
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


def create_start_neuron(brain: "Brain", encoded_input: Union[TensorLike, Sequence[float], float, int]) -> "Neuron":
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
    codec: UniversalTensorCodec,
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
    from .marblemain import Wanderer  # lazy import

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

    # Build one shared Wanderer across pairs for consistency
    w = Wanderer(brain, type_name=wanderer_type, seed=seed, loss=loss, neuro_config=neuro_config, gradient_clip=gradient_clip)
    if selfattention is not None:
        try:
            selfattention.attach_to_wanderer(w)
        except Exception:
            pass

    for item in datapairs:
        brain._progress_walk = count  # type: ignore[attr-defined]
        dp = _normalize_pair(item)
        enc_l, enc_r = dp.encode(codec)
        # Create/choose start neuron
        start = left_to_start(dp.left, brain) if left_to_start is not None else create_start_neuron(brain, enc_l)
        stats = w.walk(max_steps=steps_per_pair, start=start, lr=lr)
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

    final_loss = history[-1]["loss"] if history else 0.0
    out = {"history": history, "final_loss": final_loss, "count": count}
    try:
        report("training", "datapair_summary", {"count": count, "final_loss": final_loss}, "datapair")
    except Exception:
        pass
    return out


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
) -> Dict[str, Any]:
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
    # Thread mode only; process intentionally not implemented
    if mode == "thread":
        import threading
        lock = threading.Lock()
        def worker(idx: int):
            res = run_training_with_datapairs(
                brain,
                normed_lists[idx],
                codec,
                steps_per_pair=steps_per_pair,
                lr=lr,
                wanderer_type=wanderer_type,
                seed=seeds[idx] if seeds is not None and idx < len(seeds) else None,
                loss=loss,
                left_to_start=left_to_start,
                neuro_config=neuro_config,
            )
            with lock:
                results.append(res)
        threads = []
        for i in range(len(normed_lists)):
            t = threading.Thread(target=worker, args=(i,), daemon=True)
            threads.append(t); t.start()
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

