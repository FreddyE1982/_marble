"""Example: Streamed quality training with neuroplasticity.

This variant extends ``run_hf_image_quality_noplugins.py`` by enabling
neurogenesis and pruning via the ``BaseNeuroplasticityPlugin``.  New neurons
and synapses are created when a walk gets stuck, while excessive outgoing
connections are pruned to keep the Brain compact.

Usage::

    py -3 examples/run_hf_image_quality_neuroplasticity.py
"""

from __future__ import annotations

print("Importing packages..")

from typing import Iterator, Any, Dict
import os
import re
import time
import subprocess
import threading
import shutil
from pathlib import Path
from datasets import DownloadConfig
import torch

from marble.marblemain import (
    Brain,
    UniversalTensorCodec,
    make_datapair,
    run_training_with_datapairs,
    load_hf_streaming_dataset,
    SelfAttention,
)
import marble.plugins  # ensure plugin discovery

print("...complete")


class QualityAwareRoutine:
    """Adjust LR based on recent loss trend for stability."""

    def __init__(self, window: int = 8, decay: float = 0.9, grow: float = 1.1) -> None:
        self.window = int(window)
        self.decay = float(decay)
        self.grow = float(grow)

    def after_step(self, sa: SelfAttention, ro, wanderer, step_idx: int, ctx):
        hist = sa.history(self.window)
        if len(hist) < 2:
            return None
        losses = [h.get("current_loss") for h in hist if isinstance(h.get("current_loss"), (int, float))]
        if len(losses) < 2:
            return None
        prev, cur = losses[-2], losses[-1]
        base_lr = sa.get_param("lr_override") or sa.get_param("current_lr") or 1e-3
        try:
            base_lr = float(base_lr)
        except Exception:
            base_lr = 1e-3
        if cur > prev:
            new_lr = max(1e-5, base_lr * self.decay)
        else:
            new_lr = min(5e-3, base_lr * self.grow)
        return {"lr_override": float(new_lr)}


def quality_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    """Huber-style loss for quality scores."""
    # ``pred`` carries the neuron's output which, depending on the encoded
    # payload size, may be a high dimensional tensor.  ``target`` on the other
    # hand encodes a single float quality score.  Subtracting mismatched shapes
    # would raise and silently yield a zero loss upstream.  Flatten both sides
    # to scalars so the Huber loss always operates on compatible shapes.
    pred = pred.float().view(-1).mean()
    target = target.float().view(-1).mean()
    diff = pred - target
    abs_diff = torch.abs(diff)
    return torch.where(abs_diff < delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))


def _sample_pairs(ds, max_pairs: int | None = None) -> Iterator:
    """Yield datapairs of (prompt, image) with quality scores."""
    count = 0
    for ex in ds:
        if max_pairs is not None and count >= max_pairs:
            break
        try:
            prompt = ex.get_raw("prompt")
            img1 = ex["image1"]
            img2 = ex["image2"]
            pref1 = float(ex.get_raw("weighted_results_image1_preference"))
            pref2 = float(ex.get_raw("weighted_results_image2_preference"))
            al1 = float(ex.get_raw("weighted_results_image1_alignment"))
            al2 = float(ex.get_raw("weighted_results_image2_alignment"))
            q1 = (pref1 + al1) / 2.0
            q2 = (pref2 + al2) / 2.0
        except Exception as err:
            print(f"skipping example due to: {err}; example={ex}")
            continue
        yield make_datapair({"prompt": prompt, "image": img1}, q1)
        yield make_datapair({"prompt": prompt, "image": img2}, q2)
        count += 2


def main(
    epochs: int = 1,
    max_pairs: int | None = None,
    batch_size: int = 10,
    launch_kuzu: bool | None = None,
) -> None:
    # Image-cache configuration (defaults: enabled=True, size=20); allow env overrides.
    cache_enabled = os.environ.get("MARBLE_IMG_CACHE_ENABLED", "1").strip() not in ("0", "false", "False")
    try:
        cache_size = int(os.environ.get("MARBLE_IMG_CACHE_SIZE", "20"))
    except Exception:
        cache_size = 20
    codec = UniversalTensorCodec()
    try:
        hf_retries = int(os.environ.get("MARBLE_IMG_RETRY", "5"))
    except Exception:
        hf_retries = 5

    print("Connecting to dataset...")
    ds = load_hf_streaming_dataset(
        "Rapidata/Imagen-4-ultra-24-7-25_t2i_human_preference",
        split="train",
        streaming="memory",
        codec=codec,
        download_config=DownloadConfig(max_retries=hf_retries),
        cache_images=cache_enabled,
        cache_size=cache_size,
    )
    print("...done")
    formula = "n1 >= 0 and n2 >= 0"
    dims = (
        max(int(m.group(1)) for m in re.finditer(r"n(\d+)", formula)) if re.search(r"n(\d+)", formula) else 1
    )
    kuzu_db = os.environ.get("MARBLE_KUZU_DB", "brain_topology.db")

    print("Initializing brain...")
    brain = Brain(
        dims,
        size=None,
        formula=formula,
        store_snapshots=True,
        snapshot_path=".",
        snapshot_freq=100,
        snapshot_keep=10,
        kuzu_path=kuzu_db,
    )
    print("...done")

    sa = SelfAttention(routines=[QualityAwareRoutine(window=8)])
    neuro_cfg = {
        "grow_on_step_when_stuck": True,
        "max_new_per_walk": 2,
        "enable_prune": True,
        "prune_if_outgoing_gt": 3,
        "batch_size": batch_size,
    }

    def _run_kuzu_explorer(db_file: str, port: int = 8000) -> str:
        db_path = Path(db_file).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["kuzu_explorer", "--db", str(db_path), "--port", str(port)]
        if shutil.which(cmd[0]) is None:
            return ""

        def _run_cmd():
            try:
                subprocess.Popen(
                    cmd,
                    cwd=db_path.parent,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        threading.Thread(target=_run_cmd, daemon=True).start()
        return f"http://localhost:{port}"

    if launch_kuzu is None:
        launch_kuzu = os.environ.get("MARBLE_ENABLE_KUZU", "0") not in ("0", "false", "False")
    if launch_kuzu:
        kuzu_port = 8000
        kuzu_url = _run_kuzu_explorer(kuzu_db, kuzu_port)
        if kuzu_url:
            print(f"Kuzu Explorer running at {kuzu_url}")
        else:
            print("Kuzu Explorer could not be started")

    def _start_neuron(left: Dict[str, Any], br):
        payload = (left.get("prompt"), left.get("image"))
        enc = codec.encode(payload)
        try:
            idx = br.available_indices()[0]
        except Exception:
            idx = (0,) * int(getattr(br, "n", 1))
        if idx in getattr(br, "neurons", {}):
            n = br.neurons[idx]
        else:
            connect_to = next(iter(br.neurons)) if br.neurons else None
            if connect_to is None:
                n = br.add_neuron(idx, tensor=0.0)
            else:
                n = br.add_neuron(idx, tensor=0.0, connect_to=connect_to)
        n.receive(enc)
        return n

    print("Starting training loop...")
    for _ in range(int(epochs)):
        pairs = _sample_pairs(ds, max_pairs=max_pairs)
        start_time = time.perf_counter()
        res = run_training_with_datapairs(
            brain,
            pairs,
            codec,
            steps_per_pair=20,
            auto_max_steps_every=20,
            lr=1e-3,
            loss=quality_loss,
            selfattention=sa,
            streaming=True,
            batch_size=batch_size,
            left_to_start=_start_neuron,
            neuro_config=neuro_cfg,
            wanderer_type="batchtrainer",
        )
        duration = time.perf_counter() - start_time
        cnt = res.get("count", 0)
        walks = len(res.get("history", [])) or 1
        print(f"processed datapairs: {cnt} in {duration:.2f}s ({duration/walks:.2f}s per walk)")
        if cnt == 0:
            raise RuntimeError("run_training_with_datapairs returned count=0")
    print(
        "streamed quality training complete",
        f"neurons={len(getattr(brain, 'neurons', {}))}",
        f"synapses={len(getattr(brain, 'synapses', []))}",
        f"pruned_n={getattr(brain, 'neurons_pruned', 0)}",
        f"pruned_s={getattr(brain, 'synapses_pruned', 0)}",
    )


if __name__ == "__main__":
    main()
