"""Example: Streamed quality training on a Hugging Face dataset.

This script demonstrates how to:
- Load the Rapidata Imagen preference dataset in streaming mode
- Derive quality quotients from human preference and alignment scores
- Train a small Brain with stacked Wanderer plugins and neuroplasticity
- Employ a custom SelfAttention routine combined with adaptive grad clipping
- Gate Wanderer/neuroplasticity plugins via ``autoplugin`` and let neurons
  switch types dynamically through ``autoneuron``

Usage:
    py -3 examples/run_hf_image_quality.py
"""

from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from typing import Iterator
import urllib.request

from PIL import Image

from marble.marblemain import (
    Brain,
    UniversalTensorCodec,
    make_datapair,
    run_training_with_datapairs,
    load_hf_streaming_dataset,
    SelfAttention,
)
import marble.plugins  # ensure plugin discovery
from marble.plugins.selfattention_adaptive_grad_clip import AdaptiveGradClipRoutine
from marble.plugins.selfattention_findbestneurontype import FindBestNeuronTypeRoutine
from marble.plugins.selfattention_noise_profiler import ContextAwareNoiseRoutine


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


class ImageCache:
    """Simple in-memory LRU cache for image downloads."""

    def __init__(self, max_images: int = 64) -> None:
        self.max_images = int(max_images)
        self._store: OrderedDict[str, Image.Image] = OrderedDict()

    def get(self, url: str) -> Image.Image:
        if url in self._store:
            self._store.move_to_end(url)
            return self._store[url]
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        self._store[url] = img
        if len(self._store) > self.max_images:
            self._store.popitem(last=False)
        return img


def _sample_pairs(ds, cache: ImageCache) -> Iterator:
    for ex in ds:
        prompt = ex.get_raw("prompt")
        img1_info = ex.get_raw("image1")
        img2_info = ex.get_raw("image2")
        url1 = img1_info.get("path") if isinstance(img1_info, dict) else str(img1_info)
        url2 = img2_info.get("path") if isinstance(img2_info, dict) else str(img2_info)
        img1 = cache.get(url1)
        img2 = cache.get(url2)
        pref1 = float(ex.get_raw("weighted_results_image1_preference"))
        pref2 = float(ex.get_raw("weighted_results_image2_preference"))
        al1 = float(ex.get_raw("weighted_results_image1_alignment"))
        al2 = float(ex.get_raw("weighted_results_image2_alignment"))
        q1 = (pref1 + al1) / 2.0
        q2 = (pref2 + al2) / 2.0
        yield make_datapair({"prompt": prompt, "image": img1}, q1)
        yield make_datapair({"prompt": prompt, "image": img2}, q2)


def main(epochs: int = 1, cache_size: int = 64) -> None:
    codec = UniversalTensorCodec()
    ds = load_hf_streaming_dataset(
        "Rapidata/Imagen-4-ultra-24-7-25_t2i_human_preference",
        split="train",
        streaming=True,
        codec=codec,
    )
    cache = ImageCache(max_images=cache_size)
    # Consumed fields: prompt, image1, image2, weighted_results_image1_preference,
    # weighted_results_image2_preference, weighted_results_image1_alignment,
    # weighted_results_image2_alignment
    brain = Brain(
        2,
        size=32,
        bounds=((0, 31), (0, 31)),
        formula="abs(n1 - n2) <= 2",
        store_snapshots=True,
        snapshot_path=".",
        snapshot_freq=100,
        snapshot_keep=10,
    )
    # Include Brain-training plugins to adjust learning rate and step schedule
    sa = SelfAttention(
        routines=[
            QualityAwareRoutine(window=8),
            AdaptiveGradClipRoutine(),
            FindBestNeuronTypeRoutine(),
            ContextAwareNoiseRoutine(),
        ]
    )
    wplugins = [
        "batchtrainer",
        "qualityweightedloss",
        "epsilongreedy",
        "td_qlearning",
        "bestlosspath",
        "alternatepathscreator",
        "l2_weight_penalty",
        "distillation",
        "wanderalongsynapseweights",
        "dynamicdimensions",
        "autoplugin",
    ]
    neuro_cfg = {
        "grow_on_step_when_stuck": True,
        "max_new_per_walk": 1,
        "enable_prune": True,
        "prune_if_outgoing_gt": 3,
        "epsilongreedy_epsilon": 0.15,
        "rl_epsilon": 0.1,
        "rl_alpha": 0.05,
        "rl_gamma": 0.95,
        "l2_lambda": 1e-4,
        "batch_size": 5,
    }
    def _start_neuron(left, br):
        enc = codec.encode(left)
        try:
            idx = br.available_indices()[0]
        except Exception:
            idx = (0,) * int(getattr(br, "n", 1))
        if idx in getattr(br, "neurons", {}):
            n = br.neurons[idx]
        else:
            n = br.add_neuron(idx, tensor=0.0, type_name="autoneuron")
        n.receive(enc)
        return n

    for _ in range(int(epochs)):
        pairs = _sample_pairs(ds, cache)
        run_training_with_datapairs(
            brain,
            pairs,
            codec,
            steps_per_pair=2,
            lr=1e-3,
            wanderer_type=",".join(wplugins),
            train_type="warmup_decay,curriculum,qualityaware",
            neuro_config=neuro_cfg,
            selfattention=sa,
            streaming=True,
            batch_size=5,
            left_to_start=_start_neuron,
        )
    print("streamed quality training complete")


if __name__ == "__main__":
    main()

