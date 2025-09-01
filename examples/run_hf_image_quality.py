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

from typing import Iterator, Any, Dict, Optional
import os
import hashlib
import re
from collections import OrderedDict
from datasets import DownloadConfig

from marble.marblemain import (
    Brain,
    UniversalTensorCodec,
    make_datapair,
    run_training_with_datapairs,
    load_hf_streaming_dataset,
    SelfAttention,
    register_wanderer_type,
)
import marble.plugins  # ensure plugin discovery
from marble.plugins.selfattention_adaptive_grad_clip import AdaptiveGradClipRoutine
from marble.plugins.selfattention_findbestneurontype import FindBestNeuronTypeRoutine
from marble.plugins.selfattention_noise_profiler import ContextAwareNoiseRoutine
from marble.dashboard import start_dashboard
from marble.plugins.wanderer_autoplugin import AutoPlugin


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


class _ImageEncodingLRUCache:
    """Simple LRU cache that stores the encoded representation of images.

    - Keys are derived from common URL/path attributes or a stable repr digest.
    - Values are the result of `UniversalTensorCodec.encode(image_obj)`.
    - Used only within this example to avoid repeated downloads when the
      dataset yields the same image multiple times.
    """

    def __init__(self, max_items: int = 20, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.max_items = int(max_items)
        self._od: "OrderedDict[str, Any]" = OrderedDict()

    def _make_key(self, obj: Any) -> str:
        # Prefer explicit URL/path attributes when present (to avoid fetching bytes)
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
        # If it looks like a plain string that could be a URL or path, use it
        if isinstance(obj, str) and obj:
            return f"url:{obj}"
        # Fallback: stable digest of repr (avoids large materialization)
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
            self._od[key] = val  # move to end (most recently used)
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

    def get_or_encode(self, obj: Any, codec: UniversalTensorCodec):
        key = self._make_key(obj)
        cached = self.get(key)
        if cached is not None:
            return key, cached
        enc = codec.encode(obj)
        self.put(key, enc)
        return key, enc


def _sample_pairs(ds, codec: UniversalTensorCodec, cache: _ImageEncodingLRUCache) -> Iterator:
    for ex in ds:
        prompt = ex.get_raw("prompt")
        img1 = ex.get_raw("image1")
        img2 = ex.get_raw("image2")
        pref1 = float(ex.get_raw("weighted_results_image1_preference"))
        pref2 = float(ex.get_raw("weighted_results_image2_preference"))
        al1 = float(ex.get_raw("weighted_results_image1_alignment"))
        al2 = float(ex.get_raw("weighted_results_image2_alignment"))
        q1 = (pref1 + al1) / 2.0
        q2 = (pref2 + al2) / 2.0
        # Use cache to encode images once; store only the encoded form.
        key1, _ = cache.get_or_encode(img1, codec)
        key2, _ = cache.get_or_encode(img2, codec)
        # Pass lightweight keys in the datapair to avoid triggering downloads on encode.
        yield make_datapair({"prompt": prompt, "image_key": key1}, q1)
        yield make_datapair({"prompt": prompt, "image_key": key2}, q2)


def main(epochs: int = 1) -> None:
    # Image-cache configuration (defaults: enabled=True, size=20); allow env overrides.
    cache_enabled = os.environ.get("MARBLE_IMG_CACHE_ENABLED", "1").strip() not in ("0", "false", "False")
    try:
        cache_size = int(os.environ.get("MARBLE_IMG_CACHE_SIZE", "20"))
    except Exception:
        cache_size = 20
    codec = UniversalTensorCodec()
    ds = load_hf_streaming_dataset(
        "Rapidata/Imagen-4-ultra-24-7-25_t2i_human_preference",
        split="train",
        streaming=True,
        codec=codec,
        download_config=DownloadConfig(max_retries=5, timeout=60),
    )
    # Consumed fields: prompt, image1, image2, weighted_results_image1_preference,
    # weighted_results_image2_preference, weighted_results_image1_alignment,
    # weighted_results_image2_alignment
    formula = "n1 >= 0 and n2 >= 0"
    dims = (
        max(int(m.group(1)) for m in re.finditer(r"n(\d+)", formula))
        if re.search(r"n(\d+)", formula)
        else 1
    )
    brain = Brain(
        dims,
        size=None,
        formula=formula,
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
    register_wanderer_type("autoplugin_logger", AutoPlugin(log_path="autoplugin.log"))
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
        "autoplugin_logger",
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
        "aggressive_starting_neuroplasticity": True,
        "add_min_new_neurons_per_step": 5,
        "aggressive_phase_steps": 100,
    }
    cache = _ImageEncodingLRUCache(max_items=cache_size, enabled=cache_enabled)
    port = 8501
    start_dashboard(port)
    print(f"Dashboard available at https://alpaca-model-easily.ngrok-free.app:{port}")

    def _start_neuron(left: Dict[str, Any], br):
        # Compose a compact input using the cached image encoding whenever available
        key = left.get("image_key")
        enc_img = None
        if isinstance(key, str):
            enc_img = cache.get(key)
        # Build a tuple combining the raw prompt and cached image tokens (or the key if missing)
        payload = (left.get("prompt"), enc_img if enc_img is not None else key)
        enc = codec.encode(payload)
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
        pairs = _sample_pairs(ds, codec, cache)
        res = run_training_with_datapairs(
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
            dashboard=True,
        )
        cnt = res.get("count", 0)
        print(f"processed datapairs: {cnt}")
        if cnt == 0:
            raise RuntimeError("run_training_with_datapairs returned count=0")
    print("streamed quality training complete")


if __name__ == "__main__":
    main()
