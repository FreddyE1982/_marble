"""Example: Streamed quality training on a Hugging Face dataset.

This script demonstrates how to:
- Load the Rapidata Imagen preference dataset via memory streaming (one shard at a time)
- Derive quality quotients from human preference and alignment scores
- Train a small Brain with stacked Wanderer plugins and neuroplasticity
- Employ a custom SelfAttention routine combined with adaptive grad clipping
- Gate Wanderer/neuroplasticity plugins via ``autoplugin`` and let neurons
  switch types dynamically through ``autoneuron``

Usage:
    py -3 examples/run_hf_image_quality.py
"""

from __future__ import annotations

from typing import Iterator, Any, Dict
import os
import re
import types
import subprocess
import threading
from pathlib import Path
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
from marble.plugins.wanderer_autoplugin import AutoPlugin
from marble.plugins.wanderer_resource_allocator import clear as clear_resources


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




def _sample_pairs(ds) -> Iterator:
    """Yield datapairs of (prompt, image) with quality scores.

    The HFStreamingDatasetWrapper handles all retry and caching logic, so we
    simply access fields directly without custom wrappers.
    """

    for ex in ds:
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


def main(epochs: int = 1) -> None:
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
    ds = load_hf_streaming_dataset(
        "Rapidata/Imagen-4-ultra-24-7-25_t2i_human_preference",
        split="train",
        streaming="memory",  # stream shards into memory and encode images immediately
        codec=codec,
        download_config=DownloadConfig(max_retries=hf_retries),
        cache_images=cache_enabled,
        cache_size=cache_size,
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
    kuzu_db = os.environ.get("MARBLE_KUZU_DB", "brain_topology.db")
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
    # Ensure every newly added neuron defaults to the autoneuron type
    _orig_add = brain.add_neuron

    def _add_autoneuron(self, index, *, tensor=0.0, **kwargs):
        """Proxy ``Brain.add_neuron`` to default to ``autoneuron`` type.

        The original implementation defined this helper without a ``self``
        parameter and relied on ``types.MethodType`` to bind it.  Plugins such
        as :class:`FindBestNeuronTypeRoutine` capture ``brain.add_neuron`` and
        forward positional arguments, which led to a ``TypeError`` because the
        bound method implicitly injected the ``Brain`` instance as the first
        argument.

        Explicitly accepting ``self`` keeps the signature compatible with the
        intercepted calls while still delegating to the original method.
        """

        kwargs.setdefault("type_name", "autoneuron")
        return _orig_add(index, tensor=tensor, **kwargs)

    brain.add_neuron = types.MethodType(_add_autoneuron, brain)
    # Include Brain-training plugins to adjust learning rate and step schedule
    sa = SelfAttention(
        routines=[
            QualityAwareRoutine(window=8),
            AdaptiveGradClipRoutine(),
            FindBestNeuronTypeRoutine(),
            ContextAwareNoiseRoutine(),
        ]
    )
    mandatory_plugins = [
        "BatchTrainingPlugin",
        "QualityWeightedLossPlugin",
        "EpsilonGreedyChooserPlugin",
        "TDQLearningPlugin",
        "BestLossPathPlugin",
        "AlternatePathsCreatorPlugin",
        "L2WeightPenaltyPlugin",
        "DistillationPlugin",
        "WanderAlongSynapseWeightsPlugin",
        "DynamicDimensionsPlugin",
        "QualityAwareRoutine",
        "AdaptiveGradClipRoutine",
        "FindBestNeuronTypeRoutine",
        "ContextAwareNoiseRoutine",
    ]
    # Instantiate AutoPlugin with mandatory plugin support when available.
    try:
        ap = AutoPlugin(
            log_path="autoplugin.log", mandatory_plugins=mandatory_plugins
        )
    except TypeError:
        # Older AutoPlugin versions do not accept ``mandatory_plugins``.
        ap = AutoPlugin(log_path="autoplugin.log")
        # Preserve mandatory behaviour if the attribute exists.
        if hasattr(ap, "_mandatory"):
            ap._mandatory.update(mandatory_plugins)
        else:
            ap._mandatory = set(mandatory_plugins)
    register_wanderer_type("autoplugin_logger", ap)
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
    def _run_kuzu_explorer(db_file: str, port: int = 8000) -> str:
        db_path = Path(db_file).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "docker",
            "run",
            "-p",
            f"{port}:8000",
            "-v",
            f"{db_path.parent}:/database",
            "-e",
            f"KUZU_FILE={db_path.name}",
            "--rm",
            "kuzudb/explorer:latest",
        ]

        def _run_docker():
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        public_url = None
        threading.Thread(target=_run_docker, daemon=True).start()

        try:
            from pyngrok import ngrok

            ngrok.set_auth_token("2o9DgUKuP2W8vjV7cZFq0sDiM3A_2d1gWrkXqvy5APpUn2QNS")
            tunnel = ngrok.connect(port, "http", bind_tls=False)
            public_url = tunnel.public_url
        except Exception:
            pass

        return public_url or ""

    kuzu_port = 8000
    kuzu_url = _run_kuzu_explorer(kuzu_db, kuzu_port)
    if kuzu_url:
        print(f"Kuzu Explorer available at {kuzu_url}")
    else:
        print("Kuzu Explorer tunnel could not be established")

    def _start_neuron(left: Dict[str, Any], br):
        # Combine the raw prompt with the already encoded image
        payload = (left.get("prompt"), left.get("image"))
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
        pairs = _sample_pairs(ds)
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
            dashboard=False,
        )
        cnt = res.get("count", 0)
        print(f"processed datapairs: {cnt}")
        if cnt == 0:
            raise RuntimeError("run_training_with_datapairs returned count=0")
        clear_resources()
    print("streamed quality training complete")
    clear_resources()


if __name__ == "__main__":
    main()
