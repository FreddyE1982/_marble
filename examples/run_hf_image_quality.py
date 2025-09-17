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

print("Importing packages..")

from typing import Iterator, Any, Dict
import io
import os
import re
import subprocess
import threading
import shutil
import time
import types
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from operator import getitem
from datasets import DownloadConfig
from PIL import Image
import torch

from marble.marblemain import (
    Brain,
    UniversalTensorCodec,
    make_datapair,
    run_training_with_datapairs,
    load_hf_streaming_dataset,
    SelfAttention,
    register_wanderer_type,
    expand_wplugins,
)
import marble.plugins  # ensure plugin discovery
from marble.plugins.selfattention_adaptive_grad_clip import AdaptiveGradClipRoutine
from marble.plugins.selfattention_findbestneurontype import FindBestNeuronTypeRoutine
from marble.plugins.selfattention_noise_profiler import ContextAwareNoiseRoutine
from marble.plugins.wanderer_autoplugin import AutoPlugin
from marble.plugins.wanderer_resource_allocator import clear as clear_resources

print("...complete")
PREP_TIMES: list[float] = []
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




def _scale_image_max_side(image: Any, target: int = 500) -> Any:
    """Return ``image`` with its larger side clamped to ``target`` pixels.

    Byte inputs are re-encoded after resizing so callers relying on the
    ``bytes`` fast-path of :class:`UniversalTensorCodec` keep benefiting from
    compressed payloads.  When the source image is a JPEG, reuse its original
    quantization tables (falling back to lossless PNG if unavailable) so the
    saved pixels retain the compression artefacts encoded in the quality
    labels.
    """

    try:
        target = int(target)
    except Exception:
        target = 500
    target = max(1, target)

    def _resize_if_needed(pil_img: Image.Image) -> tuple[Image.Image, bool]:
        width, height = pil_img.size
        if not width or not height:
            return pil_img, False
        if width >= height:
            new_width = target
            new_height = max(1, int(round(height * target / width)))
        else:
            new_height = target
            new_width = max(1, int(round(width * target / height)))
        if (width, height) == (new_width, new_height):
            return pil_img, False
        return pil_img.resize((new_width, new_height), Image.LANCZOS), True

    if isinstance(image, Image.Image):
        resized, changed = _resize_if_needed(image)
        return resized if changed else image

    if hasattr(image, "shape"):
        try:
            pil_img = Image.fromarray(image)  # type: ignore[arg-type]
        except Exception:
            return image
        resized, _ = _resize_if_needed(pil_img)
        return resized

    if isinstance(image, (bytes, bytearray)):
        try:
            with Image.open(io.BytesIO(image)) as opened:
                resized, changed = _resize_if_needed(opened)
                if not changed:
                    return image
                fmt = opened.format or "PNG"
                save_format = fmt.upper()
                to_save = resized
                info: Dict[str, Any] = dict(getattr(opened, "info", {}) or {})
                save_kwargs: Dict[str, Any] = {}
                if save_format == "JPEG":
                    if to_save.mode not in ("RGB", "L"):
                        to_save = to_save.convert("RGB")
                    quantization = getattr(opened, "quantization", None)
                    if quantization:
                        save_kwargs["qtables"] = quantization
                        subsampling = info.get("subsampling")
                        if subsampling is not None:
                            save_kwargs["subsampling"] = subsampling
                    else:
                        quality = info.get("quality")
                        subsampling = info.get("subsampling")
                        if quality is not None:
                            save_kwargs["quality"] = quality
                            if subsampling is not None:
                                save_kwargs["subsampling"] = subsampling
                        else:
                            save_format = "PNG"
                            to_save = to_save.convert("RGB") if to_save.mode == "CMYK" else to_save
                            save_kwargs = {}
                    if save_format == "JPEG":
                        for key in ("progressive", "optimize", "dpi", "icc_profile", "exif"):
                            value = info.get(key)
                            if value is not None:
                                save_kwargs[key] = value
                if save_format == "PNG" and to_save.mode == "CMYK":
                    to_save = to_save.convert("RGB")
                with io.BytesIO() as out:
                    to_save.save(out, format=save_format, **save_kwargs)
                    data = out.getvalue()
                return type(image)(data)
        except Exception:
            return image

    return image


def _sample_pairs(ds, max_pairs: int | None = None) -> Iterator:
    """Yield datapairs of (prompt, image) with quality scores.

    The HFStreamingDatasetWrapper handles all retry and caching logic, so we
    simply access fields directly without custom wrappers.  When ``max_pairs``
    is provided, only that many datapairs are yielded to keep example runs
    snappy.
    """

    count = 0
    get_dp = make_datapair  # local bind for speed
    codec = UniversalTensorCodec()
    # Parallelize image encoding and numeric conversions with a slightly
    # larger thread pool to shrink per-sample latency.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ex in ds:
            if max_pairs is not None and count >= max_pairs:
                break
            try:
                raw = ex._data
                prompt = raw["prompt"]
                img_raw1 = _scale_image_max_side(raw["image1"])
                img_raw2 = _scale_image_max_side(raw["image2"])
                f = float
                # Launch image encodes and float conversions concurrently.
                f1 = pool.submit(codec.encode, img_raw1)
                f2 = pool.submit(codec.encode, img_raw2)
                pref1, pref2, al1, al2 = pool.map(
                    f,
                    [
                        raw["weighted_results_image1_preference"],
                        raw["weighted_results_image2_preference"],
                        raw["weighted_results_image1_alignment"],
                        raw["weighted_results_image2_alignment"],
                    ],
                )
                img1 = f1.result()
                img2 = f2.result()
                q1 = 0.5 * (pref1 + al1)
                q2 = 0.5 * (pref2 + al2)
            except Exception as err:
                print(f"skipping example due to: {err}; example={ex}")
                continue

            PREP_TIMES.append(time.perf_counter())
            yield get_dp({"prompt": prompt, "image": img1}, q1)
            PREP_TIMES.append(time.perf_counter())
            yield get_dp({"prompt": prompt, "image": img2}, q2)
            count += 2
def main(
    epochs: int = 1,
    max_pairs: int | None = None,
    batch_size: int = 10,
    launch_kuzu: bool | None = None,
    min_new_neurons: int = 1,
    tensorboard: bool = False,
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
        streaming="memory",  # stream shards into memory and encode images immediately
        codec=codec,
        download_config=DownloadConfig(max_retries=hf_retries),
        cache_images=cache_enabled,
        cache_size=cache_size,
    )
    print("...done")
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
        tensorboard=tensorboard,
    )

    if tensorboard and getattr(brain, "tensorboard_logdir", None):
        logdir = brain.tensorboard_logdir
        print("TensorBoard ready! In a notebook, run:")
        print(f"  %tensorboard --logdir {logdir}")

    print("...done")
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
            #FindBestNeuronTypeRoutine(),
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
        # The following plugins are intentionally left optional so that
        # ``autoplugin`` can learn when to enable or disable them.
        # "MixedPrecisionPlugin",  # auto loss scaling via mixed precision
        # "QualityAwareRoutine",
        # "AdaptiveGradClipRoutine",
        # "ContextAwareNoiseRoutine",
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
        "mixedprecision",  # ensure GradScaler handles loss scaling
        "autoplugin_logger",
        "*",  # shorthand for all other plugins
    ]
    wplugins = expand_wplugins(wplugins)
    if "synthetictrainer" in wplugins:
        wplugins.remove("synthetictrainer")
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
        "batch_size": batch_size,
        "aggressive_starting_neuroplasticity": True,
        "add_min_new_neurons_per_step": int(min_new_neurons),
        "aggressive_phase_steps": 10,
    }
    def _run_kuzu_explorer(db_file: str, port: int = 8000) -> str:
        """Launch Kuzu Explorer in a background thread without Docker.

        Returns the local URL if the ``kuzu_explorer`` binary is available.
        """

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
        if PREP_TIMES:
            latency = time.perf_counter() - PREP_TIMES.pop(0)
            print(f"prep-to-train latency: {latency:.4f}s")
        payload = (left.get("prompt"), left.get("image"))
        enc = codec.encode(payload)
        try:
            idx = br.available_indices()[0]
        except Exception:
            idx = (0,) * int(getattr(br, "n", 1))
        if idx in getattr(br, "neurons", {}):
            n = br.neurons[idx]
        else:
            existing = next(iter(br.neurons)) if getattr(br, "neurons", None) else None
            n = br.add_neuron(
                idx,
                tensor=0.0,
                type_name="autoneuron",
                connect_to=existing,
            )
        n.receive(enc)
        return n
    
    print("Starting training loop...")
    use_async = os.environ.get("MARBLE_ASYNC_PIPELINE", "1").strip() not in ("0", "false", "False")
    for _ in range(int(epochs)):
        if use_async:
            pair_q: Queue = Queue(maxsize=4)

            def _pair_iter():
                while True:
                    item = pair_q.get()
                    if item is None:
                        break
                    yield item

            res_holder: dict[str, Any] = {}

            def _train():
                res_holder["res"] = run_training_with_datapairs(
                    brain,
                    _pair_iter(),
                    codec,
                    steps_per_pair=20,
                    auto_max_steps_every=20,
                    lr=1e-3,
                    loss=quality_loss,
                    wanderer_type=",".join(wplugins),
                    train_type="warmup_decay,curriculum,qualityaware",
                    neuro_config=neuro_cfg,
                    selfattention=sa,
                    streaming=True,
                    batch_size=batch_size,
                    left_to_start=_start_neuron,
                )

            t = threading.Thread(target=_train)
            t.start()

            start_time = time.perf_counter()
            for dp in _sample_pairs(ds, max_pairs=max_pairs):
                pair_q.put(dp)
            pair_q.put(None)
            t.join()
            res = res_holder.get("res", {})
        else:
            start_time = time.perf_counter()
            res = run_training_with_datapairs(
                brain,
                _sample_pairs(ds, max_pairs=max_pairs),
                codec,
                steps_per_pair=20,
                auto_max_steps_every=20,
                lr=1e-3,
                loss=quality_loss,
                wanderer_type=",".join(wplugins),
                train_type="warmup_decay,curriculum,qualityaware",
                neuro_config=neuro_cfg,
                selfattention=sa,
                streaming=True,
                batch_size=batch_size,
                left_to_start=_start_neuron,
            )
        duration = time.perf_counter() - start_time
        cnt = res.get("count", 0)
        walks = len(res.get("history", [])) or 1
        print(
            f"processed datapairs: {cnt} in {duration:.2f}s ({duration/walks:.2f}s per walk)"
        )
        if cnt == 0:
            raise RuntimeError("run_training_with_datapairs returned count=0")
        clear_resources()
    if getattr(brain, "store_snapshots", False):
        try:
            snapshot_file = brain.save_snapshot()
            print(f"saved final brain snapshot to {snapshot_file}")
        except Exception as err:
            print(f"failed to save final snapshot: {err}")
    print("streamed quality training complete")
    clear_resources()


if __name__ == "__main__":
    main()
