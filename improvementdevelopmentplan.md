# Improvement Development Plan

This plan captures the next major upgrades for the Marble core, grounded in the
current code base and aligned with recent research on modular AI systems,
quantization, dataset streaming, and continual learning.

## Step 1 – Architect Mixture-of-Experts (MoE) routing for plugin stacks

1.1 **Map existing plugin roles and telemetry.** Catalogue every neuron,
synapse, wanderer, brain-train, self-attention, and neuroplasticity plugin
already registered in `marble/marblemain.py`, and tag each one with its
functional niche (e.g., feature extractors, path planners, meta-optimizers) by
parsing the registration tables and ARCHITECTURE notes. Extend the `REPORTER`
metrics emitted during walks to capture per-plugin activation frequency and
latency so we have the data needed for routing decisions.

1.2 **Design an MoE-style router plugin.** Build a new meta-plugin that learns
to dispatch incoming brain signals to specialized plugin "experts" in parallel,
mirroring modern MoE transformer gating (sparse expert selection, load
balancing, capacity factors). The Hugging Face analysis of Mixtral-style MoE
systems outlines the compute savings and router stability tricks we should
mirror—particularly balanced token routing, router regularization, and capacity
planning for inference and training efficiency.[^1]

1.3 **Integrate with the decision controller.** Feed router logits back through
the `decision_controller` section of `config.yaml` so phase budgets, cadence,
and constraint multipliers can throttle expert activations. Add a config flag to
enable/disable MoE routing globally, defaulting to off for backward
compatibility, and document the new parameters. Update targeted tests for plugin
stacking to ensure the router honors activation budgets and produces stable
losses even when sparse experts are swapped mid-walk.

## Step 2 – Introduce learnable sub-bit parameter compression

2.1 **Profile parameter hotspots.** Use `count_numeric_parameters.py` and
snapshot inspection to identify the neuron, synapse, and wanderer parameters
with the highest aggregate footprint during intensive walks. Capture VRAM/CPU
pressure via the resource allocator logs so we know which tensors benefit most
from compression.

2.2 **Implement tiled-bit parameter pools.** Inspired by recent Tiled Bit
Networks work on sub-bit compression, represent frequently reused weight blocks
(neuron types, synapse transforms, wanderer optimizers) as shared binary tiles
that are reshaped on-demand.[^2] The allocator should cache one tile per layer or
plugin and expand it during compute, delivering up to 8× parameter reduction
while keeping autograd intact. Guard this path behind a `resource_allocator`
config toggle (`tile_quantization: {enabled: false, tile_bits: 1, reuse_scope:
"plugin"}`) and ensure CPU fallbacks exist when torch is unavailable.

2.3 **End-to-end validation.** Extend tests covering reporter summaries and
snapshot precision to confirm that loss curves stay within tolerance when tiled
weights are active. Add benchmarks that compare walk latency and memory
consumption with and without tiling to demonstrate the savings promised by the
compression paper. Document operational caveats (e.g., tile reuse increasing
correlation between neurons) in `ARCHITECTURE.md`.

## Step 3 – Expand streaming data ingestion and sharding

3.1 **Audit existing dataset helpers.** Review `run_training_with_datapairs`,
`run_wanderer_epochs_with_datapairs`, and example scripts to confirm how
datasets enter the system today (local files vs. huggingface streaming helpers).
Identify any assumptions about random access or dataset materialization that
conflict with streaming iterators.

3.2 **Adopt IterableDataset-first workflows.** Hugging Face's streaming guide
recommends `IterableDataset` wrappers for massive corpora, with column
projection, sharding, and conversion utilities tailored for sequential jobs.[^3]
Refactor our helpers so `load_hf_streaming_dataset(..., streaming=True)` becomes
the default, expose sharding controls via `config.yaml`, and ensure batch-aware
plugins (like `batchtrainer`) set matching batch sizes in both neuro configs and
helper arguments.

3.3 **Resilience and observability.** Add retry/backoff logic around streaming
iterators, emit reporter metrics for stream lag, and provide CLI knobs to resume
from the last successfully processed shard. Update docs with explicit examples
showing how to stream terabyte-scale datasets without exhausting disk, following
the recommendations from the Hugging Face documentation. Extend dataset-focused
tests to cover streaming-mode iterators and column projection.

## Step 4 – Build continual learning + unlearning workflows

4.1 **Baseline current continual behavior.** Trace how the Wanderer and
brain-train plugins handle sequential tasks today, especially snapshot reuse and
metric logging. Capture any catastrophic forgetting patterns by running small
sequential experiments and writing the deltas to the reporter JSONL streams.

4.2 **Hypernetwork-driven task adapters.** Leverage the UnCLe framework's idea
of using hypernetworks that emit task-specific weights and support parameter
"unlearning" without original data.[^4] Implement a Wanderer auxiliary plugin
that stores compact task embeddings, regenerates adapter weights on the fly, and
projects unwanted tasks back to noise vectors when an unlearning request comes
in.

4.3 **Retention and privacy safeguards.** Wire the unlearning hooks into the
snapshot manager so removing a task purges its serialized state while preserving
others. Add config switches for "data-free unlearning" policies, log each
unlearning event via `REPORTER`, and craft tests that learn, unlearn, then
re-learn tasks to verify retention, relapse prevention, and privacy guarantees.

---

### Cross-cutting deliverables

- Update `config.yaml`, `yaml-manual.txt`, and `TUTORIAL.md` whenever new knobs
or workflows appear.
- Extend automated tests (per `AGENTS.md` rules) to cover every new plugin,
compression path, or streaming helper introduced in the steps above.
- Ensure reporter metrics surface the before/after impact of each improvement so
future runs can compare memory, throughput, and accuracy at a glance.

### Research references

[^1]: Hugging Face Blog — *Mixture of Experts Explained*. Highlights routing,
load balancing, and capacity tuning strategies for sparse MoE transformers.
https://huggingface.co/blog/moe
[^2]: Bhattacharya et al., 2024 — *Tiled Bit Networks: Sub-bit Compression for
Binary Neural Networks*. https://arxiv.org/abs/2407.12075
[^3]: Hugging Face Documentation — *Dataset streaming lets you work with a
dataset without downloading it.* https://huggingface.co/docs/datasets/stream
[^4]: Adhikari et al., 2025 — *An Unlearning Framework for Continual Learning
(UnCLe).* https://arxiv.org/abs/2509.17530
