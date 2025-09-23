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

## Step 5 – Graph-of-thought introspection for Wanderer trajectories

5.1 **Capture high-resolution walk state.** Extend the existing reporter hooks
so every Wanderer step persists its predecessor/successor edges, decision logits,
and plugin context into a graph-structured buffer. Export this buffer into a
new `reporter.graph` stream that mirrors the Graph-of-Thought (GoT) notion of
"thought" vertices with dependency edges, enabling downstream tooling to reason
about alternative branches.[^5]

5.2 **Author graph-native analyzers.** Build a diagnostics module that runs
GoT-inspired graph algorithms—cycle detection, feedback edge trimming, and
subgraph condensation—to flag loops, redundant plugin sequences, and high-impact
branches. Surface these metrics in the Wanderer dashboards and throttle
low-value branches via `decision_controller` policies.

5.3 **Interactive replay + testing.** Create CLI tooling that can replay a
stored walk graph, allowing developers to prune/merge nodes and run counterfactual
routes. Augment existing Wanderer regression tests with graph snapshots to
ensure coverage for branching, merging, and pruning behaviours.

## Step 6 – Carbon-aware scheduling and telemetry

6.1 **Instrument energy estimators.** Embed hooks similar to the Carbontracker
workflow to log estimated energy draw per plugin family, factoring in device
type, run duration, and datacenter carbon intensity. Record the metrics alongside
loss/latency so optimization loops can jointly minimize emissions and cost.[^6]

6.2 **Plan adaptive scheduling.** Extend `config.yaml` with a `sustainability`
section (e.g., `target_co2eq_per_walk`, `prefer_offpeak: true`) and update the
execution scheduler to shift intensive tasks toward greener time windows or
lower-carbon devices when thresholds are exceeded. Provide CPU-only fallbacks
for environments that cannot meet the emission budget.

6.3 **Publish sustainability reports.** Enhance reporter exports to summarize
per-run emissions, energy, and cost estimates. Pipe the aggregates into
documentation (e.g., markdown tables in `docs/`) and include regression tests
that validate the telemetry fields populate even when wanders are short.

## Step 7 – Multi-agent automation for evaluation and remediation

7.1 **Design agent protocols.** Introduce an orchestration layer that spins up
lightweight analysis agents to review reporter logs, config diffs, and test
artifacts. Base the interaction loops on AutoGen-style role hand-offs so agents
can debate anomalies and agree on remediation plans before code changes land.[^7]

7.2 **Automate regression triage.** Attach the agent layer to CI so a failure in
`tests/` triggers a conversation between diagnostics, reproduction, and fix-scout
agents. Feed their summarized findings back into Git metadata and `REPORTER`
annotations to streamline human review.

7.3 **Agent-in-the-loop safeguards.** Provide guardrails that cap agent retries,
require human acknowledgement for destructive fixes, and log all agent decisions
for auditability. Extend tests to simulate stuck conversations and verify the
fallback path hands control back to maintainers.

## Step 8 – Temporal transformation reasoning benchmarks

8.1 **Curate multimodal task suites.** Adapt datasets inspired by Visual
Transformation Telling so Wanderer walkthroughs can train on ordered state
pairs—images, sensor traces, or serialized graph states—and produce transformation
descriptions. Store dataset manifests under `examples/` with streaming loaders
to stay consistent with Step 3's ingestion goals.[^8]

8.2 **Plugin adapters for transformations.** Implement neuron and synapse
plugins that consume paired states and learn to narrate or predict the underlying
transformation. Ensure `expose_learnable_params` covers temporal attention spans,
state-diff thresholds, and generative decoder knobs.

8.3 **Evaluation harness.** Add reporter metrics that score narrative fidelity,
temporal consistency, and downstream task benefit (e.g., better Wanderer path
selection). Expand tests to replay curated sequences and assert the new plugins
stabilize when transformations repeat or branch.

## Step 9 – Selective state-space expertise for long contexts

9.1 **Survey existing sequence bottlenecks.** Profile Wanderer traces,
self-attention plugins, and reporter graphs to pinpoint where attention kernels
or recurrent helpers fall over on million-step walks. Capture latency,
activation size, and gradient stability to decide which modules should migrate
to selective state spaces.

9.2 **Implement a Mamba-inspired plugin suite.** Following the selective state
space model (SSM) architecture from the Mamba paper—input-conditioned SSM
parameters, hardware-aware recurrent kernels, and linear-time inference—craft
new neuron and self-attention plugins that expose gating temperature, state
mixing, and carry/forget spans via `expose_learnable_params`.[^9] Ensure the
plugins can swap between recurrent and convolutional modes so they mesh with
existing graph execution policies.

9.3 **Optimize routing and validation.** Integrate the SSM plugins with the MoE
router from Step 1 so long-context experts activate only when sequence length
justifies the cost. Extend regression tests to compare throughput and loss for
Transformer vs. SSM branches across synthetic and real million-token corpora.
Document trade-offs (e.g., state caching, gradient clipping) in
`ARCHITECTURE.md`.

## Step 10 – Virtualized activation memory with PagedAttention mechanics

10.1 **Map allocator pressure points.** Instrument the resource allocator's
tensor tracker to log fragmentation, spill frequency, and KV-cache duplication
whenever Wanderer or snapshot replay exceeds GPU memory. Identify patterns where
attention caches persist beyond their useful window.

10.2 **Adopt PagedAttention-style paging.** Borrow the virtual-memory
abstractions from vLLM's PagedAttention work to chunk KV caches into pageable
blocks, enabling near-zero waste reuse across batched wanderers.[^10] Implement
a `paged_cache` mode inside the allocator with eviction policies, CPU/disk
spillover, and hooks for sparse retrieval when revisiting prior branches.

10.3 **Stress testing and guardrails.** Simulate adversarial batch mixes (long
vs. short sequences, MoE-enabled vs. disabled) and ensure the pager avoids
thrashing. Extend tests to assert pager accounting never starves critical
plugins, and expose new telemetry (page faults, cache hits) through `REPORTER`
dashboards.

## Step 11 – Speculative multi-branch decoding for Wanderer inference

11.1 **Draft model scaffolding.** Analyze existing plugin stacks to identify a
lightweight "draft" configuration that can cheaply extend walk candidates. Cache
its weights using the allocator so speculative branches spin up without full
reinitialization.

11.2 **Speculative acceptance logic.** Port the speculative sampling acceptance
rules—parallel draft continuations with rejection sampling corrections—to the
Wanderer evaluation loop so multiple path continuations can be scored in one
pass while preserving target distribution fidelity.[^11]

11.3 **Quality/perf validation.** Benchmark speculative walk replay against
baseline decoding on diverse tasks, logging latency gains, rejection rates, and
reward deltas. Update plugin stacking tests to include speculative mode toggles
and ensure determinism when the feature is disabled.

## Step 12 – Mechanistic interpretability pipelines with sparse autoencoders

12.1 **Curate activation corpora.** Extend the reporter graph dumpers so neuron
and synapse activations from representative walks stream into disk-backed
datasets sized for sparse autoencoder training. Include metadata (plugin type,
decision context) to support feature attribution later.

12.2 **Train feature dictionaries.** Build an interpretability toolkit that
trains sparse autoencoders on the captured activations, mirroring the dictionary
learning recipe from Transformer Circuits' monosemantic features work—scalable
expansion factors, L1-regularized activations, and feature browsers.[^12]
Automate hyperparameter sweeps and store learned feature banks as artifacts in
`docs/`.

12.3 **Integrate feature-guided interventions.** Add debugging hooks that map
autoencoder features back to live Wanderer runs, enabling forced activation,
suppression, or routing hints. Update reporter dashboards with feature-level
metrics (activation sparsity, steering impact) and document workflows for
engineers to diagnose plugin misbehaviour.

## Step 13 – Self-refining Wanderer feedback loops

13.1 **Capture iterative feedback traces.** Extend the Wanderer reporter so
each decision stores the generated action, reviewer critiques, and revision
scores in `reporter.graph` alongside existing trajectory data. Mirror the
Self-Refine pattern by letting the same plugin author both the initial proposal
and textual feedback, persisting the deltas so later runs can mine reusable
corrections.[^13]

13.2 **Author reflection-aware controllers.** Create a light-weight
`reflection_buffer` utility that threads Reflexion-style critiques into the
decision controller, exposing knobs in `config.yaml` for memory horizon,
reflection weighting, and decay.[^14] When the router replays a familiar state,
inject prior reflections as auxiliary logits so the agent learns from past
mistakes without retraining.

13.3 **Benchmark iterative refinement.** Build regression suites that compare
baseline vs. reflection-enhanced walks on coding, planning, and evaluation
tasks. Assert that iterative refinement converges faster, reduces repeated
failures, and remains deterministic when the feature is disabled. Surface the
latency overhead and reflection hit-rate via `REPORTER` summaries.

## Step 14 – Hardened safety guardrails against adversarial prompts

14.1 **Map current vulnerability surface.** Inventory every plugin that consumes
external prompts or tool responses, capturing how user-controlled text flows
into execution helpers and config updates. Replay known red-team suffixes to
establish a baseline jailbreak rate using the existing evaluator stack.[^15]

14.2 **Introduce guardrail middleware.** Implement a security filter that scans
incoming prompts for adversarial patterns, gradient-hacked suffixes, and policy
violations before routing them into Wanderer decisions. Wire the middleware into
the `decision_controller` so MoE routing, budget throttling, and unlearning
hooks can quarantine suspicious requests instead of executing them blindly.

14.3 **Continuous red-team automation.** Extend the multi-agent evaluation
layer (Step 7) with recurring adversarial probes and log jailbreak verdicts
alongside mitigation actions. Add docs and config toggles for quarantine modes,
alert thresholds, and opt-in telemetry uploads so ops teams can audit response
quality.

## Step 15 – Lifelong skill libraries for Wanderer autonomy

15.1 **Survey reusable behaviour fragments.** Mine existing Wanderer traces,
snapshot macros, and plugin callbacks to detect repeatable skill graphs (e.g.,
resource harvesting, topology healing). Tag each with context metadata and
store the manifests under `docs/skills/` for reproducibility, inspired by
Voyager's curriculum discovery loop.[^16]

15.2 **Build a composable skill repository.** Implement a `skill_library`
module that packages skill graphs, required plugins, and safety prerequisites.
Expose APIs so routing policies can request, compose, or adapt skills at
runtime, persisting upgrades back into the repository. Provide config switches
for auto-saving, version pinning, and review workflows.

15.3 **Evaluate transfer and generalisation.** Script benchmarks that boot new
world seeds or synthetic graph challenges, measuring how fast Wanderer solves
them with vs. without the stored skills. Track skill activation frequency,
coverage, and catastrophic forgetting indicators in the reporter dashboards.

## Step 16 – Tool-native action planning for plugin stacks

16.1 **Audit tool invocation touchpoints.** Trace where the codebase already
invokes calculators, search helpers, or dataset loaders so we know which
actions can be migrated to explicit tool APIs. Document the findings alongside
current telemetry gaps.

16.2 **Implement Toolformer-inspired routing.** Train a lightweight selector
that predicts when to call specific helper APIs, the arguments to use, and how
to merge the responses back into plugin state, following the self-supervised
Toolformer recipe.[^17] Embed selector outputs in the router logits so the MoE
stack learns when external tools should override standard inference.

16.3 **Testing and observability.** Expand regression coverage to include
tool-augmented walks, verifying that API failures degrade gracefully and that
selector confidence correlates with downstream reward. Surface per-tool
latency, success rates, and fallback triggers through the existing reporter
channels and document the operational runbooks.


## Step 17 – Knowledge-graph governance for plugin interplay

17.1 **Synthesize cross-run graphs.** Extend the `reporter.graph` stream from Step 5 so every walk exports typed vertices for plugins, tensors, and decision states plus edges for data flow, success/failure, and resource consumption. Store the rollups under `docs/graphs/` with schema metadata so downstream tooling can diff interaction topologies between runs and flag missing dependencies.

17.2 **Inject graph reasoning loops.** Implement a governance helper that ingests the interaction graph and executes Graph Chain-of-Thought style reasoning passes to propose routing changes, plugin retirements, or redundancy merges.[^18] Surface its suggestions through `REPORTER` annotations and optional PR comments so maintainers can review auto-generated governance hints.

17.3 **Tighten validation.** Add targeted tests that construct miniature plugin graphs, run the reasoning helper, and assert the recommendations stay within configured policy bounds (e.g., never disable required safety plugins). Provide config toggles (`graph_governance: {enabled: false, policy: "advisory"}`) and document how they throttle automatic enforcement.

## Step 18 – Autoregressive remediation benchmarks with SWE-style issues

18.1 **Mirror real bug corpora.** Import a filtered slice of SWE-bench-style issues into `examples/benchmarks/` with scripts that replay Marble plugin stacks against recorded failing tests.[^19] Ensure dataset loaders honor the streaming ingestion rules from Step 3 so large corpora stay memory-light.

18.2 **Wire automated fix scouts.** Extend the Step 7 agent orchestration so a regression triggers a SWE-bench harness: reproduce, localize via reporter traces, and draft candidate patches. Feed accepted patches through the existing snapshot manager so we can simulate post-fix walks before merging.

18.3 **Measure remediation quality.** Emit metrics for patch success rate, turnaround time, and guardrail interventions via `REPORTER`. Add regression tests that inject synthetic failures, confirm the harness proposes viable diffs, and gracefully defers to humans when confidence falls below a configurable threshold.

## Step 19 – Parameter-efficient personalization via quantized adapters

19.1 **Audit adapter hotspots.** Profile Wanderer and synapse plugins to identify layers where user-provided data most influences performance. Tag them as adapter candidates and expose toggles in `config.yaml` (`personalization: {qlora_ranks: ..., target_plugins: [...]}`).

19.2 **Implement QLoRA-style adapters.** Introduce low-rank adapter modules that piggyback on quantized weights, following QLoRA’s double-quantization and paged optimizer tricks to keep VRAM use low while supporting backprop.[^20] Integrate them with the resource allocator so adapters spill gracefully between CPU/GPU.

19.3 **Validate personalization flows.** Extend tests to simulate per-user fine-tuning sessions, verifying that adapters serialize alongside snapshots, stay numerically stable after load, and can be rolled back when unlearning hooks (Step 4) activate.

## Step 20 – Deliberate tree search for Wanderer planning

20.1 **Capture branch heuristics.** Instrument Wanderer to log heuristic scores (reward estimates, safety costs, novelty) for each decision so they can seed a Tree-of-Thoughts style planner.[^21] Persist the heuristics in reporter traces alongside existing logits.

20.2 **Embed tree search controller.** Add an optional planner that expands multiple candidate futures per decision, prunes low-value branches, and hands the best plan back to the decision controller. Keep the planner modular so MoE routing (Step 1) can swap experts per branch.

20.3 **Stress-test determinism.** Create regression scenarios where the planner runs with fixed seeds, asserting branch selection remains deterministic unless stochastic exploration is explicitly enabled. Benchmark latency overhead and expose config knobs for tree width/depth.

## Step 21 – Hardened data integrity and poisoning defenses

21.1 **Baseline dataset hygiene.** Attach a preprocessing audit that scans incoming datasets for distribution drift, label irregularities, and adversarial triggers highlighted in data poisoning surveys.[^22] Log findings to `docs/data_audits/` with reproducible scripts.

21.2 **Integrate defensive filters.** Augment streaming loaders with optional filtering hooks (e.g., outlier scorers, gradient sanitizers) that quarantine suspicious batches before they reach Wanderer training loops. Provide overrides so trusted datasets skip heavy checks.

21.3 **Regression coverage for attacks.** Add tests that simulate simple poisoning attacks (label flips, trigger patterns) and verify the filters flag them while preserving clean data throughput. Ensure reporter summaries distinguish between benign rejections and critical incidents so ops teams can escalate appropriately.

---

### Cross-cutting deliverables

- Update `config.yaml`, `yaml-manual.txt`, and `TUTORIAL.md` whenever new knobs
or workflows appear.
- Extend automated tests (per `AGENTS.md` rules) to cover every new plugin,
compression path, or streaming helper introduced in the steps above.
- Ensure reporter metrics surface the before/after impact of each improvement so
future runs can compare memory, throughput, and accuracy at a glance.
- Publish sustainability and agent-audit appendices when Steps 6–7 add new
telemetry, keeping the documentation in sync with emitted metrics.
- Bundle new long-context, paging, speculative decoding, interpretability, and
reflection/guardrail/tooling rollouts with reproducible notebooks or CLI
recipes so teams can rerun the measurements backing Steps 9–16.
- Version governance graphs, SWE-bench corpora, personalization adapter specs, tree-search configs, and poisoning audit artifacts from Steps 17–21 under `docs/` so upgrades stay reproducible.

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
[^5]: Besta et al., 2024 — *Graph of Thoughts: Solving Elaborate Problems with
Large Language Models*. https://arxiv.org/abs/2308.09687
[^6]: Anthony et al., 2020 — *Carbontracker: Tracking and Predicting the Carbon
Footprint of Training Deep Learning Models*. https://arxiv.org/abs/2007.03051
[^7]: Wu et al., 2023 — *AutoGen: Enabling Next-Gen LLM Applications via
Multi-Agent Conversation*. https://arxiv.org/abs/2308.08155
[^8]: Cui et al., 2024 — *Visual Transformation Telling*. https://arxiv.org/abs/2305.01928
[^9]: Gu & Dao, 2024 — *Mamba: Linear-Time Sequence Modeling with Selective State
Spaces*. https://arxiv.org/abs/2312.00752
[^10]: Kwon et al., 2023 — *Efficient Memory Management for Large Language Model
Serving with PagedAttention*. https://arxiv.org/abs/2309.06180
[^11]: Chen et al., 2023 — *Accelerating Large Language Model Decoding with
Speculative Sampling*. https://arxiv.org/abs/2302.01318
[^12]: Transformer Circuits, 2023 — *Towards Monosemanticity: Decomposing
Language Models With Sparse Autoencoders*.
https://transformer-circuits.pub/2023/monosemantic-features/index.html
[^13]: Madaan et al., 2023 — *Self-Refine: Iterative Refinement with
Self-Feedback*. https://arxiv.org/abs/2303.17651
[^14]: Shinn et al., 2023 — *Reflexion: Language Agents with Verbal Reinforcement
Learning*. https://arxiv.org/abs/2303.11366
[^15]: Zou et al., 2023 — *Universal and Transferable Adversarial Attacks on
Aligned Language Models*. https://arxiv.org/abs/2307.15043
[^16]: Wang et al., 2023 — *Voyager: An Open-Ended Embodied Agent with Large
Language Models*. https://arxiv.org/abs/2305.16291
[^17]: Schick et al., 2023 — *Toolformer: Language Models Can Teach Themselves
to Use Tools*. https://arxiv.org/abs/2302.04761
[^18]: Chen et al., 2024 — *Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs*. https://arxiv.org/abs/2404.07103
[^19]: Jimenez et al., 2023 — *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* https://arxiv.org/abs/2310.06770
[^20]: Dettmers et al., 2023 — *QLoRA: Efficient Finetuning of Quantized LLMs*. https://arxiv.org/abs/2305.14314
[^21]: Yao et al., 2023 — *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. https://arxiv.org/abs/2305.10601
[^22]: Goldblum et al., 2022 — *Machine Learning Security against Data Poisoning: Are We There Yet?* https://arxiv.org/abs/2204.05986
