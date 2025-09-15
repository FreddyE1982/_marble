# Comprehensive Evaluation Plan for the Marble ML System

This document lays out a modular, multi-stage evaluation suite for Marble's learning stack using publicly available, high-complexity text data. Each stage feeds artifacts (datasets, checkpoints, telemetry logs) forward so later tests compound coverage rather than re-implementing setup work. The plan emphasises:

* Reproducible dataset handling using an open corpus with transparent licensing.
* Layered validation that starts with ingestion smoke tests and ends with adversarial, failure-injection scenarios.
* Exhaustive coverage of plugin ecosystems, the resource load balancer, neuroplasticity mechanisms, and the decision controller.
* Deterministic logging so every stage can be re-run and compared against historical baselines.

---

## 0. Prerequisites and Shared Instrumentation

**Dataset selection.** Adopt the [C4 (Colossal Clean Crawled Corpus)](https://huggingface.co/datasets/allenai/c4) "en" subset streamed via Hugging Face Datasets. It supplies large-scale, diverse text with public availability and is complex enough to stress Marble's NLP stack. All dataset downloads must be cached under `data/cache/c4/` to reuse shards across stages.

**Common fixtures.**

1. **Dataset shard generator (`tests/fixtures/c4_shard.py`):** deterministic sampling of ~50k documents per shard with configurable seed and cleaning hooks. Emits metadata JSON (document count, average length) reused by downstream tests.
2. **Checkpoint registry (`tests/fixtures/checkpoint_registry.py`):** utility to tag checkpoints with stage metadata (stage id, plugin stack, dataset shard id).
3. **Telemetry harness (`tests/fixtures/telemetry.py`):** wraps the reporter to persist structured logs (`.jsonl`) for controller/neurplasticity audits and to expose helper assertions (monotonic loss, plugin activation counts, etc.).
4. **Synthetic memory pressure shim (`tests/fixtures/memory_pressure.py`):** context manager that monkey-patches Marble's resource allocator thresholds to emulate VRAM/RAM pressure and to inject artificial tensor allocations for overflow simulations.

All subsequent tests import these fixtures; this ensures consistent seeds, artifact naming, and logging across the suite.

---

## Stage 1 – Data Ingestion & Sanity Tests

**Objective:** Guarantee the streaming loader ingests C4 shards correctly and surfaces deterministic batches to the training stack.

* **Test S1.1 (`tests/test_dataset_streaming.py`):**
  * Validate that the shard generator produces deterministic doc ordering for a fixed seed and that each record passes Marble's tokenizer without truncation errors.
  * Persist the first two shard manifests; Stage 2 consumes them as canonical training/validation splits.
* **Test S1.2 (`tests/test_dataset_metrics.py`):**
  * Compute baseline corpus stats (avg tokens per doc, vocabulary coverage) and log to telemetry for future drift detection.
  * Assert metrics fall inside expected ranges (e.g., 450±50 tokens per doc).

Artifacts: `stage1_train.jsonl`, `stage1_valid.jsonl`, telemetry logs for dataset stats.

---

## Stage 2 – Baseline Training Loop Smoke Test

**Objective:** Verify the training loop, reporter integration, and baseline plugin stack run end-to-end on small shards.

* **Test S2.1 (`tests/test_baseline_training.py`):**
  * Use Stage 1 shards and run a 200-step training loop with the minimal plugin set (no neuroplasticity, default wanderer path).
  * Assertions: loss decreases by ≥10%, reporter emits throughput/cost metrics, checkpoint saved at step 200 via registry.
* **Test S2.2 (`tests/test_autoplugin_noop.py`):**
  * Activate the auto-plugin selector in observation-only mode and ensure the decision controller logs candidate rankings without enacting changes.

Artifacts: Baseline checkpoint `stage2_baseline.pt`, telemetry log capturing base controller metrics.

---

## Stage 3 – Plugin Unit & Pairwise Compatibility Tests

**Objective:** Exercise each plugin family in isolation and in curated pairs while comparing outputs against the Stage 2 baseline.

* **Test S3.x families:**
  * **Neuron plugins (`tests/test_neuron_plugins_unit.py`):** run short (50-step) training episodes enabling each neuron plugin individually; assert numerical stability (no NaNs) and bounded activation ranges.
  * **Synapse plugins (`tests/test_synapse_plugins_unit.py`):** similar pattern focusing on weight updates and gradient norms.
  * **Brain-train plugins (`tests/test_brain_train_unit.py`):** confirm learning-rate schedules and warmup/decay behaviours.
  * **Self-attention plugins (`tests/test_self_attention_unit.py`):** check context-window alignment and reporter metrics.
  * **Wanderer plugins (`tests/test_wanderer_unit.py`):** ensure path exploration stays within configured bounds.
* **Pairwise stacks (`tests/test_plugin_pairwise.py`):** iterate through hard-wired plugin pairs (e.g., advanced neuron + ultra synapse) and compare loss trajectories against Stage 2 baseline (allow ±5% difference). Persist Stage 3 checkpoints tagged with plugin IDs.

Artifacts: Stage 3 telemetry catalog of plugin metrics, compatibility matrix stored as CSV for later reference.

---

## Stage 4 – Progressive Plugin Stack Integration

**Objective:** Validate that increasingly complex plugin stacks cooperate when activated together, relying on Stage 3 compatibility insights.

* **Test S4.1 (`tests/test_plugin_triplets.py`):**
  * Construct curated triplets (neuron, synapse, neuroplasticity) known to interact; reuse Stage 3 compatibility matrix to avoid combinations flagged unstable.
  * Assertions: combined stack improves loss at least as much as best pairwise plugin; reporter logs include all plugin-specific metrics.
* **Test S4.2 (`tests/test_plugin_pipeline_snapshots.py`):**
  * Run 500-step training with rotating plugin schedules, storing checkpoints every 100 steps. Verify checkpoint deltas show expected parameter drift patterns using registry diff utilities.
* **Test S4.3 (`tests/test_plugin_failure_injection.py`):**
  * For each plugin stack, deliberately raise plugin-specific exceptions (e.g., invalid tensor shapes) to ensure graceful degradation paths fire and the decision controller records the event.

Artifacts: Multi-stack checkpoints `stage4_stack_*.pt`, failure logs consumed by Stage 7.

---

## Stage 5 – Load Balancer & Memory Stress Validation

**Objective:** Deeply verify resource allocation, including simulated VRAM/RAM overflows, using Stage 4 plugin stacks.

* **Test S5.1 (`tests/test_load_balancer_nominal.py`):**
  * Run Stage 4 stack with telemetry harness capturing per-device memory usage. Assertions: balancer keeps VRAM usage < threshold, respects disk offload policies.
* **Test S5.2 (`tests/test_load_balancer_overflow.py`):**
  * Invoke memory pressure shim to reduce VRAM threshold (e.g., 0.4) and inject synthetic tensor spikes. Expect balancer to offload to RAM/disk without crash, and telemetry should log overflow events.
* **Test S5.3 (`tests/test_load_balancer_recovery.py`):**
  * After overflow, restore normal thresholds mid-run and verify tensors migrate back, training resumes, and losses recover within 2× baseline steps.
* **Test S5.4 (`tests/test_allocator_edge_cases.py`):**
  * Validate behaviour when disk cache is near capacity using fake filesystem fill; ensure allocator halts gracefully and the decision controller receives high-cost signals.

Artifacts: Memory telemetry snapshots, allocator decision logs, simulated overflow scenarios for Stage 7 fault replay.

---

## Stage 6 – Neuroplasticity Mechanism Verification

**Objective:** Exhaustively test all neuroplasticity mechanisms, leveraging Stage 4 checkpoints as warm starts.

* **Test S6.1 (`tests/test_neuroplasticity_activation.py`):**
  * Activate each neuroplasticity plugin individually from Stage 4 checkpoint. Assertions: plugin registers required tensors, modifies graph structures as expected (checked via graph diff).
* **Test S6.2 (`tests/test_neuroplasticity_persistence.py`):**
  * Run multi-epoch sessions verifying neuron/synapse changes persist after checkpoint reload (load Stage 6 checkpoint, resume training, confirm previous graph edits exist).
* **Test S6.3 (`tests/test_neuroplasticity_metrics.py`):**
  * Use telemetry harness to ensure metrics such as plasticity rate, rewiring counts, and stability scores are reported and stay within safe ranges.
* **Test S6.4 (`tests/test_neuroplasticity_rollback.py`):**
  * Inject failures mid-update to confirm rollback/undo logic prevents graph corruption and the decision controller is notified with correct penalty signals.

Artifacts: Graph diff reports, plasticity telemetry logs, resilient checkpoints for Stage 7.

---

## Stage 7 – Decision Controller Stress & Policy Validation

**Objective:** Validate policy learning, constraint handling, and reaction to fault signals using artifacts from earlier stages.

* **Test S7.1 (`tests/test_decision_controller_learning.py`):**
  * Replay Stage 4/6 telemetry to prime controller policies, then run live training verifying constraint multipliers converge and plugin activation frequencies match learned patterns.
* **Test S7.2 (`tests/test_decision_controller_constraints.py`):**
  * Configure non-trivial linear constraints (A/b matrix) and ensure controller respects them while still improving loss vs Stage 2 baseline.
* **Test S7.3 (`tests/test_decision_controller_fault_response.py`):**
  * Feed in failure logs from Stage 4 & Stage 5 to verify controller escalates penalties, blacklists problematic plugin combos temporarily, and emits dwell-state changes in telemetry.
* **Test S7.4 (`tests/test_decision_controller_policy_modes.py`):**
  * Compare `policy-gradient` vs `bayesian` modes using identical seeds; assert decision cadence and throughput remain within tolerance, with metrics persisted for Stage 8 comparisons.

Artifacts: Controller policy snapshots, constraint compliance reports, comparative telemetry for policy modes.

---

## Stage 8 – End-to-End Regression, Soak, and Reporting

**Objective:** Consolidate coverage with long-horizon regression runs that stitch together previous artifacts and validate overall stability.

* **Test S8.1 (`tests/test_regression_soak.py`):**
  * Execute a 5k-step training session cycling through Stage 4 plugin schedules, Stage 6 neuroplasticity, and Stage 7 controller policies. Assertions: no unhandled exceptions, memory usage stays bounded, end-to-end loss improves ≥25% over Stage 2 baseline.
* **Test S8.2 (`tests/test_artifact_integrity.py`):**
  * Validate that every artifact (checkpoints, telemetry, compatibility matrices) is present, checksummed, and re-loadable. Ensures reproducibility for downstream analytics.
* **Test S8.3 (`tests/test_reporting_pipeline.py`):**
  * Assemble cumulative metrics into a markdown/HTML report summarising plugin effectiveness, controller behaviour, and load balancer stability. Confirm report generation completes without missing data.
* **Test S8.4 (`tests/test_regression_alerts.py`):**
  * Compare current soak metrics against golden baselines; raise alerts if regression thresholds are exceeded (≥5% loss degradation, memory overflow counts >0, controller constraint violations).

Artifacts: Final regression report, aggregated telemetry database, pass/fail summary for CI gatekeeping.

---

## Stage 9 – Optional Extensions

* **Adversarial dataset splits:** Introduce specialized C4 slices (legal, medical) to test domain adaptation. Reuse Stage 1 fixtures with new filters.
* **Latency-focused microbenchmarks:** Run targeted performance tests on plugin kernels using PyTorch profiler; integrate results into Stage 5 reporting.
* **Explainability hooks:** Add tests that trace decision controller rationales and plugin contributions for auditing.

---

## Execution Order & Automation Notes

1. Implement fixtures (Stage 0) before authoring any tests.
2. Author and run tests sequentially from Stage 1 to Stage 8. Each stage has explicit prerequisites; do not skip ahead.
3. Store large artifacts under `artifacts/stage_<n>/` with README files describing contents and provenance.
4. Integrate the suite into CI by tagging stages: `pytest -m stage1`, ..., `pytest -m stage8`. Later stages should depend on previous markers completing successfully.
5. Maintain a `tests/README.md` summarising how to replay each stage locally, including estimated runtime and hardware requirements.

Following this plan yields a continuously expanding evaluation harness capable of validating Marble's ML system under realistic, stress-tested scenarios while ensuring every subsystem—from data ingestion to decision control—remains accountable.
