# Evaluation Plan for Marble ML System Using Public Text Corpora

This plan outlines a layered evaluation strategy for our learning system using a publicly available, high-complexity text dataset. Each phase introduces a new module of tests that depends on artifacts from earlier phases so that we progressively validate data handling, modeling, and training behaviours. The plan recommends using the **The Pile** (EleutherAI, CC-BY 4.0) English subset via Hugging Face Datasets because it is legally redistributable, large (~825 GB), heterogeneous, and well-documented. For faster iterations we rely on stratified shards published alongside the full dataset and on Hugging Face's streaming API to avoid massive local storage requirements.

## Phase 0 – Dataset Handshake & Governance
*Goal:* Confirm legal, reproducible access to the dataset before any code touches it.

1. **Test: `tests/test_dataset_handshake.py::test_pile_metadata_contract`**
   - Assert that `load_hf_streaming_dataset("the_pile", subset="all", streaming=True)` returns splits with documented features.
   - Validate dataset license text, citation string, and version hash, storing them in `docs/datasets/the_pile.md` for traceability.
   - Capture dataset card metadata (size estimates, languages) and cache for later tests.
2. **Test: `...::test_restricted_domains_pruned`**
   - Verify filtering logic excludes known problematic domains (e.g., duplicates flagged by EleutherAI) by counting removed rows.
   - Depends on handshake metadata fixture.

## Phase 1 – Data Ingestion Smoke Tests
*Goal:* Ensure our loaders can stream and shard the dataset deterministically.

1. **Test: `tests/test_data_streaming.py::test_stream_batch_shapes`**
   - Consume 1,000 streamed samples, ensuring batched tensors respect configured `sequence_length` and `batch_size` in `config.yaml`.
   - Use the metadata fixture from Phase 0 for expected column names.
2. **Test: `...::test_reseeded_shard_reproducibility`**
   - Re-run the loader with a fixed seed and shard id; assert exact token sequences match cached digests from the first run.
   - Builds on streaming pipeline validated above.
3. **Test: `...::test_loader_throughput_logging`**
   - Confirm reporter logs tokens-per-second and I/O latency metrics, setting thresholds to catch regressions.

## Phase 2 – Text Normalisation & Tokenisation
*Goal:* Validate preprocessing transforms before the model ever sees the data.

1. **Test: `tests/test_preprocessing.py::test_unicode_normalisation_pipeline`**
   - Feed raw samples through our normalisation chain; assert canonical NFC form and removal of banned control characters.
   - Relies on streaming fixtures from Phase 1.
2. **Test: `...::test_bpe_tokeniser_coverage`**
   - Ensure the tokenizer vocabulary covers ≥99.5% of tokens in a 100k-sample slice; report OOV counts to the reporter.
3. **Test: `...::test_sequence_packing_consistency`**
   - Compare packed sequences across CPU/GPU devices to catch device-specific divergence.

## Phase 3 – Model Wiring Unit Tests
*Goal:* Prove model components can ingest token batches and produce stable outputs.

1. **Test: `tests/test_model_blocks.py::test_transformer_block_forward`**
   - Run a forward pass on a mini-batch from Phase 2, assert tensor shapes, dtype alignment, and absence of NaNs.
2. **Test: `...::test_attention_mask_contract`**
   - Validate causal masks built from Phase 2 sequences match expected lower-triangular patterns.
3. **Test: `...::test_parameter_registration`**
   - Ensure every learnable parameter is exposed via `expose_learnable_params`, comparing totals against architecture config.

## Phase 4 – End-to-End Gradient Smoke Test
*Goal:* Confirm gradients propagate and optimisation steps execute without divergence.

1. **Test: `tests/test_training_smoke.py::test_single_batch_backprop`**
   - Run one optimisation step on a tiny shard (e.g., 32 sequences). Assert gradients are finite and optimiser state updates.
2. **Test: `...::test_loss_reporting_contract`**
   - Validate loss metrics (`train_loss`, `tokens_processed`, `wall_time`) hit the reporter and respect naming conventions.
   - Uses Phase 3 fixtures for the model and Phase 2 data batches.

## Phase 5 – Mini-Epoch Functional Test
*Goal:* Evaluate learning dynamics on a representative subset.

1. **Test: `tests/test_training_miniepoch.py::test_loss_curve_downward_trend`**
   - Train for 200 steps on a curated 5 GB shard; assert rolling average loss decreases by ≥X% relative to initial loss.
   - Compare checkpoint weights to initial weights to confirm updates exceed noise floor.
2. **Test: `...::test_checkpoint_replay_equivalence`**
   - Reload the checkpoint and replay final 10 batches; verify loss trajectory matches cached metrics within tolerance.

## Phase 6 – Full-Corpus Evaluation & Metrics Audit
*Goal:* Run a controlled evaluation sweep that mirrors production training goals.

1. **Test: `tests/test_eval_perplexity.py::test_validation_perplexity_threshold`**
   - Evaluate on The Pile validation mix; assert perplexity stays below the regression baseline gathered from prior release.
2. **Test: `tests/test_eval_downstream_generalisation.py::test_blimp_zeroshot_transfer`**
   - Fine-tune a lightweight classifier on the BLiMP suite (public text-based syntactic tasks) using embeddings from the main model; ensure average accuracy beats baseline by a margin.
   - Builds on checkpoints from Phase 5.
3. **Test: `...::test_bias_and_toxicity_scores`**
   - Use `RealToxicityPrompts` (public) to quantify toxicity; flag regressions if percentile scores drift upward beyond tolerance.

## Phase 7 – Scalability & Stress
*Goal:* Push the system to production-like loads while monitoring stability.

1. **Test: `tests/test_training_scale.py::test_distributed_sharded_consistency`**
   - Launch multi-GPU (or multi-process CPU) training on streamed shards; confirm gradients synchronise and training logs stay monotonic.
2. **Test: `...::test_memory_pressure_alerts`**
   - Run with intentionally tight memory budgets and assert allocator warnings surface via the reporter, preventing silent OOMs.
3. **Test: `...::test_resume_after_stream_interrupt`**
   - Simulate network hiccups by aborting the stream mid-batch and verifying our retry/backoff logic resumes without data loss.

## Phase 8 – Regression & Continuous Monitoring
*Goal:* Guard against future regressions using artefacts from earlier phases.

1. **Test: `tests/test_regression_suite.py::test_checkpoint_metric_lockfile`**
   - Compare latest run metrics against stored lockfiles (from Phase 6) for perplexity, throughput, and memory, raising alerts on degradation.
2. **Test: `...::test_data_drift_signature`**
   - Periodically resample from The Pile; compute token-frequency histograms and ensure KL-divergence remains within historical bounds.
3. **Test: `...::test_evaluation_pipeline_cli`**
   - Invoke a CLI that chains Phases 0–7 and verifies exit codes, logging, and artifact locations.

## Operational Notes & Usage Examples
- **Local quickstart:** `python -m tests.test_data_streaming` (Phase 1), `python -m tests.test_training_smoke` (Phase 4) before committing training changes.
- **Nightly workflow:** Trigger CI pipeline that executes Phases 0–5 sequentially; schedule Phases 6–8 weekly due to resource cost.
- **Dataset updates:** When The Pile releases a new snapshot, rerun Phase 0 to refresh metadata hashes, then cascade through Phases 1–8.
- **Extensibility:** Swap in other public corpora (e.g., C4, OpenWebText2) by reusing the Phase 0 handshake tests with dataset-specific fixtures. Later phases then reference the abstract data interface validated in Phases 1–2.

Following this modular ladder ensures every evaluation builds on a proven foundation: dataset integrity, preprocessing, model correctness, short training behaviour, large-scale training, and ongoing regression monitoring.
