# ADR: Align `run_training_with_datapairs` APIs (train_type support)

- Date: 2025-09-01
- Status: Accepted

## Context
Example scripts (e.g., `examples/run_hf_image_quality.py`) call `run_training_with_datapairs(..., train_type=...)`. The export re-imports the lock-based helper from `marble.training` at module end for serialization guarantees. That helper did not accept `train_type`, while the extended variant in `marble.marblemain` did, leading to a runtime `TypeError`.

## Decision
Extend `marble.training.run_training_with_datapairs` to accept `train_type` and implement the brain-train plugin lifecycle (`on_init`, `on_end`) by reusing the registries defined in `marble.marblemain` via lazy imports. Also apply any enabled learning paradigms to the active `Wanderer` for behavior parity. This preserves backward compatibility and unblocks examples that depend on `train_type`.

## Alternatives
- Remove the final re-import of helpers in `marble.marblemain`. Rejected to keep the lock-based implementations authoritative and avoid diverging code paths.
- Change examples to avoid `train_type`. Rejected; narrows functionality and hides legitimate plugin composition capabilities.

## Consequences
- The lock-based helper is now API-compatible with the main helper. Existing tests and scripts continue to work; scripts using `train_type` now run without error.
- Minimal additional lazy imports are introduced inside the helper to avoid import cycles.

## Verification
- Verified at runtime that `from marble.marblemain import run_training_with_datapairs` exposes the `marble.training` implementation and that its signature includes `train_type`.
- Smoke-tested a tiny datapair run confirming the helper executes and returns results.

