"""PhaseShiftRoutine tweaks attention temperature via a learnable phase shift."""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _phase_param(wanderer, phase_shift: float = 0.0):
    return phase_shift


class PhaseShiftRoutine:
    """Adjust temperature using a learnable phase shift."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        ph_t = _phase_param(wanderer)
        try:
            ph = float(ph_t.detach().to("cpu").item())
        except Exception:
            ph = 0.0
        mf = metric_factor(ctx, "phase_shift")

        # Gather reported state to ground decisions in self-attention context
        try:
            cur_loss = float(
                ctx.get("cur_loss_tensor").detach().to("cpu").item()
            )
        except Exception:
            cur_loss = None

        prev_loss = None
        try:
            last = reporter_ro.item(
                f"step_{max(1, step_index) - 1}", "wanderer_steps", "logs"
            )
            if isinstance(last, dict) and ("current_loss" in last):
                prev_loss = float(last.get("current_loss"))
        except Exception:
            prev_loss = None

        loss_speed = (
            (cur_loss - prev_loss)
            if cur_loss is not None and prev_loss is not None
            else 0.0
        )

        neuron_count = int(len(getattr(wanderer.brain, "neurons", {})))
        try:
            new_temp = 1.0 + ph * (1.0 + loss_speed / max(1.0, float(neuron_count))) * mf
            selfattention.set_param("temperature", new_temp)
        except Exception:
            new_temp = float("nan")

        report(
            "selfattention",
            "phase_shift",
            {
                "step": step_index,
                "phase": ph,
                "loss_speed": loss_speed,
                "neuron_count": neuron_count,
                "temperature": new_temp,
            },
            "events",
        )
        return None


__all__ = ["PhaseShiftRoutine"]

