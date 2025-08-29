from __future__ import annotations

from __future__ import annotations

"""Wanderer plugin that pre-trains on synthetic datapairs.

A learnable parameter ``n`` (number of samples) is exposed via
:func:`expose_learnable_params`. On initialisation the plugin generates ``n``
random datapairs and runs a short training session on them using
:func:`~marble.training.run_training_with_datapairs`. The additional training
happens once per Wanderer instance and is intentionally lightweight to keep
execution time predictable.
"""

from typing import List, Tuple

from ..wanderer import expose_learnable_params
from ..training import run_training_with_datapairs
from ..codec import UniversalTensorCodec


class SyntheticTrainingPlugin:
    """Generate synthetic samples and train on them during ``on_init``."""

    @staticmethod
    @expose_learnable_params
    def _get_params(wanderer: "Wanderer", *, n: float = 5.0):
        """Return tensor for number of synthetic samples to generate."""
        return (n,)

    def on_init(self, wanderer: "Wanderer") -> None:  # noqa: D401
        """Run a one-off synthetic training pass when the Wanderer is created."""
        if getattr(wanderer, "_synthetic_trained", False):
            return
        (n_t,) = self._get_params(wanderer)
        torch = getattr(wanderer, "_torch", None)
        if torch is None:
            wanderer._synthetic_trained = True
            return
        device = getattr(wanderer, "_device", "cpu")
        num = max(0, int(n_t.detach().to("cpu").item()))
        data = [
            (torch.randn(1, device=device), torch.randn(1, device=device))
            for _ in range(num)
        ]
        codec = UniversalTensorCodec()
        try:
            run_training_with_datapairs(
                wanderer.brain,  # type: ignore[attr-defined]
                data,
                codec,
                steps_per_pair=1,
                lr=1e-2,
                wanderer_type="epsilongreedy",
                seed=getattr(wanderer, "_seed", None),
            )
        except Exception:
            pass
        wanderer._synthetic_trained = True

__all__ = ["SyntheticTrainingPlugin"]
