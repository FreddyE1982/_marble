from __future__ import annotations

from __future__ import annotations

"""Wanderer plugin that maintains a learnable latent vector.

The plugin exposes the latent dimensionality via :func:`expose_learnable_params`
so the Wanderer can optimise it alongside other parameters.  The latent vector
is stored on the Wanderer and its L2 norm biases synapse selection: a higher
norm slightly favours edges with larger weights.  This keeps the plugin
lightweight while demonstrating integration of custom learnable parameters.
"""

from typing import List, Tuple

from ..wanderer import register_wanderer_type, expose_learnable_params


class LatentSpacePlugin:
    """Expose a learnable latent vector and use it to bias choices."""

    @staticmethod
    @expose_learnable_params
    def _get_params(wanderer: "Wanderer", *, latent_dim: float = 2.0):
        """Return the latent dimensionality tensor."""
        return (latent_dim,)

    def on_init(self, wanderer: "Wanderer") -> None:
        """Allocate the latent vector on first use."""
        (latent_dim_t,) = self._get_params(wanderer)
        torch = getattr(wanderer, "_torch", None)
        if torch is None:
            return
        dim = max(1, int(latent_dim_t.detach().to("cpu").item()))
        if "latent_vector" not in getattr(wanderer, "_learnables", {}):
            wanderer.ensure_learnable_param(
                "latent_vector", torch.zeros(dim, device=getattr(wanderer, "_device", "cpu"))
            )

    def choose_next(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        choices: List[Tuple["Synapse", str]],
    ):
        """Select the choice with maximal weight plus latent bias."""
        if not choices:
            return None, "forward"
        torch = getattr(wanderer, "_torch", None)
        latent = wanderer.get_learnable_param_tensor("latent_vector")
        bias = 0.0
        if torch is not None:
            bias = float(latent.norm().detach().to("cpu").item())
        return max(choices, key=lambda cd: float(getattr(cd[0], "weight", 1.0)) + bias)


try:  # pragma: no cover - registration failure should not break import
    register_wanderer_type("latentspace", LatentSpacePlugin())
except Exception:
    pass

__all__ = ["LatentSpacePlugin"]
