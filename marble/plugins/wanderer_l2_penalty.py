from __future__ import annotations

from typing import Any, List

from ..wanderer import register_wanderer_type


class L2WeightPenaltyPlugin:
    """Adds L2 penalty over visited neurons' weights and biases to the loss.

    Reads lambda from wanderer._neuro_cfg['l2_lambda'] (default 0.0).
    """

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch = getattr(wanderer, "_torch", None)
        lam = float(getattr(wanderer, "_neuro_cfg", {}).get("l2_lambda", 0.0))
        if lam <= 0.0 or torch is None:
            # No contribution
            return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        total = None
        for n in getattr(wanderer, "_visited", []) or []:
            try:
                w_param, b_param = wanderer._param_map[id(n)]
                term = (w_param.view(-1) ** 2).sum() + (b_param.view(-1) ** 2).sum()
                total = term if total is None else (total + term)
            except Exception:
                continue
        if total is None:
            return torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        return lam * total


try:
    register_wanderer_type("l2_weight_penalty", L2WeightPenaltyPlugin())
except Exception:
    pass

__all__ = ["L2WeightPenaltyPlugin"]
