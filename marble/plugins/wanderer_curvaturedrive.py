"""Curvature-driven wanderer plugin."""

from __future__ import annotations

from typing import List, Tuple

import random

from ..reporter import report
from ..wanderer import expose_learnable_params


class CurvatureDrivePlugin:
    """Encourage smooth paths based on recent movement curvature."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, curve_factor: float = 1.0):
        return (curve_factor,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        visited = getattr(wanderer, "_visited", [])
        (factor_t,) = self._params(wanderer)
        torch = getattr(wanderer, "_torch", None)
        device = getattr(wanderer, "_device", "cpu")
        if torch is not None and len(visited) >= 2:
            prev = visited[-1]
            prev2 = visited[-2]
            pv = torch.tensor(
                [float(a) - float(b) for a, b in zip(getattr(prev, "position", (0,)), getattr(prev2, "position", (0,)))],
                dtype=torch.float32,
                device=device,
            )
            scores = []
            for syn, direction in choices:
                target = syn.target if direction == "forward" else syn.source
                tv = torch.tensor(
                    [float(a) - float(b) for a, b in zip(getattr(target, "position", (0,)), getattr(prev, "position", (0,)))],
                    dtype=torch.float32,
                    device=device,
                )
                sc = -torch.norm(pv - tv) * factor_t
                scores.append(sc)
            weights = torch.softmax(torch.stack(scores), dim=0)
            idx = int(torch.multinomial(weights, 1).item())
        else:
            idx = random.randrange(len(choices))
        report("wanderer", "curvature_drive", {"choice": idx}, "plugins")
        return choices[idx]


__all__ = ["CurvatureDrivePlugin"]

