from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type


@dataclass
class LearnableParam:
    """Container tracking metadata for a learnable parameter.

    Parameters are always stored as floating point tensors for autograd but we
    keep the original Python type and optional bounds so the value can be
    clamped and cast back after each optimizer step.
    """

    tensor: Any
    orig_type: Type
    opt: bool = False
    lr: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    display_name: Optional[str] = None

    def apply_constraints(self) -> None:
        """Clamp and cast the underlying tensor in-place."""
        t = self.tensor
        data = getattr(t, "data", t)
        try:
            if self.min_value is not None or self.max_value is not None:
                minv = self.min_value if self.min_value is not None else float("-inf")
                maxv = self.max_value if self.max_value is not None else float("inf")
                data.clamp_(minv, maxv)
            if self.orig_type is int:
                data.round_()
        except Exception:
            # If anything goes wrong, leave the tensor untouched
            pass
