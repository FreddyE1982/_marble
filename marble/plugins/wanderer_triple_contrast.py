from __future__ import annotations

from typing import Any, List

from ..reporter import report


class TripleWanderersContrastPlugin:
    """Runs two auxiliary wanderers and contrasts outputs of three walks.

    Upon each loss computation, spawns two additional Wanderer instances that
    perform walks on the same Brain. The final outputs of all three walks are
    compared pairwise using mean squared error, and the averaged value is
    returned as an auxiliary loss term.
    """

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch = getattr(wanderer, "_torch", None)
        if torch is None:
            return 0.0
        dev = getattr(wanderer, "_device", "cpu")
        main_out = outputs[-1] if outputs else torch.tensor(0.0, device=dev)
        brain = wanderer.brain
        steps = int(wanderer._walk_ctx.get("steps", len(outputs) or 1))
        lr = float(getattr(wanderer, "current_lr", 1e-2))
        # Gather outputs from two additional walks using fresh Wanderer instances
        extra_outs = []
        try:
            from ..marblemain import Wanderer as _Wanderer
            for _ in range(2):
                seed = wanderer.rng.randint(0, 2**31 - 1)
                w2 = _Wanderer(brain, seed=seed, mixedprecision=False)
                w2.walk(max_steps=steps, lr=lr)
                o2 = w2._walk_ctx.get("outputs", [])
                extra_outs.append(o2[-1] if o2 else torch.tensor(0.0, device=dev))
        except Exception:
            return torch.tensor(0.0, device=dev)
        all_outs = [main_out] + extra_outs
        losses = []
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = all_outs[i], all_outs[j]
                if hasattr(a, "detach") and hasattr(b, "detach"):
                    diff = a.detach().to(dev) - b.detach().to(dev)
                    losses.append((diff.view(-1) ** 2).mean())
                else:
                    at = torch.tensor([float(v) for v in (a if isinstance(a, (list, tuple)) else [a])], dtype=torch.float32, device=dev)
                    bt = torch.tensor([float(v) for v in (b if isinstance(b, (list, tuple)) else [b])], dtype=torch.float32, device=dev)
                    diff = at - bt
                    losses.append((diff.view(-1) ** 2).mean())
        if not losses:
            return torch.tensor(0.0, device=dev)
        loss = sum(losses) / len(losses)
        try:
            report("wanderer", "triple_contrast_loss", {"loss": float(loss.detach().to("cpu").item())}, "events")
        except Exception:
            pass
        return loss

__all__ = ["TripleWanderersContrastPlugin"]
