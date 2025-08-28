from __future__ import annotations

from typing import Any, List


class ContrastiveInfoNCEPlugin:
    """Adds an InfoNCE-style contrastive loss across walk outputs.

    Config via wanderer._neuro_cfg:
      - contrastive_tau (temperature, default 0.1)
      - contrastive_lambda (weight, default 1.0)
    Positives are adjacent outputs in the same walk; negatives are all other outputs.
    """

    def _normalize(self, torch, x):
        x = x.view(-1)
        n = x.norm(p=2) + 1e-8
        return x / n

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch = getattr(wanderer, "_torch", None)
        if torch is None or len(outputs) < 2:
            return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        tau = float(getattr(wanderer, "_neuro_cfg", {}).get("contrastive_tau", 0.1))
        w = float(getattr(wanderer, "_neuro_cfg", {}).get("contrastive_lambda", 1.0))
        dev = getattr(wanderer, "_device", "cpu")
        # Build matrix of normalized embeddings
        vecs = []
        for y in outputs:
            if hasattr(y, "detach"):
                v = y.detach().to(dev).float().view(-1)
            else:
                v = torch.tensor([float(vv) for vv in (y if isinstance(y, (list, tuple)) else [y])], dtype=torch.float32, device=dev)
            vecs.append(self._normalize(torch, v))
        X = torch.stack(vecs, dim=0)  # [T, D]
        # Similarities
        S = X @ X.t()  # [T, T]
        S = S / max(1e-8, float(tau))
        # For each i>=1, positive is (i, i-1)
        loss_terms = []
        for i in range(1, X.shape[0]):
            logits = S[i]
            pos = logits[i - 1]
            # Exclude self from denominator by masking
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[i] = False
            denom = torch.logsumexp(logits[mask], dim=0)
            loss_i = - (pos - denom)
            loss_terms.append(loss_i)
        if not loss_terms:
            return torch.tensor(0.0, device=dev)
        return w * (torch.stack(loss_terms).mean())

__all__ = ["ContrastiveInfoNCEPlugin"]
