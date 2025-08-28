from marble.marblemain import (
    Brain,
    UniversalTensorCodec,
    make_datapair,
    run_training_with_datapairs,
    SelfAttention,
)


class AdaptiveLRRoutine:
    """Simple self-attention routine that adjusts LR based on recent per-step losses.

    Strategy:
    - Look at the last K per-step current_loss values retrieved via SelfAttention.history(K)
    - If the recent average is increasing vs. earlier average, decay LR; else increase mildly.
    - Uses and sets Wanderer attribute `lr_override` so changes apply next step.
    """

    def __init__(self, k_recent: int = 10, decay: float = 0.9, grow: float = 1.05, min_lr: float = 1e-5, max_lr: float = 1e-1) -> None:
        self.k_recent = int(k_recent)
        self.decay = float(decay)
        self.grow = float(grow)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)

    def after_step(self, sa: SelfAttention, ro, wanderer, step_idx: int, ctx):
        # Pull last 2K per-step logs from reporter via SelfAttention.history()
        hist = sa.history(self.k_recent * 2)
        if len(hist) < self.k_recent + 1:
            return None
        recent = [h.get("current_loss") for h in hist[-self.k_recent :]]
        prev = [h.get("current_loss") for h in hist[-2 * self.k_recent : -self.k_recent]]
        try:
            r_avg = sum(float(x) for x in recent if isinstance(x, (int, float))) / max(1, len(recent))
            p_avg = sum(float(x) for x in prev if isinstance(x, (int, float))) / max(1, len(prev))
        except Exception:
            return None
        cur_lr = sa.get_param("current_lr") or sa.get_param("lr_override") or 1e-3
        try:
            cur_lr = float(cur_lr)
        except Exception:
            cur_lr = 1e-3
        if r_avg > p_avg:
            new_lr = max(self.min_lr, cur_lr * self.decay)
        else:
            new_lr = min(self.max_lr, cur_lr * self.grow)
        return {"lr_override": float(new_lr)}


def main():
    b = Brain(2, size=(6, 6))
    codec = UniversalTensorCodec()

    # Build dataset from Wine
    from marble.marblemain import _try_load_wine

    data = _try_load_wine()
    X = data["data"] if isinstance(data, dict) else data.data
    y = data["target"] if isinstance(data, dict) else data.target
    pairs = [make_datapair([float(v) for v in X[i]], int(y[i])) for i in range(50)]

    sa = SelfAttention(routines=[AdaptiveLRRoutine(k_recent=10)])

    res = run_training_with_datapairs(
        b,
        pairs,
        codec,
        steps_per_pair=5,
        lr=1e-3,
        loss="nn.MSELoss",
        gradient_clip={"method": "norm", "max_norm": 1.0, "norm_type": 2.0},
        selfattention=sa,
    )
    print("wine with selfattention final_loss:", res.get("final_loss"))


if __name__ == "__main__":
    main()
