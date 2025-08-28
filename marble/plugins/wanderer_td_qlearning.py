from __future__ import annotations

from typing import Any, Optional, List, Tuple

from ..reporter import report


class TDQLearningPlugin:
    """Tabular TD(0) Q-learning over synapses stored in synapse._plugin_state['q'].

    - choose_next: epsilon-greedy over Q(synapse) for available choices.
    - on_step: TD update for the previous chosen synapse using reward = -current per-step loss.
    Config via wanderer._neuro_cfg: rl_epsilon (0.1), rl_alpha (0.1), rl_gamma (0.9)
    """

    def __init__(self) -> None:
        self._last_syn: Optional["Synapse"] = None

    def _q(self, syn: "Synapse") -> float:
        st = getattr(syn, "_plugin_state", None)
        if st is None:
            syn._plugin_state = {}
            st = syn._plugin_state
        q = st.get("q", 0.0)
        try:
            return float(q)
        except Exception:
            return 0.0

    def _set_q(self, syn: "Synapse", q: float) -> None:
        st = getattr(syn, "_plugin_state", None)
        if st is None:
            syn._plugin_state = {}
            st = syn._plugin_state
        st["q"] = float(q)

    def choose_next(self, wanderer: "Wanderer", current: "Neuron", choices: List[Tuple["Synapse", str]]):
        import random as _r
        if not choices:
            return None, "forward"
        eps = float(getattr(wanderer, "_neuro_cfg", {}).get("rl_epsilon", 0.1))
        if _r.random() < eps:
            return choices[_r.randrange(len(choices))]
        # Greedy by Q
        best = choices[0]
        best_q = self._q(best[0])
        for s, d in choices[1:]:
            q = self._q(s)
            if q > best_q:
                best = (s, d); best_q = q
        # Track chosen for TD update on next step
        self._last_syn = best[0]
        return best

    def on_step(self, wanderer: "Wanderer", current: "Neuron", next_syn: Optional["Synapse"], direction: str, step_index: int, out_value: Any) -> None:
        # TD update for previously chosen synapse using reward from current step loss
        if self._last_syn is None:
            return
        try:
            cur_loss_t = wanderer._walk_ctx.get("cur_loss_tensor")  # type: ignore[attr-defined]
            r = -float(cur_loss_t.detach().to("cpu").item()) if cur_loss_t is not None else 0.0
        except Exception:
            r = 0.0
        alpha = float(getattr(wanderer, "_neuro_cfg", {}).get("rl_alpha", 0.1))
        gamma = float(getattr(wanderer, "_neuro_cfg", {}).get("rl_gamma", 0.9))
        # Estimate max_a' Q(s',a') from the NEXT state's outgoing choices
        max_next = 0.0
        try:
            if next_syn is not None:
                next_node = next_syn.target if direction == "forward" else next_syn.source
            else:
                next_node = current
            choices = wanderer._gather_choices(next_node)
            if choices:
                max_next = max(self._q(s) for s, _ in choices)
        except Exception:
            pass
        q = self._q(self._last_syn)
        td_target = r + gamma * max_next
        new_q = q + alpha * (td_target - q)
        self._set_q(self._last_syn, new_q)
        try:
            report("wanderer", "td_q_update", {"q": new_q, "r": r}, "plugins")
        except Exception:
            pass
        # Reset last_syn only after applying update once; keep it tied to previous transition
        self._last_syn = next_syn

__all__ = ["TDQLearningPlugin"]
