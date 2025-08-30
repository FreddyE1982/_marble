from __future__ import annotations

"""QuantumType neuron plugin implementing weight superposition.

Each neuron keeps ``n`` candidate weights and biases simultaneously. A
learnable wave function (implemented via softmax over logits) assigns a
probability to every candidate state. During the forward pass the output is
the expectation over all states which keeps the computation fully
differentiable and avoids the high gradient variance that a pure sampling
approach would introduce.

The plugin exposes the wave-function logits through
``expose_learnable_params`` so that they are registered as learnable
parameters on the :class:`~marble.wanderer.Wanderer`.  The per-neuron weights
and biases are tracked in ``neuron._plugin_state['learnable_params']`` which
ensures gradients flow to every superposed value.

This design solves several of the downsides mentioned in the high level
discussion:

* **Stable gradients** – The expectation formulation eliminates stochastic
  gradients caused by random collapse.
* **Differentiable probabilities** – Logits are trained directly and turned
  into probabilities via ``softmax`` which provides smooth gradients.
* **Single-pass computation** – All states are evaluated in parallel and
  combined in one pass, avoiding expensive resampling.

The plugin also keeps track of possible neuron positions.  The neuron's
``position`` attribute is updated to the probability‑weighted expectation of
all candidate positions so that downstream logic receives a deterministic
coordinate.
"""

from typing import Any, List, Sequence, Tuple

from ..wanderer import expose_learnable_params
from ..reporter import report


class QuantumTypeNeuronPlugin:
    """Neuron plugin that models a superposition of multiple weights.

    Parameters
    ----------
    n_states:
        Number of simultaneous weight/bias/position states maintained by the
        neuron.  More states increase expressiveness at the cost of compute.
    """

    def __init__(self, n_states: int = 2) -> None:
        self.n_states = int(max(1, n_states))

    # ------------------------------------------------------------------
    # Learnable global wave-function parameters
    @expose_learnable_params
    def _wave_logits(
        self,
        wanderer: "Wanderer",
        *,
        logits: Sequence[float] | None = None,
    ):
        """Return learnable logits for the wave function.

        ``expose_learnable_params`` ensures that a single tensor named
        ``"logits"`` lives on the :class:`Wanderer`.  The tensor holds
        ``n_states`` values representing the unnormalised log probabilities for
        the neuron to collapse into each state.
        """

        if logits is None:
            logits = [0.0] * self.n_states
        return logits

    # ------------------------------------------------------------------
    def _ensure_internal_params(self, neuron: "Neuron") -> Tuple[Any, Any, List[Any]]:
        """Initialise and return (weights, biases, positions) for the neuron."""

        store = neuron._plugin_state.setdefault("learnable_params", {})
        torch = neuron._torch
        device = getattr(neuron, "_device", "cpu")

        if "weights" not in store:
            if torch is not None:
                store["weights"] = torch.full(
                    (self.n_states,), float(neuron.weight), dtype=torch.float32, device=device
                )
            else:
                store["weights"] = [float(neuron.weight)] * self.n_states

        if "biases" not in store:
            if torch is not None:
                store["biases"] = torch.full(
                    (self.n_states,), float(neuron.bias), dtype=torch.float32, device=device
                )
            else:
                store["biases"] = [float(neuron.bias)] * self.n_states

        if "positions" not in store:
            pos = getattr(neuron, "position", None)
            store["positions"] = [pos for _ in range(self.n_states)]

        return store["weights"], store["biases"], store["positions"]

    # ------------------------------------------------------------------
    def forward(self, neuron: "Neuron", input_value=None):
        """Compute expectation over all weight states."""

        torch = neuron._torch
        device = getattr(neuron, "_device", "cpu")
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)
        weights, biases, positions = self._ensure_internal_params(neuron)

        wanderer = neuron._plugin_state.get("wanderer")
        logits = None
        if wanderer is not None:
            try:
                logits = wanderer.get_learnable_param_tensor("logits")
            except Exception:
                logits = None
            if logits is None:
                try:
                    logits = self._wave_logits(wanderer)
                except Exception:
                    logits = None

        if torch is not None and neuron._is_torch_tensor(x):
            w = weights if hasattr(weights, "to") else torch.tensor(weights, dtype=torch.float32, device=device)
            b = biases if hasattr(biases, "to") else torch.tensor(biases, dtype=torch.float32, device=device)
            if logits is None:
                logits_t = torch.zeros(self.n_states, dtype=torch.float32, device=device)
            else:
                logits_t = logits if hasattr(logits, "to") else torch.tensor(logits, dtype=torch.float32, device=device)
            probs = torch.softmax(logits_t, dim=0)

            out_each = x * w.view(-1, *([1] * (x.dim() if hasattr(x, "dim") else 1))) + b.view(
                -1, *([1] * (x.dim() if hasattr(x, "dim") else 1))
            )
            result = (probs.view(-1, *([1] * (out_each.dim() - 1))) * out_each).sum(0)

            # Update expected position deterministically
            try:
                pos_tensors = []
                for p in positions:
                    if isinstance(p, (tuple, list)):
                        pos_tensors.append(torch.tensor(p, dtype=torch.float32, device=device))
                    else:
                        pos_tensors.append(torch.zeros(1, dtype=torch.float32, device=device))
                if pos_tensors:
                    pos_stack = torch.stack(pos_tensors)
                    exp_pos = (probs.view(-1, 1) * pos_stack).sum(0)
                    neuron.position = tuple(float(v.detach().to("cpu").item()) for v in exp_pos)
            except Exception:
                pass

            try:
                report(
                    "neuron",
                    "quantum_forward",
                    {"states": self.n_states, "prob_max": float(probs.max().detach().to("cpu").item())},
                    "plugins",
                )
            except Exception:
                pass
            return result

        # Fallback to pure Python lists
        x_list = x if isinstance(x, list) else [float(x)]
        w_list = list(weights)
        b_list = list(biases)
        log_list = list(logits) if logits is not None else [0.0] * self.n_states
        max_log = max(log_list)
        exp_vals = [pow(2.718281828459045, l - max_log) for l in log_list]
        norm = sum(exp_vals) if exp_vals else 1.0
        probs = [v / norm for v in exp_vals]
        out_vals: List[List[float]] = []
        for p, wv, bv in zip(probs, w_list, b_list):
            out_vals.append([p * (wv * float(v) + bv) for v in x_list])
        result = [sum(vals) for vals in zip(*out_vals)]
        try:
            report(
                "neuron",
                "quantum_forward",
                {"states": self.n_states, "prob_max": max(probs) if probs else 0.0},
                "plugins",
            )
        except Exception:
            pass
        return result if len(result) > 1 else result[0]

    # ------------------------------------------------------------------
    def receive(self, neuron: "Neuron", value):
        """Store incoming value as base tensor."""

        neuron.tensor = neuron._ensure_tensor(value)


__all__ = ["QuantumTypeNeuronPlugin"]

