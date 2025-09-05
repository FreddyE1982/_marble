# Ensure environment flags are set before any potential torch import.
try:
    import os as _os
    _os.environ.setdefault("PYTORCH_DISABLE_NNPACK", "1")
except Exception:
    pass

from .auto_param import enable_auto_param_learning
from .plugin_encoder import PluginEncoder
from .action_sampler import compute_logits, sample_actions, select_plugins
from .offpolicy import Trajectory, importance_weights, doubly_robust
from .policy_gradient import PolicyGradientAgent
from .decision_controller import BUDGET_LIMIT

__all__ = [
    "enable_auto_param_learning",
    "PluginEncoder",
    "compute_logits",
    "sample_actions",
    "select_plugins",
    "Trajectory",
    "importance_weights",
    "doubly_robust",
    "PolicyGradientAgent",
    "BUDGET_LIMIT",
]
