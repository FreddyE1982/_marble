# Ensure environment flags are set before any potential torch import.
try:
    import os as _os
    _os.environ.setdefault("PYTORCH_DISABLE_NNPACK", "1")
except Exception:
    pass

from .auto_param import enable_auto_param_learning
from .plugin_encoder import PluginEncoder
from .action_sampler import compute_logits, sample_actions, select_plugins

__all__ = [
    "enable_auto_param_learning",
    "PluginEncoder",
    "compute_logits",
    "sample_actions",
    "select_plugins",
]
