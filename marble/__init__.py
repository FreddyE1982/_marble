# Ensure environment flags are set before any potential torch import.
try:
    import os as _os
    _os.environ.setdefault("PYTORCH_DISABLE_NNPACK", "1")
except Exception:
    pass
