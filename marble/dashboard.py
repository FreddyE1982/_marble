from __future__ import annotations
import json, os, threading, time, datetime, tempfile
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict

# Global path to share metrics between training process and Streamlit app
_METRICS_FILE = Path(tempfile.gettempdir()) / "marble_dashboard_metrics.json"

# Track plugin action counts
_plugin_actions: Dict[str, int] = defaultdict(int)

# Flag indicating whether dashboard is running
_dashboard_active = False


def dashboard_active() -> bool:
    """Return True if the dashboard thread has been started."""
    return _dashboard_active


def _last_snapshot_info(brain: Any) -> tuple[None | str, None | float]:
    path = getattr(brain, "snapshot_path", None)
    if not path:
        return None, None
    try:
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".marble")]
        if not files:
            return None, None
        latest = max(files, key=os.path.getmtime)
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(latest)).isoformat()
        size_mb = os.path.getsize(latest) / (1024 * 1024)
        return ts, size_mb
    except Exception:
        return None, None


def update_metrics(
    brain: Any,
    wanderer: Any,
    step_index: int,
    max_steps: int,
    cur_loss: float,
    mean_loss: float,
    loss_speed: float,
    mean_loss_speed: float,
) -> None:
    """Write latest training metrics to the shared metrics file."""
    global _plugin_actions
    try:
        from .graph import _NEURON_TYPES
        from .wanderer import WANDERER_TYPES_REGISTRY, NEURO_TYPES_REGISTRY
        import torch
    except Exception:
        return

    plugin_names = [p.__class__.__name__ for p in getattr(wanderer, "_wplugins", []) or []]
    plugin_names += [p.__class__.__name__ for p in getattr(wanderer, "_neuro_plugins", []) or []]
    for name in plugin_names:
        _plugin_actions[name] += 1
    total_actions = sum(_plugin_actions.values())
    most_active = max(_plugin_actions.items(), key=lambda x: x[1])[0] if _plugin_actions else None

    active_plugins = len(plugin_names)
    available_plugins = len(WANDERER_TYPES_REGISTRY) + len(NEURO_TYPES_REGISTRY)

    used_neuron_types = {
        getattr(n, "type_name", None)
        for n in getattr(brain, "neurons", {}).values()
        if getattr(n, "type_name", None)
    }
    neuron_types_used = len(used_neuron_types)
    neuron_types_available = len(_NEURON_TYPES)

    paths = len(getattr(brain, "synapses", []))

    snapshot_time, snapshot_size = _last_snapshot_info(brain)

    data = {
        "epoch": int(getattr(brain, "_progress_epoch", 0) + 1),
        "total_epochs": int(getattr(brain, "_progress_total_epochs", 1)),
        "sample": int(getattr(brain, "_progress_walk", 0) + 1),
        "walk_step": int(step_index + 1),
        "walk_total": int(max_steps),
        "current_loss": float(cur_loss),
        "mean_loss": float(mean_loss),
        "loss_speed": float(loss_speed),
        "mean_loss_speed": float(mean_loss_speed),
        "paths": int(paths),
        "plugins_active": int(active_plugins),
        "plugins_available": int(available_plugins),
        "plugin_actions": int(total_actions),
        "most_active_plugin": most_active,
        "neuron_types_used": int(neuron_types_used),
        "neuron_types_available": int(neuron_types_available),
        "neurons_added": int(getattr(brain, "neurons_added", 0)),
        "neurons_removed": int(getattr(brain, "neurons_pruned", 0)),
        "synapses_added": int(getattr(brain, "synapses_added", 0)),
        "synapses_removed": int(getattr(brain, "synapses_pruned", 0)),
        "cuda": bool(torch.cuda.is_available()),
        "last_snapshot_time": snapshot_time,
        "last_snapshot_size_mb": snapshot_size,
    }
    try:
        _METRICS_FILE.write_text(json.dumps(data))
    except Exception:
        pass


def _run_streamlit(port: int) -> None:
    import subprocess, sys
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_ngrok(port: int) -> None:
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token("2o9DgUKuP2W8vjV7cZFq0sDiM3A_2d1gWrkXqvy5APpUn2QNS")
        ngrok.connect(port, "http", hostname="alpaca-model-easily.ngrok-free.app")
    except Exception:
        pass


def start_dashboard(port: int = 8501) -> None:
    """Start the Streamlit dashboard and ngrok tunnel in background threads."""
    global _dashboard_active
    if _dashboard_active:
        return
    _dashboard_active = True
    threading.Thread(target=_run_streamlit, args=(port,), daemon=True).start()
    threading.Thread(target=_run_ngrok, args=(port,), daemon=True).start()


# Streamlit app entry point

def main() -> None:
    import streamlit as st

    st.title("Marble Training Dashboard")
    placeholder = st.empty()

    try:
        data = json.loads(_METRICS_FILE.read_text())
    except Exception:
        data = {}

    placeholder.json(data)
    time.sleep(1.0)
    st.experimental_rerun()


if __name__ == "__main__":
    main()
