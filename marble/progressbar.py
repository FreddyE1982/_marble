from __future__ import annotations

"""Centralized progress bar utilities for Wanderer.

This module exposes a :class:`ProgressBar` singleton that wraps ``tqdm`` while
ensuring that no other part of the code base can instantiate a progress bar via
``tqdm`` directly. Any attempt to use ``tqdm`` outside this module raises a
``RuntimeError``.

The :class:`ProgressBar` mirrors the behaviour previously embedded in
``wanderer.py``: descriptions, postfix metrics and verbose walk messages are all
handled here so that Wanderer only has to supply the raw numbers.
"""

from typing import Any, Dict, Optional, ClassVar

import inspect
from pathlib import Path

import tqdm as _tqdm_mod
from tqdm.auto import tqdm as _tqdm_impl

# ---------------------------------------------------------------------------
# Guard against external tqdm usage
# ---------------------------------------------------------------------------
_ORIGINAL_TQDM = _tqdm_impl

_REPO_ROOT = Path(__file__).resolve().parent.parent


class _PatchedTqdm(_ORIGINAL_TQDM):  # pragma: no cover - thin wrapper
    """`tqdm` subclass that forbids direct usage from within the repo.

    External libraries are allowed to instantiate progress bars, but calls made
    from Marble modules raise a ``RuntimeError``. Third-party progress bars are
    silently disabled to avoid cluttering output.
    """

    def __new__(cls, *args: Any, **kwargs: Any):  # type: ignore[override]
        caller = inspect.stack()[1].filename
        if str(caller).startswith(str(_REPO_ROOT)) and not caller.endswith("progressbar.py"):
            raise RuntimeError("Use progressbar.ProgressBar; direct tqdm usage is forbidden")
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        kwargs.setdefault("disable", True)
        super().__init__(*args, **kwargs)


# Patch all public tqdm entry points so any external import uses the wrapper.
_tqdm_mod.tqdm = _PatchedTqdm  # type: ignore[assignment]
try:  # pragma: no cover - import side effects
    import tqdm.auto as _tqdm_auto_mod

    _tqdm_auto_mod.tqdm = _PatchedTqdm  # type: ignore[assignment]
except Exception:  # pragma: no cover - if tqdm.auto missing
    pass


class ProgressBarBase:
    """Base class enforcing a single progress bar instance."""

    _instance: ClassVar[Optional["ProgressBarBase"]] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "ProgressBarBase":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def start(self, total: int, *, leave: bool = False, verbose: bool = False, **meta: Any) -> None:
        """Start the progress bar.

        Parameters
        ----------
        total:
            Total number of steps.
        leave:
            Whether to leave the bar on screen when finished.
        verbose:
            If ``True``, per-walk start/end messages are emitted.
        meta:
            Additional values used by subclasses.
        """
        raise NotImplementedError

    def update(self, **data: Any) -> None:
        """Update the progress bar with the provided metrics."""
        raise NotImplementedError

    def end(self, **meta: Any) -> None:
        """Finalize the progress bar."""
        raise NotImplementedError


class ProgressBar(ProgressBarBase):
    """Concrete progress bar replicating Wanderer's original output."""

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._bar: Optional[Any] = None
        self.verbose = False
        self._initialized = True

    # ---- ProgressBarBase API -------------------------------------------------
    def start(self, total: int, *, leave: bool = False, verbose: bool = False, **meta: Any) -> None:
        if self._bar is not None:
            raise RuntimeError("Progress bar already running")
        self.verbose = verbose
        self._bar = _ORIGINAL_TQDM(total=total, leave=leave)
        try:
            self._bar.bar_format = (
                "{desc} {percentage:3.0f}% steps: {n_fmt}/{total_fmt}"
                " [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
        except Exception:  # pragma: no cover - cosmetic only
            pass
        if self.verbose:
            self._bar.write(
                f"{meta.get('cur_ep', 1)}/{meta.get('tot_ep', 1)} epochs "
                f"{meta.get('cur_walk', 1)}/{meta.get('tot_walks', 1)} walks: start"
            )

    def update(self, **data: Any) -> None:  # type: ignore[override]
        if self._bar is None:
            return
        cur_ep = data.get("cur_ep", 1)
        tot_ep = data.get("tot_ep", 1)
        cur_walk = data.get("cur_walk", 1)
        tot_walks = data.get("tot_walks", 1)
        desc = f"{cur_ep}/{tot_ep} epochs {cur_walk}/{tot_walks} walks:"
        # ``tqdm`` refreshes immediately when ``refresh=True`` (the default).
        # When combined with our explicit ``update`` call this caused the
        # progress line to be re-rendered multiple times per step, producing a
        # stream of duplicated lines instead of an in-place update in both
        # consoles and notebooks.  Passing ``refresh=False`` defers rendering to
        # the ``update`` call so the bar is rewritten exactly once per step.
        self._bar.set_description(desc, refresh=False)
        status: Dict[str, Any] = data.get("status", {})
        path_idx = data.get("path_index")
        try:
            self._bar.set_postfix(
                neurons=f"{data.get('cur_size', 0)}/{data.get('cap', '-')}",
                loss=f"{data.get('cur_loss', 0.0):.4f}",
                mean_loss=f"{data.get('mean_loss', 0.0):.4f}",
                loss_speed=f"{data.get('loss_speed', 0.0):.4f}",
                mean_loss_speed=f"{data.get('mean_loss_speed', 0.0):.4f}",
                neurons_added=status.get("neurons_added", 0),
                synapses=data.get("synapses", 0),
                synapses_added=status.get("synapses_added", 0),
                neurons_pruned=status.get("neurons_pruned", 0),
                synapses_pruned=status.get("synapses_pruned", 0),
                paths=data.get("paths", 0),
                path_index=path_idx if path_idx is not None else "-",
                speed=f"{data.get('mean_speed', 0.0):.2f}",
                refresh=False,
            )
        except Exception:  # pragma: no cover - cosmetic only
            pass
        self._bar.update(1)

    def end(self, **meta: Any) -> None:  # type: ignore[override]
        if self._bar is None:
            return
        status: Dict[str, Any] = meta.get("status", {})
        path_idx = meta.get("path_index")
        try:
            self._bar.set_postfix(
                neurons=f"{meta.get('cur_size', 0)}/{meta.get('cap', '-')}",
                loss=f"{meta.get('loss', 0.0):.4f}",
                mean_loss=f"{meta.get('mean_loss', 0.0):.4f}",
                loss_speed=f"{meta.get('loss_speed', 0.0):.4f}",
                mean_loss_speed=f"{meta.get('mean_loss_speed', 0.0):.4f}",
                neurons_added=status.get("neurons_added", 0),
                synapses=meta.get("synapses", 0),
                synapses_added=status.get("synapses_added", 0),
                neurons_pruned=status.get("neurons_pruned", 0),
                synapses_pruned=status.get("synapses_pruned", 0),
                paths=meta.get("paths", 0),
                path_index=path_idx if path_idx is not None else "-",
                speed=f"{meta.get('mean_speed', 0.0):.2f}",
                refresh=True,
            )
        except Exception:  # pragma: no cover - cosmetic only
            pass
        self._bar.close()
        if self.verbose:
            msg = (
                f"{meta.get('cur_ep', 1)}/{meta.get('tot_ep', 1)} epochs "
                f"{meta.get('cur_walk', 1)}/{meta.get('tot_walks', 1)} walks: end "
                f"(loss={meta.get('loss')}, steps={meta.get('steps')}"
            )
            if path_idx is not None:
                msg += f", path_index={path_idx}"
            msg += ")"
            self._bar.write(msg)
        self._bar = None
