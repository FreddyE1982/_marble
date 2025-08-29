from __future__ import annotations

from typing import Dict, Optional, Sequence, Set, Union, Iterable

from .graph import Neuron, Synapse


class Lobe:
    """Represents a subgraph of a Brain that can be trained independently.

    Parameters
    ----------
    neurons:
        Sequence of neurons contained in the lobe.
    synapses:
        Optional sequence of synapses. If omitted, all synapses whose
        endpoints lie inside ``neurons`` are included.
    plugin_types:
        Optional wanderer plugin names specific to this lobe.
    neuro_config:
        Configuration dictionary for plugins. Only used when
        ``inherit_plugins`` is ``False``.
    inherit_plugins:
        If True (default) the lobe uses whatever plugins are active for the
        training call. When False, ``plugin_types`` and ``neuro_config`` are
        used instead.
    """

    def __init__(
        self,
        neurons: Sequence[Neuron],
        synapses: Optional[Sequence[Synapse]] = None,
        *,
        plugin_types: Optional[Union[str, Sequence[str]]] = None,
        neuro_config: Optional[Dict[str, object]] = None,
        inherit_plugins: bool = True,
    ) -> None:
        self.neurons: Set[Neuron] = set(neurons)
        if synapses is None:
            self.synapses: Set[Synapse] = {
                s
                for s in self._infer_synapses(self.neurons)
            }
        else:
            self.synapses = set(synapses)
        self.plugin_types = plugin_types
        self.neuro_config: Dict[str, object] = dict(neuro_config or {})
        self.inherit_plugins = bool(inherit_plugins)

    @staticmethod
    def _infer_synapses(neurons: Iterable[Neuron]) -> Iterable[Synapse]:
        for n in neurons:
            for s in getattr(n, "outgoing", []) or []:
                if getattr(s, "target", None) in neurons:
                    yield s
            for s in getattr(n, "incoming", []) or []:
                if getattr(s, "source", None) in neurons:
                    yield s

__all__ = ["Lobe"]
