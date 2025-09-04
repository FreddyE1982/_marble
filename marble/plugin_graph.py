from __future__ import annotations

"""Simple dependency graph for Marble plugins.

Each plugin is represented as a node.  Directed edges indicate that the
**destination** plugin depends on the **source** plugin or must run in a later
phase.  After a plugin has been executed it is removed from the graph so that
remaining plugins with satisfied dependencies can be discovered quickly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class PluginGraph:
    """Directed graph tracking plugin dependencies and execution order."""

    _deps: Dict[str, Set[str]] = field(default_factory=dict)
    _executed: Set[str] = field(default_factory=set)

    def add_plugin(self, name: str) -> None:
        """Ensure ``name`` exists in the graph."""
        self._deps.setdefault(name, set())

    def add_dependency(self, prerequisite: str, plugin: str) -> None:
        """Add an edge ``prerequisite -> plugin``."""
        self.add_plugin(prerequisite)
        self.add_plugin(plugin)
        self._deps[plugin].add(prerequisite)

    def mark_executed(self, name: str) -> None:
        """Mark ``name`` as executed and update downstream dependencies."""
        if name not in self._deps:
            self.add_plugin(name)
        self._executed.add(name)
        self._deps.pop(name, None)
        for deps in self._deps.values():
            deps.discard(name)

    def recommend_next_plugin(self, current_phase: str | None = None) -> List[str]:
        """Return plugins whose dependencies are satisfied.

        Parameters
        ----------
        current_phase:
            Optional plugin name that has just finished executing.  When
            provided, it is marked as executed before computing the
            recommendations.
        """
        if current_phase is not None:
            self.mark_executed(current_phase)
        return [name for name, deps in self._deps.items() if not deps]

    def reset(self) -> None:
        """Clear all nodes and execution state."""
        self._deps.clear()
        self._executed.clear()


# Global instance used by plugin loader and decision controller
PLUGIN_GRAPH = PluginGraph()


def recommend_next_plugin(current_phase: str | None = None) -> List[str]:
    """Public helper returning ready-to-run plugins.

    This proxies to :class:`PLUGIN_GRAPH` so callers do not need to access the
    global instance directly.
    """

    return PLUGIN_GRAPH.recommend_next_plugin(current_phase)


__all__ = ["PluginGraph", "PLUGIN_GRAPH", "recommend_next_plugin"]
