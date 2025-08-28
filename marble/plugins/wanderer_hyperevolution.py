from __future__ import annotations

from ..marblemain import HyperEvolutionPlugin as _Base
from ..wanderer import register_wanderer_type


class HyperEvolutionPlugin(_Base):
    pass


try:
    register_wanderer_type("hyperEvolution", HyperEvolutionPlugin())
except Exception:
    pass

__all__ = ["HyperEvolutionPlugin"]

