from __future__ import annotations

"""BuildingBlock: reverse the direction of a synapse."""

from ..buildingblock import BuildingBlock
from ..graph import Neuron, Synapse
from ..wanderer import expose_learnable_params


class ReverseSynapsePlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, synapse: Synapse):
        if synapse not in getattr(brain, "synapses", []):
            return None
        src = synapse.source
        dst = synapse.target
        if isinstance(src, Neuron):
            try:
                src.outgoing.remove(synapse)
            except ValueError:
                pass
            if synapse.direction == "bi":
                try:
                    src.incoming.remove(synapse)
                except ValueError:
                    pass
        if isinstance(dst, Neuron):
            try:
                dst.incoming.remove(synapse)
            except ValueError:
                pass
            if synapse.direction == "bi":
                try:
                    dst.outgoing.remove(synapse)
                except ValueError:
                    pass
        synapse.source, synapse.target = dst, src
        if isinstance(dst, Neuron):
            dst.outgoing.append(synapse)
            if synapse.direction == "bi":
                dst.incoming.append(synapse)
        if isinstance(src, Neuron):
            src.incoming.append(synapse)
            if synapse.direction == "bi":
                src.outgoing.append(synapse)
        return synapse.source, synapse.target


__all__ = ["ReverseSynapsePlugin"]
