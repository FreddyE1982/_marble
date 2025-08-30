from __future__ import annotations

"""BuildingBlock: move a synapse to new source/target neurons."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..graph import Neuron, Synapse
from ..wanderer import expose_learnable_params


class MoveSynapsePlugin(BuildingBlock):
    def _detach(self, syn: Synapse) -> None:
        src = syn.source
        dst = syn.target
        if isinstance(src, Neuron):
            try:
                src.outgoing.remove(syn)
            except Exception:
                pass
            if syn.direction == "bi":
                try:
                    src.incoming.remove(syn)
                except Exception:
                    pass
        else:
            try:
                src.outgoing_synapses.remove(syn)
            except Exception:
                pass
            if syn.direction == "bi":
                try:
                    src.incoming_synapses.remove(syn)
                except Exception:
                    pass
        if isinstance(dst, Neuron):
            try:
                dst.incoming.remove(syn)
            except Exception:
                pass
            if syn.direction == "bi":
                try:
                    dst.outgoing.remove(syn)
                except Exception:
                    pass
        else:
            try:
                dst.incoming_synapses.remove(syn)
            except Exception:
                pass
            if syn.direction == "bi":
                try:
                    dst.outgoing_synapses.remove(syn)
                except Exception:
                    pass

    @expose_learnable_params
    def apply(self, brain, synapse: Synapse, new_source_index: Sequence[int], new_target_index: Sequence[int]):
        if synapse not in getattr(brain, "synapses", []):
            return None
        new_src = brain.get_neuron(self._to_index(brain, new_source_index))
        new_dst = brain.get_neuron(self._to_index(brain, new_target_index))
        if new_src is None or new_dst is None:
            return None
        self._detach(synapse)
        synapse.source = new_src
        synapse.target = new_dst
        new_src.outgoing.append(synapse)
        new_dst.incoming.append(synapse)
        if synapse.direction == "bi":
            new_src.incoming.append(synapse)
            new_dst.outgoing.append(synapse)
        return synapse


__all__ = ["MoveSynapsePlugin"]
