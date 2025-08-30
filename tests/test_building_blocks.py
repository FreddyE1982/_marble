import unittest

import marble.plugins  # noqa: F401 - ensure plugins load
from marble.marblemain import Brain
from marble.buildingblock import get_buildingblock_type


class TestBuildingBlocks(unittest.TestCase):
    def test_neuron_blocks(self):
        brain = Brain(1, size=2)
        create = get_buildingblock_type("create_neuron")
        neuron = create.apply(brain, (0,), [0.0])
        weight = get_buildingblock_type("change_neuron_weight")
        weight.apply(brain, (0,), 2.0)
        bias = get_buildingblock_type("change_neuron_bias")
        bias.apply(brain, (0,), 1.5)
        move = get_buildingblock_type("move_neuron")
        move.apply(brain, (0,), (1,))
        ntype = get_buildingblock_type("change_neuron_type")
        ntype.apply(brain, (1,), "sigmoid")
        print("neuron", neuron.weight, neuron.bias, neuron.type_name, neuron.position)
        self.assertEqual(brain.get_neuron((1,)), neuron)

    def test_synapse_blocks(self):
        brain = Brain(1, size=2)
        create = get_buildingblock_type("create_neuron")
        n0 = create.apply(brain, (0,), [0.0])
        n1 = create.apply(brain, (1,), [0.0])
        cs = get_buildingblock_type("create_synapse")
        syn = cs.apply(brain, (0,), (1,), weight=1.0, bias=0.5)
        w = get_buildingblock_type("change_synapse_weight")
        w.apply(brain, syn, 2.5)
        b = get_buildingblock_type("change_synapse_bias")
        b.apply(brain, syn, 1.0)
        mv = get_buildingblock_type("move_synapse")
        mv.apply(brain, syn, (1,), (0,))
        st = get_buildingblock_type("change_synapse_type")
        st.apply(brain, syn, "delay")
        print("synapse", syn.weight, syn.bias, syn.type_name, syn.source.position, syn.target.position)
        self.assertEqual(syn.source, n1)
        self.assertEqual(syn.target, n0)


if __name__ == "__main__":
    unittest.main()
