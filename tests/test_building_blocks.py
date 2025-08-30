import unittest

import marble.plugins  # noqa: F401 - ensure plugins load
from marble.marblemain import Brain
from marble.buildingblock import get_buildingblock_type


class TestBuildingBlocks(unittest.TestCase):
    def test_neuron_blocks(self):
        brain = Brain(1, size=2)
        create = get_buildingblock_type("create_neuron")
        n0 = create.apply(brain, (0,), [0.0])
        n1 = create.apply(brain, (1,), [0.0])
        weight = get_buildingblock_type("change_neuron_weight")
        weight.apply(brain, (0,), 2.0)
        bias = get_buildingblock_type("change_neuron_bias")
        bias.apply(brain, (0,), 1.5)
        scale_w = get_buildingblock_type("scale_neuron_weight")
        scale_w.apply(brain, (0,), 2.0)
        scale_b = get_buildingblock_type("scale_neuron_bias")
        scale_b.apply(brain, (0,), 2.0)
        age = get_buildingblock_type("increment_neuron_age")
        age.apply(brain, (0,), 5)
        swap = get_buildingblock_type("swap_neurons")
        swap.apply(brain, (0,), (1,))
        delete = get_buildingblock_type("delete_neuron")
        delete.apply(brain, (0,))
        print("neuron", n0.weight, n0.bias, n0.age, n0.position, brain.get_neuron((0,)))
        self.assertIsNone(brain.get_neuron((0,)))
        self.assertEqual(n0.position, (1,))
        self.assertEqual(n0.weight, 4.0)
        self.assertEqual(n0.bias, 3.0)
        self.assertEqual(n0.age, 5)

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
        scale_w = get_buildingblock_type("scale_synapse_weight")
        scale_w.apply(brain, syn, 2.0)
        scale_b = get_buildingblock_type("scale_synapse_bias")
        scale_b.apply(brain, syn, 2.0)
        age = get_buildingblock_type("increment_synapse_age")
        age.apply(brain, syn, 3)
        rev = get_buildingblock_type("reverse_synapse")
        rev.apply(brain, syn)
        delete = get_buildingblock_type("delete_synapse")
        delete.apply(brain, syn)
        print("synapse", syn.weight, syn.bias, syn.age, len(brain.synapses))
        self.assertEqual(len(brain.synapses), 0)
        self.assertEqual(syn.weight, 5.0)
        self.assertEqual(syn.bias, 2.0)
        self.assertEqual(syn.age, 3)
        self.assertEqual(syn.source, n1)
        self.assertEqual(syn.target, n0)


if __name__ == "__main__":
    unittest.main()
