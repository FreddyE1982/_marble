import unittest

from marble.marblemain import Brain
from marble.graph import _NEURON_TYPES
from marble.wanderer import Wanderer
from marble.plugins.autoneuron import AutoNeuronPlugin


class TestDynamicNeuronBuilder(unittest.TestCase):
    def test_builder_registers_type(self) -> None:
        b = Brain(1, size=2)
        b.add_neuron((0,), tensor=[0.0])
        # plugin auto-registered under name "neuronbuilder"
        w = Wanderer(b, type_name="neuronbuilder")
        w.walk(max_steps=1)
        dyn = [n for n in _NEURON_TYPES.keys() if n.startswith("dyn_")]
        self.assertTrue(dyn)
        auto = AutoNeuronPlugin()
        self.assertIn(dyn[0], auto._candidate_types())


if __name__ == "__main__":
    unittest.main()
