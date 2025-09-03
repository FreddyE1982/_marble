import unittest

from marble.marblemain import Brain


class TestNeuronConnectivity(unittest.TestCase):
    def test_second_neuron_requires_connection(self):
        b = Brain(1, size=2)
        b.add_neuron((0,), tensor=0.0)
        with self.assertRaises(RuntimeError):
            b.add_neuron((1,), tensor=0.0)

    def test_second_neuron_with_connection(self):
        b = Brain(1, size=2)
        b.add_neuron((0,), tensor=0.0)
        b.add_neuron((1,), tensor=0.0, connect_to=(0,))
        self.assertEqual(len(b.synapses), 1)

