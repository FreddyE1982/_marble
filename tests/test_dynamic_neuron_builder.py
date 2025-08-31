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
        w.ensure_learnable_param("dyn_use_reset_neuron_age", -10.0)
        w.ensure_learnable_param("dyn_use_randomize_neuron_weight", -10.0)
        w.ensure_learnable_param("dyn_use_randomize_neuron_bias", -10.0)
        w.ensure_learnable_param("dyn_use_scale_neuron_tensor", -10.0)
        w.ensure_learnable_param("dyn_use_normalize_neuron_tensor", -10.0)
        w.ensure_learnable_param("dyn_use_noise_neuron_tensor", -10.0)
        w.ensure_learnable_param("dyn_use_clamp_neuron_tensor", -10.0)
        w.ensure_learnable_param("dyn_use_reset_neuron_tensor", -10.0)
        w.ensure_learnable_param("dyn_use_shuffle_neuron_tensor", -10.0)
        w.ensure_learnable_param("dyn_shift_neuron_tensor_delta", 2.0)
        w.walk(max_steps=1)
        dyn = [n for n in _NEURON_TYPES.keys() if n.startswith("dyn_")]
        self.assertTrue(dyn)
        n = b.get_neuron((0,))
        self.assertEqual(list(n.tensor), [2.0])
        auto = AutoNeuronPlugin()
        self.assertIn(dyn[0], auto._candidate_types())


if __name__ == "__main__":
    unittest.main()
