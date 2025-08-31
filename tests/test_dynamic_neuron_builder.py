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
        for name in [
            "reset_neuron_age",
            "randomize_neuron_weight",
            "randomize_neuron_bias",
            "scale_neuron_tensor",
            "normalize_neuron_tensor",
            "noise_neuron_tensor",
            "clamp_neuron_tensor",
            "reset_neuron_tensor",
            "shuffle_neuron_tensor",
            "abs_neuron_tensor",
            "square_neuron_tensor",
            "sqrt_neuron_tensor",
            "log_neuron_tensor",
            "exp_neuron_tensor",
            "sigmoid_neuron_tensor",
            "tanh_neuron_tensor",
            "relu_neuron_tensor",
            "softmax_neuron_tensor",
            "dropout_neuron_tensor",
        ]:
            w.ensure_learnable_param(f"dyn_use_{name}", -10.0)
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
