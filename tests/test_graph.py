import unittest


class TestGraphBasics(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Neuron, Synapse, register_neuron_type, register_synapse_type

        self.Neuron = Neuron
        self.Synapse = Synapse
        self.register_neuron_type = register_neuron_type
        self.register_synapse_type = register_synapse_type

    def test_neuron_defaults_and_connections(self):
        n1 = self.Neuron([1.0, 2.0], weight=2.0, bias=1.0, age=3)
        n2 = self.Neuron([0.0], weight=1.0, bias=0.0)
        s = n1.connect_to(n2, direction="uni", age=5)

        # Connection book-keeping
        self.assertIn(s, n1.outgoing)
        self.assertIn(s, n2.incoming)
        self.assertEqual(n1.age, 3)
        self.assertEqual(s.age, 5)

        # Forward default behavior
        y = n1.forward()
        # y should be w*x + b -> [2*1+1, 2*2+1] = [3,5]
        if hasattr(y, "tolist"):
            ylist = y.tolist()
        else:
            ylist = y
        print("graph forward n1->y:", ylist)
        self.assertEqual([round(v, 6) for v in ylist], [3.0, 5.0])

        # Transmit value via synapse to n2 (should set tensor in n2)
        s.transmit(y)
        out = n2.forward()
        if hasattr(out, "tolist"):
            out = out.tolist()
        print("graph transmit to n2->out:", out)
        self.assertEqual([round(v, 6) for v in out], [3.0, 5.0])

    def test_bidirectional(self):
        n1 = self.Neuron([0.0])
        n2 = self.Neuron([0.0])
        s = self.Synapse(n1, n2, direction="bi")
        s.transmit([1.0], direction="forward")
        y = n2.forward()
        ylist = y.tolist() if hasattr(y, "tolist") else y
        print("graph bi forward to n2:", ylist)
        self.assertEqual([round(v, 6) for v in ylist], [1.0])

        s.transmit([2.0], direction="backward")
        y = n1.forward()
        ylist = y.tolist() if hasattr(y, "tolist") else y
        print("graph bi backward to n1:", ylist)
        self.assertEqual([round(v, 6) for v in ylist], [2.0])

    def test_plugins(self):
        # Define simple plugins inline
        class SquareNeuronPlugin:
            def on_init(self, neuron):
                neuron._plugin_state["square_init"] = True

            def forward(self, neuron, input_value=None):
                x = neuron.forward.__self__._ensure_tensor(neuron.tensor if input_value is None else input_value)  # noqa: B950
                if getattr(neuron, "_torch", None) is not None and neuron._is_torch_tensor(x):
                    y = x * neuron.weight + neuron.bias
                    return y * y
                else:
                    xl = x if isinstance(x, list) else list(x)
                    base = [neuron.weight * float(v) + neuron.bias for v in xl]
                    return [b * b for b in base]

        class DoubleTransmitSynapsePlugin:
            def on_init(self, synapse):
                synapse._plugin_state["double_init"] = True

            def transmit(self, synapse, value, *, direction="forward"):
                # Always double the value before default transmit
                x = synapse._ensure_tensor(value)
                synapse.transmit.__self__  # keep linter happy; we don't use recursion incorrectly
                if getattr(synapse, "_torch", None) is not None and synapse._is_torch_tensor(x):
                    doubled = x * 2
                else:
                    xl = x if isinstance(x, list) else list(x)
                    doubled = [2 * float(v) for v in xl]
                # Call base behavior manually
                if direction == "forward":
                    if synapse.direction in ("uni", "bi"):
                        synapse.target.receive(doubled)
                    else:
                        raise ValueError("This synapse does not allow forward transmission")
                else:
                    if synapse.direction == "bi":
                        synapse.source.receive(doubled)
                    else:
                        raise ValueError("This synapse does not allow backward transmission")

        self.register_neuron_type("square", SquareNeuronPlugin())
        self.register_synapse_type("double", DoubleTransmitSynapsePlugin())

        n1 = self.Neuron([1.0, 2.0], weight=2.0, bias=1.0, type_name="square")
        n2 = self.Neuron([0.0])
        s = self.Synapse(n1, n2, type_name="double")

        # Plugin init flags present
        self.assertTrue(n1._plugin_state.get("square_init", False))
        self.assertTrue(s._plugin_state.get("double_init", False))

        # Neuron forward squares ( (2x+1)^2 )
        y = n1.forward()
        ylist = y.tolist() if hasattr(y, "tolist") else y
        print("plugin square forward:", ylist)
        self.assertEqual([round(v, 6) for v in ylist], [9.0, 25.0])

        # Synapse doubles transmission
        s.transmit([3.0])
        out = n2.forward()
        out = out.tolist() if hasattr(out, "tolist") else out
        print("plugin double transmit -> n2:", out)
        self.assertEqual([round(v, 6) for v in out], [6.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
