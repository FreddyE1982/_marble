import os
import tempfile
import unittest

import marble.plugins  # ensure plugin discovery
from marble.marblemain import Brain, Wanderer, register_neuron_type
from marble.plugins.autoneuron import AutoNeuronPlugin


class TestAutoNeuronLogging(unittest.TestCase):
    def test_logging_occurs(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        auto_n = AutoNeuronPlugin(log_path=tmp.name)
        register_neuron_type("autoneuron_logger", auto_n)
        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[1.0], type_name="autoneuron_logger")
        w = Wanderer(b, seed=0, neuroplasticity_type="base")
        for _ in range(4):
            w.walk(max_steps=20, start=n, lr=0.01)
        auto_n.finalize_logs(w)
        with open(tmp.name, "r", encoding="utf-8") as fh:
            data = fh.read()
        print("neuron log:\n" + data)
        os.unlink(tmp.name)
        self.assertTrue(data.strip())


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
