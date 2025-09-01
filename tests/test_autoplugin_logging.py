import os
import tempfile
import unittest

import marble.plugins  # ensure plugin discovery
from marble.marblemain import Brain, Wanderer, register_wanderer_type
from marble.plugins.wanderer_autoplugin import AutoPlugin


class TestAutoPluginLogging(unittest.TestCase):
    def test_logging_occurs(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        register_wanderer_type("autoplugin_logger", AutoPlugin(log_path=tmp.name))
        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = Wanderer(b, type_name="autoplugin_logger", neuroplasticity_type="base", seed=0)
        auto = next(p for p in w._wplugins if isinstance(p, AutoPlugin))
        w.walk(max_steps=20, start=n, lr=0.01)
        with open(tmp.name, "r", encoding="utf-8") as fh:
            pre_data = fh.read()
        self.assertTrue(pre_data.strip())
        auto.finalize_logs(w)
        with open(tmp.name, "r", encoding="utf-8") as fh:
            data = fh.read()
        print("auto log:\n" + data)
        os.unlink(tmp.name)
        self.assertTrue(data.strip())


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
