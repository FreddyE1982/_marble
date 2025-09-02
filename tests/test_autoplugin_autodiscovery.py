import os
import tempfile
import unittest

import marble.plugins  # ensure plugin discovery
from marble.marblemain import Brain, Wanderer, register_wanderer_type
from marble.plugins.wanderer_autoplugin import AutoPlugin


class TestAutoPluginAutoDiscovery(unittest.TestCase):
    def test_unforced_plugin_logged(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        register_wanderer_type("autoplugin_logger2", AutoPlugin(log_path=tmp.name))
        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = Wanderer(b, type_name="autoplugin_logger2", neuroplasticity_type="base", seed=0)
        auto = next(p for p in w._wplugins if isinstance(p, AutoPlugin))
        w.ensure_learnable_param("autoplugin_bias_EpsilonGreedyChooserPlugin", 0.0)
        w.get_learnable_param_tensor("autoplugin_bias_EpsilonGreedyChooserPlugin").data.fill_(10.0)
        w.walk(max_steps=5, start=n, lr=0.01)
        auto.finalize_logs(w)
        with open(tmp.name, "r", encoding="utf-8") as fh:
            data = fh.read()
        os.unlink(tmp.name)
        stack = [getattr(getattr(p, "_plugin", p), "__class__").__name__ for p in w._wplugins]
        print("stack:", stack)
        print("log:", data.strip())
        self.assertIn("EpsilonGreedyChooserPlugin", data)
        self.assertIn("EpsilonGreedyChooserPlugin", stack)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
