import unittest


class TestAdvancedSynapsePlugins(unittest.TestCase):
    def _build_brain_with_plugin(self, plugin_name: str):
        from marble.marblemain import Brain

        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n2 = b.add_neuron((1.0, 0.0), tensor=[0.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        # remove auto synapse and reconnect with plugin
        for s in list(getattr(n2, "outgoing", [])):
            if getattr(getattr(s, "target", None), "position", None) == (0.0, 0.0):
                b.remove_synapse(s)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="uni", type_name=plugin_name)
        return b

    def _walk_and_collect(self, brain):
        from marble.marblemain import Wanderer

        w = Wanderer(brain, seed=1)
        start = brain.get_neuron((0.0, 0.0))
        w.walk(max_steps=1, lr=1e-2, start=start)
        return w

    def test_synapse_plugins_register_learnables(self):
        plugin_params = {
            "dropout": ["dropout_p"],
            "hebbian": ["hebb_rate", "hebb_decay"],
            "resonant": ["res_freq", "res_damp"],
            "delay": ["delay_alpha"],
            "spike_gate": ["gate_thresh", "gate_sharp"],
        }
        for name, params in plugin_params.items():
            brain = self._build_brain_with_plugin(name)
            w = self._walk_and_collect(brain)
            print("synapse plugin", name, "learnables", list(w._learnables.keys()))
            for p in params:
                self.assertIn(p, w._learnables, f"{name} missing {p}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

