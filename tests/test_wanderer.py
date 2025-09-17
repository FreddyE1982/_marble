import contextlib
import io
import unittest


class TestWanderer(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer
        self.Brain = Brain
        self.Wanderer = Wanderer

    def test_wanderer_basic_updates(self):
        # Build a tiny sparse brain with two neurons connected bidirectionally
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0, 2.0], weight=1.0, bias=0.5)
        n2 = b.add_neuron((1.0, 0.0), tensor=[0.5, 1.5], weight=0.8, bias=-0.2, connect_to=(0.0, 0.0))
        for s in list(getattr(n2, "outgoing", [])):
            if s.target is n1:
                b.remove_synapse(s)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        # Snapshot original weights/biases
        orig = {(id(n1)): (n1.weight, n1.bias), (id(n2)): (n2.weight, n2.bias)}

        w = self.Wanderer(b, seed=123)
        stats = w.walk(max_steps=5, lr=1e-2)
        print("wanderer basic stats:", stats)

        self.assertGreaterEqual(stats["steps"], 1)
        self.assertGreaterEqual(stats["visited"], 1)

        # Ensure at least one visited neuron's params changed
        changed = False
        for vn in w._visited:
            ow, ob = orig[id(vn)] if id(vn) in orig else (None, None)
            if ow is None:
                continue
            if abs(float(vn.weight) - float(ow)) > 1e-9 or abs(float(vn.bias) - float(ob)) > 1e-9:
                changed = True
                break
        self.assertTrue(changed)

    def test_wanderer_plugin_choose_next_called(self):
        from marble.marblemain import register_wanderer_type, Wanderer, Brain

        class DeterministicPlugin:
            def on_init(self, wanderer):
                wanderer._plugin_state["choose_calls"] = 0

            def choose_next(self, wanderer, current, choices):
                # Count calls and choose the first available option deterministically
                wanderer._plugin_state["choose_calls"] += 1
                return choices[0]

        register_wanderer_type("det", DeterministicPlugin())

        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n2 = b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        for s in list(getattr(n2, "outgoing", [])):
            if s.target is b.get_neuron((0.0, 0.0)):
                b.remove_synapse(s)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        w = Wanderer(b, type_name="det", seed=1)
        stats = w.walk(max_steps=4, lr=1e-2)
        print("wanderer plugin choose calls:", w._plugin_state.get("choose_calls", 0))

        # choose_next should be called once per step taken when choices exist (>=1)
        self.assertGreaterEqual(w._plugin_state.get("choose_calls", 0), 1)
        self.assertGreaterEqual(stats["steps"], 1)

    def test_neuron_fire_counter(self):
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n2 = b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        for s in list(getattr(n2, "outgoing", [])):
            if s.target is b.get_neuron((0.0, 0.0)):
                b.remove_synapse(s)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        w = self.Wanderer(b, seed=0)
        stats = w.walk(max_steps=3, start=n1, lr=1e-2)
        print("neuron fire counter:", w.neuron_fire_count)
        self.assertEqual(w.neuron_fire_count, stats["steps"])

    def test_backward_request_turns_synapse_bidirectional(self):
        from marble.marblemain import register_wanderer_type, Wanderer, Brain

        class ForceBackwardPlugin:
            def choose_next(self, wanderer, current, choices):
                if not choices:
                    return None, "forward"
                syn, _ = choices[0]
                return syn, "backward"

        register_wanderer_type("force_backward_once", ForceBackwardPlugin())

        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n2 = b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0), direction="uni")
        syn = b.connect((0.0, 0.0), (1.0, 0.0), direction="uni")

        w = Wanderer(b, type_name="force_backward_once", seed=0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats = w.walk(max_steps=2, start=n1, lr=1e-2)
        output = buf.getvalue()
        print("force backward stats:", stats)
        print("force backward output:", output.strip())

        self.assertEqual(syn.direction, "bi")
        self.assertIn(
            "tried to backward transmit on a synapse that does not allow it, changed the synapse to be bi directional",
            output,
        )
        self.assertGreaterEqual(stats["steps"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
