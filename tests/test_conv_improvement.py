import unittest


class TestConvImprovement(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import (
            Brain,
            Wanderer,
            register_wanderer_type,
            wire_param_synapses,
            wire_data_synapses,
        )
        self.Brain = Brain
        self.Wanderer = Wanderer
        self.wire_params = wire_param_synapses
        self.wire_data = wire_data_synapses

        class DeterministicPlugin:
            def choose_next(self, wanderer, current, choices):
                # Always choose the first available option deterministically
                return choices[0]

        register_wanderer_type("det_conv", DeterministicPlugin())

    def _build_params(self, b, kernel_vals, stride=1, padding=0, dilation=1, bias=0.0):
        def add_free(tensor):
            for idx in b.available_indices():
                if b.get_neuron(idx) is None:
                    return b.add_neuron(idx, tensor=tensor)
            # As fallback use the first index (should not happen in small tests)
            return b.add_neuron(b.available_indices()[0], tensor=tensor)

        p = []
        p.append(add_free(list(kernel_vals)))
        p.append(add_free([float(stride)]))
        p.append(add_free([float(padding)]))
        p.append(add_free([float(dilation)]))
        p.append(add_free([float(bias)]))
        return p

    def _loss_after_walk(self, w, steps, start):
        stats = w.walk(max_steps=steps, start=start, lr=1e-2)
        return stats["loss"], stats

    def test_conv1d_improves_loss(self):
        from marble.marblemain import register_neuron_type
        # Brain setup
        b = self.Brain(2, size=(8, 8))
        avail = b.available_indices()
        # Data neuron with non-zero values
        data = b.add_neuron(avail[0], tensor=[1.0, 2.0, 3.0])
        # Destination neuron
        dst = b.add_neuron(avail[1], tensor=[0.0])
        # Base alternative neuron
        base = b.add_neuron(avail[2], tensor=[0.0])
        # Path baseline: data -> base -> dst
        b.connect(avail[0], avail[2], direction="uni")
        b.connect(avail[2], avail[1], direction="uni")

        # Build conv1d that outputs strictly zero using kernel=[0,0,0], bias=0, stride=1, pad=0, dil=1
        params = self._build_params(b, [0.0, 0.0, 0.0], 1, 0, 1, 0.0)
        # Pick a free index for conv
        conv = None
        for idx in avail:
            if b.get_neuron(idx) is None:
                conv = b.add_neuron(idx, tensor=[0.0])
                break
        if conv is None:
            conv = b.add_neuron(avail[3], tensor=[0.0])
        # Wire PARAMS and DATA, then outgoing
        self.wire_params(b, conv, params)
        self.wire_data(b, conv, [data])
        b.connect(getattr(conv, "position"), avail[1], direction="uni")
        # Promote to conv1d
        conv.type_name = "conv1d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("conv1d")
        if hasattr(plug, "on_init"):
            plug.on_init(conv)

        # Ensure deterministic wandering: create two synapses out of data (conv first, then base)
        # Already wired conv as the first connect? If needed, ensure order: connect conv first
        # Rewire ordering by reconnecting base edge second
        # (We already connected baseline before conv; add another base edge to ensure at least one choice exists after conv)
        # Build wanderer and measure losses
        w = self.Wanderer(b, type_name="det_conv", seed=1)
        # Start at data; det plugin picks first outgoing, which should be data->base originally; ensure conv is chosen
        # To ensure conv first, we add conv edge last then recreate base edge so conv is first in list
        # Clear and reconnect deterministically
        # For simplicity in this controlled test, create an extra data2->conv order if needed

        # Loss with conv path: we want the first edge to be data->conv
        # We'll rebuild the outgoing list by removing and re-adding base edge after conv
        # Remove base edge and add again so conv appears first
        # Find base edge
        base_edge = None
        for s in list(data.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            # Now add it again so it comes after the conv edge
            b.connect(avail[0], avail[2], direction="uni")

        loss_conv, _ = self._loss_after_walk(w, 2, start=data)

        # Now remove conv path to force base-only path
        # Remove conv outgoing and data->conv
        for s in list(conv.incoming):
            try:
                b.remove_synapse(s)
            except Exception:
                pass
        for s in list(data.outgoing):
            if s.target is conv:
                b.remove_synapse(s)
        for s in list(conv.outgoing):
            b.remove_synapse(s)
        # Reset wanderer
        w2 = self.Wanderer(b, type_name="det_conv", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 2, start=data)

        print("conv1d loss:", loss_conv, "base loss:", loss_base)
        self.assertLess(loss_conv, loss_base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
