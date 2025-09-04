import unittest


class TestConv2D3DImprovement(unittest.TestCase):
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
                return choices[0]

        register_wanderer_type("det_conv_nd", DeterministicPlugin())

    def _add_free(self, b, tensor, connect_to=None, direction="bi", keep_connection=True):
        for idx in b.available_indices():
            if b.get_neuron(idx) is None:
                if connect_to is None:
                    connect = next(iter(b.neurons.keys())) if b.neurons else None
                else:
                    connect = getattr(connect_to, "position", connect_to)
                n = b.add_neuron(idx, tensor=tensor, connect_to=connect, direction=direction)
                if not keep_connection:
                    for s in list(n.incoming) + list(n.outgoing):
                        b.remove_synapse(s)
                return n
        if connect_to is None:
            connect = next(iter(b.neurons.keys())) if b.neurons else None
        else:
            connect = getattr(connect_to, "position", connect_to)
        n = b.add_neuron(b.available_indices()[0], tensor=tensor, connect_to=connect, direction=direction)
        if not keep_connection:
            for s in list(n.incoming) + list(n.outgoing):
                b.remove_synapse(s)
        return n

    def _build_params(self, b, kernel_vals, stride=1, padding=0, dilation=1, bias=0.0):
        p = []
        p.append(self._add_free(b, list(kernel_vals), keep_connection=False))
        p.append(self._add_free(b, [float(stride)], keep_connection=False))
        p.append(self._add_free(b, [float(padding)], keep_connection=False))
        p.append(self._add_free(b, [float(dilation)], keep_connection=False))
        p.append(self._add_free(b, [float(bias)], keep_connection=False))
        return p

    def _loss_after_walk(self, w, steps, start):
        stats = w.walk(max_steps=steps, start=start, lr=1e-2)
        return stats["loss"], stats

    def test_conv2d_improves_loss(self):
        b = self.Brain(2, size=(10, 10))
        avail = b.available_indices()
        # Build 2 data rows to form 2x3 input
        row1 = self._add_free(b, [1.0, 2.0, 3.0], keep_connection=False)
        row2 = self._add_free(b, [2.0, 3.0, 4.0], keep_connection=False)
        dst = self._add_free(b, [0.0], keep_connection=False)
        base = self._add_free(b, [0.0], keep_connection=False)
        # Baseline path
        b.connect(getattr(row1, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        # Conv2d with zero kernel (2x2 zeros) and zero bias, stride=1, pad=0, dil=1
        params = self._build_params(b, [0.0, 0.0, 0.0, 0.0], 1, 0, 1, 0.0)
        conv = self._add_free(b, [0.0], connect_to=getattr(dst, "position"), direction="uni")
        self.wire_params(b, conv, params)
        self.wire_data(b, conv, [row1, row2])
        # Promote to conv2d
        conv.type_name = "conv2d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("conv2d")
        if hasattr(plug, "on_init"):
            plug.on_init(conv)

        # Ensure conv path is chosen first
        # Remove and re-add base edge so conv edge appears first
        base_edge = None
        for s in list(row1.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(row1, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_conv_nd", seed=1)
        loss_conv, _ = self._loss_after_walk(w, 3, start=row1)

        # Remove conv edges and test base path
        for s in list(conv.incoming):
            b.remove_synapse(s)
        for s in list(conv.outgoing):
            b.remove_synapse(s)
        for s in list(row1.outgoing):
            if s.target is conv:
                b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_conv_nd", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 3, start=row1)

        print("conv2d loss:", loss_conv, "base loss:", loss_base)
        self.assertLess(loss_conv, loss_base)

    def test_conv3d_improves_loss(self):
        b = self.Brain(2, size=(12, 12))
        avail = b.available_indices()
        # Build 2 slices, each 2x2 (flattened): total volume 2x2x2 when stacked
        slice1 = self._add_free(b, [1.0, 2.0, 3.0, 4.0], keep_connection=False)
        slice2 = self._add_free(b, [2.0, 3.0, 4.0, 5.0], keep_connection=False)
        dst = self._add_free(b, [0.0], keep_connection=False)
        base = self._add_free(b, [0.0], keep_connection=False)
        # Baseline
        b.connect(getattr(slice1, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        # Conv3d with zero kernel cube (2x2x2 zeros) and zero bias
        params = self._build_params(b, [0.0] * 8, 1, 0, 1, 0.0)
        conv = self._add_free(b, [0.0], connect_to=getattr(dst, "position"), direction="uni")
        self.wire_params(b, conv, params)
        self.wire_data(b, conv, [slice1, slice2])
        conv.type_name = "conv3d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("conv3d")
        if hasattr(plug, "on_init"):
            plug.on_init(conv)

        # Ensure conv path is chosen first
        base_edge = None
        for s in list(slice1.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(slice1, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_conv_nd", seed=1)
        loss_conv, _ = self._loss_after_walk(w, 3, start=slice1)

        # Remove conv edges and test base path
        for s in list(conv.incoming):
            b.remove_synapse(s)
        for s in list(conv.outgoing):
            b.remove_synapse(s)
        for s in list(slice1.outgoing):
            if s.target is conv:
                b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_conv_nd", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 3, start=slice1)

        print("conv3d loss:", loss_conv, "base loss:", loss_base)
        self.assertLess(loss_conv, loss_base)


if __name__ == "__main__":
    unittest.main(verbosity=2)

