import unittest


class TestConvTransposeImprovement(unittest.TestCase):
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

        register_wanderer_type("det_deconv", DeterministicPlugin())

    def _add_free(self, b, tensor):
        for idx in b.available_indices():
            if b.get_neuron(idx) is None:
                return b.add_neuron(idx, tensor=tensor)
        return b.add_neuron(b.available_indices()[0], tensor=tensor)

    def _build_params(self, b, kernel_vals, stride=1, padding=0, dilation=1, bias=0.0):
        p = []
        p.append(self._add_free(b, list(kernel_vals)))
        p.append(self._add_free(b, [float(stride)]))
        p.append(self._add_free(b, [float(padding)]))
        p.append(self._add_free(b, [float(dilation)]))
        p.append(self._add_free(b, [float(bias)]))
        return p

    def _loss_after_walk(self, w, steps, start):
        stats = w.walk(max_steps=steps, start=start, lr=1e-2)
        return stats["loss"], stats

    def test_conv_transpose_1d_improves_loss(self):
        b = self.Brain(2, size=(8, 8))
        data = self._add_free(b, [1.0, 2.0, 3.0])
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        b.connect(getattr(data, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        params = self._build_params(b, [0.0, 0.0, 0.0], 1, 0, 1, 0.0)
        conv = self._add_free(b, [0.0])
        self.wire_params(b, conv, params)
        self.wire_data(b, conv, [data])
        b.connect(getattr(conv, "position"), getattr(dst, "position"), direction="uni")
        conv.type_name = "conv_transpose1d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("conv_transpose1d")
        if hasattr(plug, "on_init"):
            plug.on_init(conv)

        # Ensure conv chosen first
        base_edge = None
        for s in list(data.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(data, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_deconv", seed=1)
        loss_conv, _ = self._loss_after_walk(w, 2, start=data)

        # Remove conv path and compare
        for s in list(conv.incoming):
            b.remove_synapse(s)
        for s in list(conv.outgoing):
            b.remove_synapse(s)
        for s in list(data.outgoing):
            if s.target is conv:
                b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_deconv", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 2, start=data)

        print("convT1d loss:", loss_conv, "base loss:", loss_base)
        self.assertLess(loss_conv, loss_base)

    def test_conv_transpose_2d_improves_loss(self):
        b = self.Brain(2, size=(10, 10))
        row1 = self._add_free(b, [1.0, 2.0, 3.0])
        row2 = self._add_free(b, [2.0, 3.0, 4.0])
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        b.connect(getattr(row1, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        params = self._build_params(b, [0.0, 0.0, 0.0, 0.0], 1, 0, 1, 0.0)
        conv = self._add_free(b, [0.0])
        self.wire_params(b, conv, params)
        self.wire_data(b, conv, [row1, row2])
        b.connect(getattr(conv, "position"), getattr(dst, "position"), direction="uni")
        conv.type_name = "conv_transpose2d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("conv_transpose2d")
        if hasattr(plug, "on_init"):
            plug.on_init(conv)

        base_edge = None
        for s in list(row1.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(row1, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_deconv", seed=1)
        loss_conv, _ = self._loss_after_walk(w, 3, start=row1)

        for s in list(conv.incoming):
            b.remove_synapse(s)
        for s in list(conv.outgoing):
            b.remove_synapse(s)
        for s in list(row1.outgoing):
            if s.target is conv:
                b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_deconv", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 3, start=row1)

        print("convT2d loss:", loss_conv, "base loss:", loss_base)
        self.assertLess(loss_conv, loss_base)

    def test_conv_transpose_3d_improves_loss(self):
        b = self.Brain(2, size=(12, 12))
        slice1 = self._add_free(b, [1.0, 2.0, 3.0, 4.0])
        slice2 = self._add_free(b, [2.0, 3.0, 4.0, 5.0])
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        b.connect(getattr(slice1, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        params = self._build_params(b, [0.0] * 8, 1, 0, 1, 0.0)
        conv = self._add_free(b, [0.0])
        self.wire_params(b, conv, params)
        self.wire_data(b, conv, [slice1, slice2])
        b.connect(getattr(conv, "position"), getattr(dst, "position"), direction="uni")
        conv.type_name = "conv_transpose3d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("conv_transpose3d")
        if hasattr(plug, "on_init"):
            plug.on_init(conv)

        base_edge = None
        for s in list(slice1.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(slice1, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_deconv", seed=1)
        loss_conv, _ = self._loss_after_walk(w, 3, start=slice1)

        for s in list(conv.incoming):
            b.remove_synapse(s)
        for s in list(conv.outgoing):
            b.remove_synapse(s)
        for s in list(slice1.outgoing):
            if s.target is conv:
                b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_deconv", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 3, start=slice1)

        print("convT3d loss:", loss_conv, "base loss:", loss_base)
        self.assertLess(loss_conv, loss_base)


if __name__ == "__main__":
    unittest.main(verbosity=2)

