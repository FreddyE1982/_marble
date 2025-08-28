import unittest


class TestUnfoldFoldUnpoolImprovement(unittest.TestCase):
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

        register_wanderer_type("det_new", DeterministicPlugin())

    def _add_free(self, b, tensor):
        for idx in b.available_indices():
            if b.get_neuron(idx) is None:
                return b.add_neuron(idx, tensor=tensor)
        return b.add_neuron(b.available_indices()[0], tensor=tensor)

    def _loss_after_walk(self, w, steps, start):
        stats = w.walk(max_steps=steps, start=start, lr=1e-2)
        return stats["loss"], stats

    def test_unfold2d_improves_loss(self):
        from marble.marblemain import _NEURON_TYPES
        b = self.Brain(2, size=(10, 10))
        # Two rows with small values keep MSE small vs base amplified path
        row1 = self._add_free(b, [0.1, 0.1, 0.1, 0.1])
        row2 = self._add_free(b, [0.1, 0.1, 0.1, 0.1])
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        base.weight = 2.5
        b.connect(getattr(row1, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        # Params: k=2, s=2, p=0, d=1
        k = self._add_free(b, [2.0])
        s = self._add_free(b, [2.0])
        p = self._add_free(b, [0.0])
        d = self._add_free(b, [1.0])
        plugn = self._add_free(b, [0.0])
        self.wire_params(b, plugn, [k, s, p, d])
        self.wire_data(b, plugn, [row1, row2])
        b.connect(getattr(plugn, "position"), getattr(dst, "position"), direction="uni")
        plugn.type_name = "unfold2d"
        plug = _NEURON_TYPES.get("unfold2d")
        if hasattr(plug, "on_init"):
            plug.on_init(plugn)

        # Ensure plugin chosen first
        base_edge = None
        for s in list(row1.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(row1, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_new", seed=1)
        loss_plugin, _ = self._loss_after_walk(w, 1, start=row1)

        # Remove plugin path and compare
        for s in list(plugn.incoming):
            b.remove_synapse(s)
        for s in list(plugn.outgoing):
            b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_new", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 1, start=row1)
        print("unfold2d loss:", loss_plugin, "base loss:", loss_base)
        self.assertLess(loss_plugin, loss_base)

    def test_fold2d_improves_loss(self):
        from marble.marblemain import _NEURON_TYPES
        b = self.Brain(2, size=(10, 10))
        # Columns input for fold: k=1 => L should equal out_h*out_w
        cols = self._add_free(b, [0.1, 0.1, 0.1, 0.1])  # out 2x2, k=1 => 4 entries
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        base.weight = 2.5
        b.connect(getattr(cols, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        # Params: out_h=2, out_w=2, k=1, s=1, p=0, d=1
        oh = self._add_free(b, [2.0])
        ow = self._add_free(b, [2.0])
        k = self._add_free(b, [1.0])
        st = self._add_free(b, [1.0])
        pd = self._add_free(b, [0.0])
        dl = self._add_free(b, [1.0])
        fold = self._add_free(b, [0.0])
        self.wire_params(b, fold, [oh, ow, k, st, pd, dl])
        self.wire_data(b, fold, [cols])
        b.connect(getattr(fold, "position"), getattr(dst, "position"), direction="uni")
        fold.type_name = "fold2d"
        plug = _NEURON_TYPES.get("fold2d")
        if hasattr(plug, "on_init"):
            plug.on_init(fold)

        # Ensure plugin chosen first
        base_edge = None
        for s in list(cols.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(cols, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_new", seed=1)
        loss_plugin, _ = self._loss_after_walk(w, 1, start=cols)
        # Remove plugin path
        for s in list(fold.incoming):
            b.remove_synapse(s)
        for s in list(fold.outgoing):
            b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_new", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 1, start=cols)
        print("fold2d loss:", loss_plugin, "base loss:", loss_base)
        self.assertLess(loss_plugin, loss_base)

    def test_maxunpool_improves_loss(self):
        import torch
        from marble.marblemain import _NEURON_TYPES
        b = self.Brain(2, size=(10, 10))
        # Build an input and pool it with indices via torch, then unpool in plugin
        x = [0.1, 0.2, 0.1, 0.3]
        data = self._add_free(b, x)
        # Prepare pooled values and indices using torch
        xt = torch.tensor(x, dtype=torch.float32).view(1, 1, 2, 2)
        pv, idx = torch.nn.functional.max_pool2d(xt, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), return_indices=True)
        pooled_vals = pv.view(-1).tolist()
        indices = idx.view(-1).tolist()
        vals_n = self._add_free(b, pooled_vals)
        idx_n = self._add_free(b, [int(i) for i in indices])
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        base.weight = 2.5
        b.connect(getattr(data, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        # Params: k=2, s=2, p=0
        k = self._add_free(b, [2.0])
        s = self._add_free(b, [2.0])
        p = self._add_free(b, [0.0])
        up = self._add_free(b, [0.0])
        self.wire_params(b, up, [k, s, p])
        self.wire_data(b, up, [vals_n, idx_n])
        b.connect(getattr(up, "position"), getattr(dst, "position"), direction="uni")
        up.type_name = "maxunpool2d"
        plug = _NEURON_TYPES.get("maxunpool2d")
        if hasattr(plug, "on_init"):
            plug.on_init(up)

        # Ensure plugin chosen first
        base_edge = None
        for s in list(data.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(data, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_new", seed=1)
        loss_plugin, _ = self._loss_after_walk(w, 1, start=vals_n)
        # Remove plugin path
        for s in list(up.incoming):
            b.remove_synapse(s)
        for s in list(up.outgoing):
            b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_new", seed=1)
        loss_base, _ = self._loss_after_walk(w2, 1, start=data)
        print("maxunpool2d loss:", loss_plugin, "base loss:", loss_base)
        self.assertLess(loss_plugin, loss_base)


if __name__ == "__main__":
    unittest.main(verbosity=2)

