import unittest


class TestMaxPoolImprovement(unittest.TestCase):
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

        register_wanderer_type("det_pool", DeterministicPlugin())

    def _add_free(self, b, tensor):
        for idx in b.available_indices():
            if b.get_neuron(idx) is None:
                if b.neurons:
                    first_idx = next(iter(b.neurons.keys()))
                    n = b.add_neuron(idx, tensor=tensor, connect_to=first_idx)
                    for s in list(getattr(n, "outgoing", [])):
                        if s.target is b.get_neuron(first_idx):
                            b.remove_synapse(s)
                    return n
                return b.add_neuron(idx, tensor=tensor)
        first_idx = next(iter(b.neurons.keys()))
        n = b.add_neuron(b.available_indices()[0], tensor=tensor, connect_to=first_idx)
        for s in list(getattr(n, "outgoing", [])):
            if s.target is b.get_neuron(first_idx):
                b.remove_synapse(s)
        return n

    def _loss_after_walk(self, w, steps, start):
        stats = w.walk(max_steps=steps, start=start, lr=1e-2)
        return stats["loss"], stats

    def test_maxpool1d_improves_loss(self):
        b = self.Brain(2, size=(8, 8))
        data = self._add_free(b, [1.0, 2.0, 3.0])
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        base.weight = 2.5  # amplify base path to make pooling comparatively better
        b.connect(getattr(data, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        # Params: kernel=2, stride=2, padding=0 => reduces to [2]
        k = self._add_free(b, [2.0])
        s = self._add_free(b, [2.0])
        p = self._add_free(b, [0.0])
        pool = self._add_free(b, [0.0])
        self.wire_params(b, pool, [k, s, p])
        self.wire_data(b, pool, [data])
        b.connect(getattr(pool, "position"), getattr(dst, "position"), direction="uni")
        pool.type_name = "maxpool1d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("maxpool1d")
        if hasattr(plug, "on_init"):
            plug.on_init(pool)

        # Ensure pool chosen first
        base_edge = None
        for s in list(data.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(data, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_pool", seed=1)
        loss_pool, stats_pool = self._loss_after_walk(w, 1, start=data)

        # Remove pool path and compare
        for s in list(pool.incoming):
            b.remove_synapse(s)
        for s in list(pool.outgoing):
            b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_pool", seed=1)
        loss_base, stats_base = self._loss_after_walk(w2, 1, start=data)

        print("maxpool1d loss:", loss_pool, "base loss:", loss_base)
        self.assertLess(loss_pool, loss_base)

    def test_maxpool2d_improves_loss(self):
        b = self.Brain(2, size=(10, 10))
        row1 = self._add_free(b, [1.0, 2.0, 3.0, 4.0])
        row2 = self._add_free(b, [1.0, 2.0, 3.0, 4.0])
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        base.weight = 2.5
        b.connect(getattr(row1, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        # Params: kernel=2, stride=2, padding=0 => reduces 2x4 rows into pooled of size 1x2
        k = self._add_free(b, [2.0])
        s = self._add_free(b, [2.0])
        p = self._add_free(b, [0.0])
        pool = self._add_free(b, [0.0])
        self.wire_params(b, pool, [k, s, p])
        self.wire_data(b, pool, [row1, row2])
        b.connect(getattr(pool, "position"), getattr(dst, "position"), direction="uni")
        pool.type_name = "maxpool2d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("maxpool2d")
        if hasattr(plug, "on_init"):
            plug.on_init(pool)

        base_edge = None
        for s in list(row1.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(row1, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_pool", seed=1)
        loss_pool, stats_pool = self._loss_after_walk(w, 1, start=row1)
        # Remove pool path
        for s in list(pool.incoming):
            b.remove_synapse(s)
        for s in list(pool.outgoing):
            b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_pool", seed=1)
        loss_base, stats_base = self._loss_after_walk(w2, 1, start=row1)
        print("maxpool2d loss:", loss_pool, "base loss:", loss_base)
        self.assertLess(loss_pool, loss_base)

    def test_maxpool3d_improves_loss(self):
        b = self.Brain(2, size=(12, 12))
        slice1 = self._add_free(b, [1.0, 2.0, 3.0, 4.0])
        slice2 = self._add_free(b, [1.0, 2.0, 3.0, 4.0])
        dst = self._add_free(b, [0.0])
        base = self._add_free(b, [0.0])
        base.weight = 2.5
        b.connect(getattr(slice1, "position"), getattr(base, "position"), direction="uni")
        b.connect(getattr(base, "position"), getattr(dst, "position"), direction="uni")

        # Params: kernel=2, stride=2, padding=0 on 2 slices of 2x2 => 1x1x2 pooled
        k = self._add_free(b, [2.0])
        s = self._add_free(b, [2.0])
        p = self._add_free(b, [0.0])
        pool = self._add_free(b, [0.0])
        self.wire_params(b, pool, [k, s, p])
        self.wire_data(b, pool, [slice1, slice2])
        b.connect(getattr(pool, "position"), getattr(dst, "position"), direction="uni")
        pool.type_name = "maxpool3d"
        from marble.marblemain import _NEURON_TYPES  # noqa: E402
        plug = _NEURON_TYPES.get("maxpool3d")
        if hasattr(plug, "on_init"):
            plug.on_init(pool)

        base_edge = None
        for s in list(slice1.outgoing):
            if s.target is base:
                base_edge = s
                break
        if base_edge is not None:
            b.remove_synapse(base_edge)
            b.connect(getattr(slice1, "position"), getattr(base, "position"), direction="uni")

        w = self.Wanderer(b, type_name="det_pool", seed=1)
        loss_pool, stats_pool = self._loss_after_walk(w, 1, start=slice1)
        for s in list(pool.incoming):
            b.remove_synapse(s)
        for s in list(pool.outgoing):
            b.remove_synapse(s)
        w2 = self.Wanderer(b, type_name="det_pool", seed=1)
        loss_base, stats_base = self._loss_after_walk(w2, 1, start=slice1)
        print("maxpool3d loss:", loss_pool, "base loss:", loss_base)
        self.assertLess(loss_pool, loss_base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
