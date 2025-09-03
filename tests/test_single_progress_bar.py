import types


def test_single_progress_bar(monkeypatch):
    from marble.marblemain import Brain
    import marble.wanderer as wanderer
    from marble.wanderer import Wanderer

    class DummyPbar:
        instances = 0

        def __init__(self, *args, **kwargs):
            DummyPbar.instances += 1
            self.total = kwargs.get("total", 0)

        def set_description(self, *args, **kwargs):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def refresh(self):
            pass

        def write(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(wanderer, "_tqdm_factory", lambda: DummyPbar)

    b = Brain(1)
    start = b.add_neuron((0,))
    w = Wanderer(b, seed=0)
    b._progress_total_walks = 2  # type: ignore[attr-defined]
    b._progress_walk = 0  # type: ignore[attr-defined]
    w.walk(max_steps=1, start=start)
    b._progress_walk = 1  # type: ignore[attr-defined]
    w.walk(max_steps=1, start=start)

    assert DummyPbar.instances == 1
    assert getattr(w, "_pbar", None) is None
