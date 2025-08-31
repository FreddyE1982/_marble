import importlib
import inspect
import unittest

import torch

from marble.marblemain import Brain, Wanderer, SelfAttention, attach_selfattention
from marble.reporter import REPORTER, clear_report_group


class SelfAttentionMetricsIntegrationTests(unittest.TestCase):
    def make_wanderer(self):
        b = Brain(1, size=1)
        idx = next(iter(b.available_indices()))
        b.add_neuron(idx, tensor=[0.0])
        return Wanderer(b, seed=1)

    def test_all_plugins_log_metrics(self):
        import marble.plugins as plugins

        for mod in plugins.__all__:
            if not mod.startswith("selfattention_"):
                continue
            module = importlib.import_module(f"marble.plugins.{mod}")
            cls = None
            for obj in module.__dict__.values():
                if inspect.isclass(obj) and obj.__name__.endswith(("Plugin", "Routine")):
                    cls = obj
                    break
            if cls is None:
                continue
            with self.subTest(plugin=mod):
                clear_report_group("selfattention")
                w = self.make_wanderer()
                sa = SelfAttention(routines=[cls()])
                attach_selfattention(w, sa)
                ctx = {
                    "cur_loss_tensor": torch.tensor(1.0),
                    "sa_loss": 1.0,
                    "sa_loss_speed": 0.1,
                    "sa_loss_accel": 0.01,
                    "sa_model_complexity": 5,
                }
                sa._routines[0].after_step(sa, sa._reporter_ro, w, 0, ctx)
                data = REPORTER.group("selfattention", "metrics")
                print(mod, data)
                self.assertTrue(any(k.endswith("_metrics") for k in data))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

