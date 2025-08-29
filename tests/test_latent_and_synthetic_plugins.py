import unittest

import unittest

from marble.marblemain import Brain, Wanderer, REPORTER, report, clear_report_group
from marble.plugins import wanderer_latentspace  # noqa: F401
from marble.plugins import wanderer_synthetic_trainer  # noqa: F401


class TestLatentAndSyntheticPlugins(unittest.TestCase):
    def setUp(self) -> None:
        clear_report_group("tests", "plugins")

    def test_latent_space_plugin_params(self) -> None:
        b = Brain(1, size=1)
        w = Wanderer(b, type_name="latentspace", loss="nn.MSELoss")
        w.walk(max_steps=1, lr=1e-2)
        lv = w.get_learnable_param_tensor("latent_vector")
        report("tests", "latent_shape", {"shape": list(lv.shape)}, "plugins")
        print("latent shape:", list(lv.shape))
        self.assertTrue(lv.numel() >= 1)

    def test_synthetic_training_plugin_runs(self) -> None:
        b = Brain(1, size=1)
        w = Wanderer(b, type_name="synthetictrainer", loss="nn.MSELoss")
        w.walk(max_steps=1, lr=1e-2)
        done = getattr(w, "_synthetic_trained", False)
        report("tests", "synthetic_done", {"done": bool(done)}, "plugins")
        print("synthetic done:", done)
        self.assertTrue(done)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
