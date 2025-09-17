import tempfile
import unittest
from pathlib import Path

import yaml

from marble.learnables_yaml import learnablesOFF, learnablesON, updatelearnablesyaml


class UpdateLearnablesYamlTests(unittest.TestCase):
    def test_updatelearnablesyaml_populates_known_learnables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "learnables.yaml"
            updatelearnablesyaml(yaml_path)

            with yaml_path.open("r", encoding="utf8") as fh:
                data = yaml.safe_load(fh) or {}

        self.assertTrue("Wanderer.swish_beta" in data)
        self.assertTrue("Wanderer.autoplugin_bias_gravitywell" in data)
        self.assertTrue("Wanderer.autoneuron_gain_base" in data)
        self.assertTrue("Wanderer.dyn_query" in data)
        self.assertTrue("Wanderer.latent_vector" in data)
        self.assertGreater(len(data), 100)

    def test_learnablesoff_turns_everything_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "learnables.yaml"

            learnablesOFF(yaml_path)

            with yaml_path.open("r", encoding="utf8") as fh:
                data = yaml.safe_load(fh) or {}

        self.assertTrue(data)
        self.assertTrue(all(value == "OFF" for value in data.values()))

    def test_learnableson_enables_non_loss_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "learnables.yaml"

            learnablesOFF(yaml_path)
            learnablesON(yaml_path)

            with yaml_path.open("r", encoding="utf8") as fh:
                data = yaml.safe_load(fh) or {}

        non_loss = {key for key in data if "loss" not in key.lower()}
        loss_related = {key for key in data if "loss" in key.lower()}

        self.assertTrue(non_loss)
        self.assertTrue(loss_related)
        self.assertTrue(all(data[key] == "ON" for key in non_loss))
        self.assertTrue(all(data[key] == "OFF" for key in loss_related))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
