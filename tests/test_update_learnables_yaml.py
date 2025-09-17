import tempfile
import unittest
from pathlib import Path

import yaml

from marble.learnables_yaml import updatelearnablesyaml


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
