import sys
import types
import unittest

from marble import learnableIsOn


if "datasets" not in sys.modules:
    class _DownloadConfig:  # pragma: no cover - lightweight stub for import
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    sys.modules["datasets"] = types.SimpleNamespace(DownloadConfig=_DownloadConfig)


from examples.run_hf_image_quality_dc import (
    QUALITY_LEARNABLES,
    enable_quality_learnables,
)


class TestQualityLearnables(unittest.TestCase):
    def test_enable_quality_learnables(self) -> None:
        enabled = enable_quality_learnables()
        missing = [name for name in QUALITY_LEARNABLES if not learnableIsOn(name)]
        self.assertTrue(
            enabled,
            "Expected at least one quality learnable to be enabled",
        )
        self.assertFalse(
            missing,
            f"Learnables were not fully enabled: {missing}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

