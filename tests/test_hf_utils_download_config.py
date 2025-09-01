import unittest
from unittest import mock

import marble.hf_utils as hf_utils


class DummyDS:
    def load_dataset(self, **kwargs):
        self.kwargs = kwargs
        return []


class HFUtilsDownloadConfigTest(unittest.TestCase):
    def test_download_config_forwarded(self):
        dummy = DummyDS()
        with mock.patch.object(hf_utils, "_ensure_hf_imports", return_value=(None, dummy)):
            dc = object()
            wrapper = hf_utils.load_hf_streaming_dataset("dummy", download_config=dc)
        print("forwarded_download_config", dummy.kwargs.get("download_config") is dc)
        self.assertIs(dummy.kwargs.get("download_config"), dc)
        self.assertIsInstance(wrapper, hf_utils.HFStreamingDatasetWrapper)


if __name__ == "__main__":
    unittest.main()

