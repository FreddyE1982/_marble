import base64
import os
import tempfile
import unittest
from unittest import mock

import marble.hf_utils as hf_utils


class Image:
    pass


class DummyDataset:
    def __init__(self, path):
        self.features = {"image": Image()}
        self._path = path

    def __iter__(self):
        yield {"image": {"url": self._path}}


class DummyDSModule:
    def __init__(self, path):
        self._path = path

    def load_dataset(self, **kwargs):
        return DummyDataset(self._path)


class HFLazyImageLoadingTest(unittest.TestCase):
    def test_lazy_image_download_and_encode(self):
        png_bytes = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBgE9TkwAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(png_bytes)
            path = tmp.name
        dummy_mod = DummyDSModule(path)
        with mock.patch.object(
            hf_utils.HFStreamingDatasetWrapper,
            "_download_image",
            autospec=True,
            wraps=hf_utils.HFStreamingDatasetWrapper._download_image,
        ) as dl_mock:
            with mock.patch.object(hf_utils, "_ensure_hf_imports", return_value=(None, dummy_mod)):
                wrapper = hf_utils.load_hf_streaming_dataset("dummy", streaming="memory_lazy_images")
            sample = next(iter(wrapper))
            self.assertEqual(dl_mock.call_count, 0)
            self.assertEqual(sample.get_raw("image"), path)
            encoded = sample["image"]
            self.assertEqual(dl_mock.call_count, 1)
            self.assertNotIsInstance(encoded, str)
            decoded = wrapper._codec.decode(encoded)
            with open(path, "rb") as f:
                self.assertEqual(decoded, f.read())
            self.assertNotIsInstance(sample.get_raw("image"), str)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()

