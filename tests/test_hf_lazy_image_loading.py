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
            # Download happens during iteration; no additional calls on access.
            self.assertEqual(dl_mock.call_count, 1)
            self.assertNotIsInstance(sample.get_raw("image"), str)
            encoded = sample["image"]
            self.assertEqual(dl_mock.call_count, 1)
            self.assertNotIsInstance(encoded, str)
            decoded = wrapper._codec.decode(encoded)
            with open(path, "rb") as f:
                self.assertEqual(decoded, f.read())
        os.unlink(path)

    def test_multiple_image_fields(self):
        png1 = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBgE9TkwAAAABJRU5ErkJggg=="
        )
        png2 = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAADElEQVR42mP8/xcAAwMBgD+pKj8AAAAASUVORK5CYII="
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
            t1.write(png1)
            p1 = t1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
            t2.write(png2)
            p2 = t2.name

        class DS(DummyDataset):
            def __init__(self, path1, path2):
                self.features = {"image1": Image(), "image2": Image()}
                self._p1 = path1
                self._p2 = path2

            def __iter__(self):
                yield {"image1": {"url": self._p1}, "image2": {"url": self._p2}}

        class Mod(DummyDSModule):
            def __init__(self, p1, p2):
                self._p1 = p1
                self._p2 = p2

            def load_dataset(self, **kwargs):
                return DS(self._p1, self._p2)

        dummy_mod = Mod(p1, p2)
        with mock.patch.object(
            hf_utils.HFStreamingDatasetWrapper,
            "_download_image",
            autospec=True,
            wraps=hf_utils.HFStreamingDatasetWrapper._download_image,
        ) as dl_mock:
            with mock.patch.object(hf_utils, "_ensure_hf_imports", return_value=(None, dummy_mod)):
                wrapper = hf_utils.load_hf_streaming_dataset("dummy", streaming="memory_lazy_images")
            sample = next(iter(wrapper))
            self.assertEqual(dl_mock.call_count, 2)
            img1 = sample["image1"]
            img2 = sample["image2"]
            self.assertEqual(dl_mock.call_count, 2)
            dec1 = wrapper._codec.decode(img1)
            dec2 = wrapper._codec.decode(img2)
            with open(p1, "rb") as f1, open(p2, "rb") as f2:
                self.assertEqual(dec1, f1.read())
                self.assertEqual(dec2, f2.read())
        os.unlink(p1)
        os.unlink(p2)


if __name__ == "__main__":
    unittest.main()

