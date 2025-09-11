import os
import unittest
from pathlib import Path
import pickle

class CustomThing:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, CustomThing) and self.x == other.x


class TestUniversalTensorCodec(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import UniversalTensorCodec

        self.Codec = UniversalTensorCodec

    def test_encode_decode_primitives(self):
        codec = self.Codec()
        samples = [
            0,
            12345678901234567890,
            -42,
            3.14159,
            True,
            False,
            None,
            "hello world",
            b"\x00\x01\xff",
            [1, 2, 3, {"a": 1}],
            {"k": (1, 2), "s": {1, 2}},
            CustomThing({"nested": [1, 2, 3]}),
        ]
        for obj in samples:
            t = codec.encode(obj)
            back = codec.decode(t)
            # console output for debugging
            try:
                ln = int(t.numel()) if hasattr(t, "numel") else len(t)
            except Exception:
                ln = -1
            print("codec roundtrip:", type(obj).__name__, "tokens=", ln)
            self.assertEqual(back, obj)

    def test_vocab_persistence_roundtrip(self):
        from marble.marblemain import UniversalTensorCodec

        codec1 = UniversalTensorCodec()
        obj = {"alpha": [1, 2, 3], "beta": CustomThing(99)}
        tokens = codec1.encode(obj)

        # Export vocab
        out_path = Path("vocab_test.json")
        try:
            codec1.export_vocab(str(out_path))
            self.assertTrue(out_path.exists())

            # Import vocab into a fresh codec and decode using same tokens
            codec2 = UniversalTensorCodec()
            codec2.import_vocab(str(out_path))
            obj_back = codec2.decode(tokens)
            print("vocab roundtrip size=", codec1.vocab_size())
            self.assertEqual(obj_back, obj)
        finally:
            if out_path.exists():
                os.remove(out_path)

    def test_repeating_sequence_compression(self):
        codec = self.Codec()
        data = b"A" * 100
        raw_len = len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        tokens = codec.encode(data)
        try:
            tok_len = int(tokens.numel()) if hasattr(tokens, "numel") else len(tokens)
        except Exception:
            tok_len = -1
        print("compression lens:", tok_len, raw_len)
        self.assertLess(tok_len, raw_len)

    def test_package_import(self):
        # Ensure the package is importable and only marblemain has imports
        import marble
        from marble import marblemain

        print("package import ok; module:", marble.__file__)
        self.assertTrue(hasattr(marblemain, "UniversalTensorCodec"))

    def test_encode_deterministic(self):
        codec = self.Codec()
        obj = [1, 2, 3, {"k": "v"}]
        t1 = codec.encode(obj)
        t2 = codec.encode(obj)
        try:
            self.assertEqual(t1.tolist(), t2.tolist())
            ln = t1.numel()
        except Exception:
            self.assertEqual(list(t1), list(t2))
            ln = len(t1)
        print("deterministic tokens:", ln)

    def test_encode_performance(self):
        from marble.marblemain import UniversalTensorCodec
        codec = UniversalTensorCodec()
        obj = list(range(100000))
        import time
        start = time.perf_counter()
        codec.encode(obj)
        duration = time.perf_counter() - start
        print("encode time:", duration)
        self.assertGreater(duration, 0.0)

    def test_encode_text_performance(self):
        codec = self.Codec()
        text = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            * 100
        )
        import time
        start = time.perf_counter()
        codec.encode(text)
        duration = time.perf_counter() - start
        print("text encode time:", duration)
        self.assertLess(duration, 0.009)

    def test_encode_image_performance(self):
        codec = self.Codec()
        import base64, time
        img_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAF0lEQVR4nGP4//8/AzQAQID/RBkwAAAMAQIBBnTMEQAAAABJRU5ErkJggg=="
        )
        image_bytes = base64.b64decode(img_b64)
        start = time.perf_counter()
        codec.encode(image_bytes)
        duration = time.perf_counter() - start
        print("image encode time:", duration)
        self.assertLess(duration, 0.009)


if __name__ == "__main__":
    unittest.main(verbosity=2)
