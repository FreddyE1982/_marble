import unittest


class TestDataPair(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import UniversalTensorCodec, DataPair, REPORTER

        self.Codec = UniversalTensorCodec
        self.DataPair = DataPair
        self.reporter = REPORTER

    def test_encode_decode_pair(self):
        codec = self.Codec()
        left = {"id": 123, "name": "alpha", "vals": [1, 2, 3]}
        right = (b"\x00\x01\x02", {"nested": (True, None, 3.14)})
        dp = self.DataPair(left, right)
        enc_l, enc_r = dp.encode(codec)
        # print info about encoded token counts
        try:
            ln_l = int(enc_l.numel()) if hasattr(enc_l, "numel") else len(enc_l)
        except Exception:
            ln_l = -1
        try:
            ln_r = int(enc_r.numel()) if hasattr(enc_r, "numel") else len(enc_r)
        except Exception:
            ln_r = -1
        print("datapair encoded tokens:", ln_l, ln_r)

        back = self.DataPair.decode((enc_l, enc_r), codec)
        print("datapair decoded types:", type(back.left).__name__, type(back.right).__name__)
        self.assertEqual(back.left, left)
        self.assertEqual(back.right, right)

        # Reporter usage for auditability
        self.reporter.item["last_pair_types", "datapair", "tests"] = {
            "left": type(left).__name__,
            "right": type(right).__name__,
        }
        logged = self.reporter.item("last_pair_types", "datapair", "tests")
        print("reporter datapair test log:", logged)
        self.assertIsInstance(logged, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)

