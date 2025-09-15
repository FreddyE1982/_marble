import gc
import unittest

from marble.marblemain import Brain, UniversalTensorCodec, run_training_with_datapairs, make_datapair


deleted = []


class Payload:
    def __init__(self, v):
        self.v = v

    def __del__(self):
        deleted.append(self.v)


class StreamingMemoryCleanupTest(unittest.TestCase):
    def test_processed_pairs_released(self):
        def gen():
            for i in range(5):
                yield make_datapair(Payload(i), Payload(i + 1))

        b = Brain(2, size=(1, 1))
        codec = UniversalTensorCodec()
        run_training_with_datapairs(b, gen(), codec, steps_per_pair=1, lr=1e-2, streaming=True)
        gc.collect()
        self.assertEqual(len(deleted), 10)


if __name__ == "__main__":
    unittest.main()
