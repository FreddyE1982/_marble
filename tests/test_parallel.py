import unittest


class TestParallelWanderers(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, UniversalTensorCodec, make_datapair
        self.Brain = Brain
        self.Codec = UniversalTensorCodec
        self.make_dp = make_datapair

    def test_thread_parallel(self):
        from marble.marblemain import run_wanderers_parallel
        b = self.Brain(2, size=(6, 6))
        codec = self.Codec()
        ds = [
            [self.make_dp({"i": i}, float(i % 2)) for i in range(3)]
        ]
        datasets = [ds[0], ds[0]]
        pre_brain_id = id(b)
        pre_graph_id = id(b.neurons)
        results = run_wanderers_parallel(
            b,
            datasets,
            codec,
            mode="thread",
            steps_per_pair=2,
            lr=1e-2,
            loss="nn.MSELoss",
        )
        print("parallel thread results count:", len(results))
        self.assertEqual(len(results), 2)
        self.assertEqual(id(b), pre_brain_id)
        self.assertEqual(id(b.neurons), pre_graph_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
