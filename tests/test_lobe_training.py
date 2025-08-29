import unittest

from marble.marblemain import Brain, run_wanderer_training


class LobeTrainingTests(unittest.TestCase):
    def test_lobe_independent_training_and_plugins(self):
        b = Brain(1, size=4)
        n0 = b.add_neuron((0,), tensor=1.0)
        n1 = b.add_neuron((1,), tensor=0.0)
        n2 = b.add_neuron((2,), tensor=1.0)
        n3 = b.add_neuron((3,), tensor=0.0)
        s01 = b.connect((0,), (1,))
        s23 = b.connect((2,), (3,))

        lobe1 = b.define_lobe("first", [n0, n1], [s01])

        w0_before = float(n0.weight)
        w2_before = float(n2.weight)

        res1 = run_wanderer_training(
            b,
            num_walks=1,
            max_steps=1,
            lr=0.1,
            start_selector=lambda _b: n0,
            wanderer_type="epsilongreedy",
            loss=lambda outs: outs[0].sum(),
            lobe=lobe1,
        )
        print("plugins after lobe1:", res1["history"][0]["plugins"])
        self.assertIn("EpsilonGreedyChooserPlugin", res1["history"][0]["plugins"])
        self.assertNotEqual(w0_before, float(n0.weight))
        self.assertEqual(w2_before, float(n2.weight))

        lobe2 = b.define_lobe(
            "second",
            [n2, n3],
            [s23],
            inherit_plugins=False,
            plugin_types="wanderalongsynapseweights",
        )
        res2 = run_wanderer_training(
            b,
            num_walks=1,
            max_steps=1,
            lr=0.1,
            start_selector=lambda _b: n2,
            wanderer_type="epsilongreedy",
            loss=lambda outs: outs[0].sum(),
            lobe=lobe2,
        )
        print("plugins after lobe2:", res2["history"][0]["plugins"])
        self.assertIn("WanderAlongSynapseWeightsPlugin", res2["history"][0]["plugins"])
        self.assertNotIn("EpsilonGreedyChooserPlugin", res2["history"][0]["plugins"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
