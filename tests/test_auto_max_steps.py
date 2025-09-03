import unittest

from marble.marblemain import Brain, UniversalTensorCodec, run_training_with_datapairs, make_datapair


class TestAutoMaxSteps(unittest.TestCase):
    def test_auto_max_steps_uses_longest_path(self):
        brain = Brain(1)
        n0 = brain.add_neuron((0,), tensor=0.0)
        n1 = brain.add_neuron((1,), tensor=0.0)
        n2 = brain.add_neuron((2,), tensor=0.0)
        s01 = n0.connect_to(n1)
        brain.synapses.append(s01)
        s12 = n1.connect_to(n2)
        brain.synapses.append(s12)
        codec = UniversalTensorCodec()
        data = [make_datapair([0.0], [0.0])]
        res = run_training_with_datapairs(
            brain,
            data,
            codec,
            steps_per_pair=None,
            auto_max_steps_every=1,
            left_to_start=lambda _l, _b: n0,
            seed=0,
            streaming=False,
        )
        steps = res["history"][0]["steps"]
        self.assertEqual(steps, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
