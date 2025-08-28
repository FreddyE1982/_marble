import unittest

from marble.marblemain import Brain, UniversalTensorCodec, run_training_with_datapairs


class TestBatchTrainingPlugin(unittest.TestCase):
    def test_batch_training_and_single_inference(self):
        codec = UniversalTensorCodec()
        brain = Brain(1)
        data = [([1.0], [2.0]), ([2.0], [4.0]), ([3.0], [6.0]), ([4.0], [8.0])]
        run_training_with_datapairs(
            brain,
            data,
            codec,
            steps_per_pair=2,
            lr=0.05,
            wanderer_type="batchtrainer",
            batch_size=2,
            streaming=False,
            seed=1,
        )
        neuron = next(iter(brain.neurons.values()))
        out = neuron.forward([2.0])
        if hasattr(out, "detach"):
            pred = float(out.detach().to("cpu").view(-1)[0].item())
        elif isinstance(out, list):
            pred = float(out[0])
        else:
            pred = float(out)
        self.assertAlmostEqual(pred, 4.0, delta=3.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
