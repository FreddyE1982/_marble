import unittest

from marble.reporter import REPORTER, clear_report_group


class TestQuantumTypeNeuron(unittest.TestCase):
    def setUp(self) -> None:
        clear_report_group("quantum")

    def test_forward_expectation(self) -> None:
        from marble.marblemain import Brain, Wanderer, _NEURON_TYPES

        b = Brain(1, size=(3,))
        src = b.add_neuron((0,), tensor=[2.0])
        q = b.add_neuron((1,), tensor=[0.0], type_name="quantumtype")
        dst = b.add_neuron((2,), tensor=[0.0])
        b.connect((0,), (1,), direction="uni")
        b.connect((1,), (2,), direction="uni")

        plug = _NEURON_TYPES["quantumtype"]
        weights, biases, _ = plug._ensure_internal_params(q)
        torch = getattr(q, "_torch", None)
        device = getattr(q, "_device", "cpu")
        if torch is not None:
            weights.data = torch.tensor([2.0, -1.0], dtype=torch.float32, device=device)
            biases.data = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
        else:
            weights[:] = [2.0, -1.0]
            biases[:] = [0.0, 0.0]

        wdr = Wanderer(b, seed=1)
        q._plugin_state["wanderer"] = wdr
        logits = plug._wave_logits(wdr, logits=[0.0, 1.0])
        if torch is not None and hasattr(logits, "data"):
            logits.data = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
        else:
            try:
                logits[:] = [0.0, 1.0]
            except Exception:
                pass

        out = q.forward([3.0])

        if torch is not None and getattr(out, "detach", None):
            val = float(out.detach().to("cpu").view(-1)[0].item())
        elif isinstance(out, list):
            val = float(out[0])
        else:
            val = float(out)

        REPORTER.item[("output", "quantum")] = val

        expected = (0.2689414213699951 * (2.0 * 3.0) + 0.7310585786300049 * (-1.0 * 3.0))
        self.assertAlmostEqual(val, expected, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

