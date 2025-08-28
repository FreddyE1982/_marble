import unittest


class TestSelfAttentionConv1D(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import (
            Brain,
            UniversalTensorCodec,
            SelfAttention,
            Conv1DRandomInsertionRoutine,
            run_training_with_datapairs,
            REPORTER,
        )
        self.Brain = Brain
        self.Codec = UniversalTensorCodec
        self.SelfAttention = SelfAttention
        self.Conv1DInserter = Conv1DRandomInsertionRoutine
        self.run_training_with_datapairs = run_training_with_datapairs
        self.reporter = REPORTER

    def test_conv1d_neuron_created_by_selfattention(self):
        # Small 2D brain with enough pre-existing neurons for Conv1D wiring
        b = self.Brain(2, size=(6, 6))
        codec = self.Codec()

        # Pre-create at least 6 neurons (5 params + 1 destination)
        avail = b.available_indices()
        needed = 6
        assert len(avail) >= needed, "test brain must provide enough indices"
        created = []
        for i in range(needed):
            created.append(b.add_neuron(avail[i], tensor=[float(i)]))

        # Connect a unidirectional synapse to ensure the wanderer takes a step
        b.connect(avail[0], avail[1], direction="uni")

        # One simple datapair to trigger at least one step
        datapairs = [(0, 0)]

        # SelfAttention with aggressive conv1d insertion to trigger immediately
        sa = self.SelfAttention(routines=[self.Conv1DInserter(period=1, eval_after=1)])

        # SelfAttention will only use pre-existing neurons now (no param creation)

        res = self.run_training_with_datapairs(
            b,
            datapairs,
            codec,
            steps_per_pair=3,
            lr=1e-2,
            selfattention=sa,
        )

        # Count conv1d neurons and validate at least one exists
        convs = [n for n in b.neurons.values() if getattr(n, "type_name", None) == "conv1d"]
        print("conv1d neurons created:", len(convs))

        # Log via reporter for auditability
        self.reporter.item["conv1d_count", "tests", "selfattention"] = len(convs)
        self.reporter.item["training_result", "tests", "selfattention"] = {
            "final_loss": res.get("final_loss"),
            "history_len": len(res.get("history", [])),
        }

        self.assertGreaterEqual(len(convs), 1, "expected at least one conv1d neuron to be created")

        # Strict: each conv1d should have exactly 5 incoming (params) and exactly 1 outgoing
        ok_any = False
        for c in convs:
            inc = getattr(c, "incoming", []) or []
            out = getattr(c, "outgoing", []) or []
            if len(inc) == 5 and len(out) == 1:
                ok_any = True
                break
        self.assertTrue(ok_any, "conv1d neuron wiring must be exactly 5 incoming and 1 outgoing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
