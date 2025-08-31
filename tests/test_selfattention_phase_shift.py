import unittest


class PhaseShiftRoutineTests(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import (
            Brain,
            Wanderer,
            SelfAttention,
            run_training_with_datapairs,
            UniversalTensorCodec,
            REPORTER,
            clear_report_group,
        )
        from marble.plugins.selfattention_phase_shift import PhaseShiftRoutine

        self.Brain = Brain
        self.Wanderer = Wanderer
        self.SelfAttention = SelfAttention
        self.run_training_with_datapairs = run_training_with_datapairs
        self.PhaseShiftRoutine = PhaseShiftRoutine
        self.Codec = UniversalTensorCodec
        self.reporter = REPORTER
        self.clear = clear_report_group

    def test_phase_shift_reports_metrics(self):
        # ensure clean reporter state
        self.clear("selfattention")

        b = self.Brain(1, size=2)
        idxs = list(b.available_indices())
        b.add_neuron(idxs[0], tensor=[0.0])
        if len(idxs) > 1:
            b.add_neuron(idxs[1], tensor=[0.0])
            b.connect(idxs[0], idxs[1], direction="bi")

        sa = self.SelfAttention(routines=[self.PhaseShiftRoutine()])
        codec = self.Codec()
        datapairs = [(0, 0)]

        res = self.run_training_with_datapairs(
            b,
            datapairs,
            codec,
            steps_per_pair=2,
            lr=1e-2,
            selfattention=sa,
        )
        # Access logged event and ensure reported metrics exist
        events = self.reporter.group("selfattention", "events")
        log = events.get("phase_shift")
        print("phase_shift log", log)
        self.assertIsInstance(log, dict)
        for key in ("loss_speed", "neuron_count", "temperature"):
            self.assertIn(key, log)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
