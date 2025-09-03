import unittest


class TestLearningParadigmStacking(unittest.TestCase):
    def test_stack_multiple_paradigms_and_walk(self):
        from marble.marblemain import (
            Brain,
            Wanderer,
            apply_paradigms_to_wanderer,
            ContrastiveInfoNCEPlugin,
            TDQLearningPlugin,
            DistillationPlugin,
            REPORTER,
        )

        # Minimal graph: 3 nodes in a line
        b = Brain(2, size=(6, 6))
        it = iter(b.available_indices())
        i1 = next(it); i2 = next(it); i3 = next(it)
        n1 = b.add_neuron(i1, tensor=[1.0])
        n2 = b.add_neuron(i2, tensor=[0.5], connect_to=i1, direction="uni")
        for s in list(getattr(n2, "outgoing", [])):
            if s.target is n1:
                b.remove_synapse(s)
        n3 = b.add_neuron(i3, tensor=[0.25], connect_to=i2, direction="uni")
        for s in list(getattr(n3, "outgoing", [])):
            if s.target is n2:
                b.remove_synapse(s)
        s12 = b.connect(i1, i2, direction="uni")
        s23 = b.connect(i2, i3, direction="uni")

        # Load multiple paradigms
        b.load_paradigm("contrastive", {"contrastive_tau": 0.1, "contrastive_lambda": 0.5})
        b.load_paradigm("reinforcement", {"rl_epsilon": 0.0, "rl_alpha": 0.2, "rl_gamma": 0.9})
        b.load_paradigm("student_teacher", {"distill_lambda": 0.05, "teacher_momentum": 0.9})
        b.load_paradigm("hebbian", {"hebbian_eta": 0.05, "hebbian_decay": 0.0})

        # Create wanderer and apply paradigms explicitly
        w = Wanderer(b, seed=0)
        apply_paradigms_to_wanderer(b, w)

        # Verify paradigms are active and walk completes
        act = b.active_paradigms()
        print("active paradigms:", [p.__class__.__name__ for p in act])
        self.assertGreaterEqual(len(act), 4)

        # Walk and ensure it completes and logs loss
        res = w.walk(max_steps=3, start=n1, lr=1e-2)
        self.assertIsInstance(res.get("loss", 0.0), float)
        print("stacked walk loss:", res.get("loss"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
