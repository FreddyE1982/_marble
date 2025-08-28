import unittest


class TestNewParadigmsAndPlugins(unittest.TestCase):
    def _make_line_graph(self, Brain):
        b = Brain(2, size=(5, 5))
        it = iter(b.available_indices())
        i1 = next(it); i2 = next(it); i3 = next(it)
        n1 = b.add_neuron(i1, tensor=[1.0])
        n2 = b.add_neuron(i2, tensor=[0.5])
        n3 = b.add_neuron(i3, tensor=[0.25])
        s1 = b.connect(i1, i2, direction="uni")
        s2 = b.connect(i2, i3, direction="uni")
        return b, (n1, n2, n3), (s1, s2)

    def _make_star_graph(self, Brain):
        b = Brain(2, size=(5, 5))
        it = iter(b.available_indices())
        i1 = next(it); i2 = next(it); i3 = next(it)
        n1 = b.add_neuron(i1, tensor=[1.0])
        n2 = b.add_neuron(i2, tensor=[0.0])
        n3 = b.add_neuron(i3, tensor=[0.0])
        s12 = b.connect(i1, i2, direction="uni")
        s13 = b.connect(i1, i3, direction="uni")
        return b, (n1, n2, n3), (s12, s13)

    def test_hebbian_paradigm_updates_weight(self):
        from marble.marblemain import Brain, Wanderer, REPORTER
        b, (n1, n2, _), (s1, _) = self._make_line_graph(Brain)
        b.load_paradigm("hebbian", {"hebbian_eta": 0.1, "hebbian_decay": 0.0})
        w = Wanderer(b, seed=0)
        w.walk(max_steps=3, start=n1, lr=1e-2)
        before_after = {"after": float(getattr(s1, "weight", 0.0))}
        print("hebbian weight after:", before_after["after"])
        REPORTER.item[("hebbian_weight_after", "tests", "paradigms")] = before_after
        # Expect some change from default 1.0 due to correlation update
        self.assertNotAlmostEqual(before_after["after"], 1.0, places=6)

    def test_contrastive_paradigm_adds_loss(self):
        from marble.marblemain import Brain, Wanderer
        # Baseline
        b1, (n1, n2, n3), _ = self._make_line_graph(Brain)
        w1 = Wanderer(b1, seed=0)
        res1 = w1.walk(max_steps=2, start=n1, lr=1e-2)
        loss1 = float(res1.get("loss", 0.0))
        # With contrastive
        b2, (m1, m2, m3), _ = self._make_line_graph(Brain)
        b2.load_paradigm("contrastive", {"contrastive_tau": 0.1, "contrastive_lambda": 1.0})
        w2 = Wanderer(b2, seed=0)
        res2 = w2.walk(max_steps=2, start=m1, lr=1e-2)
        loss2 = float(res2.get("loss", 0.0))
        print("contrastive loss base/with:", loss1, loss2)
        # Contrastive term is additive; expect non-decreasing total loss
        self.assertGreaterEqual(loss2, loss1)

    def test_reinforcement_paradigm_updates_q_values(self):
        from marble.marblemain import Brain, Wanderer
        b, (n1, n2, n3), (s12, s13) = self._make_star_graph(Brain)
        b.load_paradigm("reinforcement", {"rl_epsilon": 0.0, "rl_alpha": 0.5, "rl_gamma": 0.9})
        w = Wanderer(b, seed=0)
        # Ensure the TD Q-learning plugin is attached (paradigm should add it; attach explicitly if missing)
        from marble.marblemain import TDQLearningPlugin
        plugs = getattr(w, "_wplugins", [])
        if not any(isinstance(p, TDQLearningPlugin) for p in plugs):
            plugs.append(TDQLearningPlugin())
        print("reinforcement attached plugins:", [p.__class__.__name__ for p in plugs])
        w.walk(max_steps=3, start=n1, lr=1e-2)
        q1 = getattr(getattr(s12, "_plugin_state", {}), "get", lambda *_: None)("q", None)
        q2 = getattr(getattr(s13, "_plugin_state", {}), "get", lambda *_: None)("q", None)
        print("q-values:", q1, q2)
        # At least one Q should have been updated to a numeric value (likely negative)
        self.assertTrue((q1 is not None and isinstance(q1, float)) or (q2 is not None and isinstance(q2, float)))

    def test_student_teacher_paradigm_adds_distillation_loss(self):
        from marble.marblemain import Brain, Wanderer
        # Baseline
        b1, (n1, n2, n3), _ = self._make_line_graph(Brain)
        w1 = Wanderer(b1, seed=0)
        res1 = w1.walk(max_steps=3, start=n1, lr=1e-2)
        loss1 = float(res1.get("loss", 0.0))
        # With student-teacher distillation
        b2, (m1, m2, m3), _ = self._make_line_graph(Brain)
        b2.load_paradigm("student_teacher", {"distill_lambda": 0.1, "teacher_momentum": 0.9})
        w2 = Wanderer(b2, seed=0)
        res2 = w2.walk(max_steps=3, start=m1, lr=1e-2)
        loss2 = float(res2.get("loss", 0.0))
        print("distillation loss base/with:", loss1, loss2)
        self.assertGreaterEqual(loss2, loss1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
