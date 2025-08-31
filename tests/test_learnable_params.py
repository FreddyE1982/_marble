import unittest


class LearnableParamTests(unittest.TestCase):
    def test_global_learnables_and_decorator(self):
        from marble.marblemain import (
            Brain,
            Wanderer,
            SelfAttention,
            expose_learnable_params,
        )

        brain = Brain(1, size=1)
        w = Wanderer(brain)
        sa = SelfAttention()
        sa._bind(w)
        w._selfattentions.append(sa)

        # Wanderer learnable
        w.ensure_learnable_param("wp", 0.5)
        w.set_param_optimization("wp", enabled=True, lr=0.1)

        # Brain learnable
        brain.ensure_learnable_param("bp", 1.0)
        brain.set_param_optimization("bp", enabled=True, lr=0.1)

        # SelfAttention global learnable
        sa.ensure_global_learnable_param("sp", 2.0)
        sa.set_global_param_optimization("sp", enabled=True, lr=0.1)

        @expose_learnable_params
        def foo(wanderer, a: float = 1.0):
            torch = wanderer._torch  # type: ignore[attr-defined]
            return (
                a
                + wanderer.get_learnable_param_tensor("wp")
                + brain.get_learnable_param_tensor("bp")
                + sa.get_global_param_tensor("sp")
            )

        foo(w)  # register parameter 'a'
        w.set_param_optimization("a", enabled=True, lr=0.1)

        # Record previous values
        torch = w._torch  # type: ignore[attr-defined]
        a_before = w.get_learnable_param_tensor("a").clone()
        bp_before = brain.get_learnable_param_tensor("bp").clone()
        sp_before = sa.get_global_param_tensor("sp").clone()
        wp_before = w.get_learnable_param_tensor("wp").clone()

        loss = foo(w)
        loss.backward()
        w._update_learnables()
        brain._update_learnables()
        sa._update_learnables(w)
        print(
            "params before/after:",
            a_before.tolist(),
            w.get_learnable_param_tensor("a").tolist(),
        )
        self.assertFalse(torch.equal(a_before, w.get_learnable_param_tensor("a")))
        self.assertFalse(torch.equal(wp_before, w.get_learnable_param_tensor("wp")))
        self.assertFalse(torch.equal(bp_before, brain.get_learnable_param_tensor("bp")))
        self.assertFalse(torch.equal(sp_before, sa.get_global_param_tensor("sp")))
class AutoParamLearningTests(unittest.TestCase):
    def test_auto_learn_all_numeric_parameters(self):
        import importlib.util
        import os
        import sys
        import tempfile

        from marble.marblemain import Brain, Wanderer
        from marble.auto_param import enable_auto_param_learning

        brain = Brain(1, size=1, learn_all_numeric_parameters=True)
        w = Wanderer(brain)

        module_code = (
            "def foo(wanderer, b=2.0):\n"
            "    return wanderer.get_learnable_param_tensor('b')\n"
        )

        path = os.path.join(os.getcwd(), "mymod_temp.py")
        with open(path, "w", encoding="utf8") as fh:
            fh.write(module_code)
        try:
            with enable_auto_param_learning(brain):
                mod = importlib.import_module("mymod_temp")
            mod.foo(w)
        finally:
            if "mymod_temp" in sys.modules:
                del sys.modules["mymod_temp"]
            os.remove(path)
        self.assertIn("b", w._learnables)


if __name__ == "__main__":
    unittest.main()

