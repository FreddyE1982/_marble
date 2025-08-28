import unittest


class TestLearningParadigmHelpers(unittest.TestCase):
    def test_helpers_load_list_apply_remove(self):
        from marble.marblemain import (
            Brain,
            Wanderer,
            add_paradigm,
            ensure_paradigm_loaded,
            list_paradigms,
            remove_paradigm,
            apply_paradigms_to_wanderer,
        )

        b = Brain(2, size=(6, 6))
        # Load via add/ensure
        p1 = add_paradigm(b, "adaptive_lr", {"factor_down": 0.6})
        p2 = ensure_paradigm_loaded(b, "adaptive_lr")
        self.assertIs(p1, p2)

        lst = list_paradigms(b)
        print("paradigms:", lst)
        self.assertTrue(any(isinstance(x, dict) and x.get("class") == "AdaptiveLRParadigm" for x in lst))

        # Apply to a wanderer
        w = Wanderer(b)
        apply_paradigms_to_wanderer(b, w)
        # Should have at least one SelfAttention attached
        sals = getattr(w, "_selfattentions", [])
        self.assertTrue(len(sals) >= 1)

        # Remove by name
        ok = remove_paradigm(b, "adaptive_lr")
        self.assertTrue(ok)
        # Ensure cleared
        self.assertEqual(len(getattr(b, "_paradigms", [])), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

