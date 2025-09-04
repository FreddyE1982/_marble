import unittest
import torch

from marble.marblemain import Brain, Wanderer


class ActorCriticPluginTests(unittest.TestCase):
    def make_brain(self):
        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        first = b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        second = b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        third = b.add_neuron((0.0, 1.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        for n in (second, third):
            for s in list(getattr(n, "outgoing", [])):
                if s.target is first:
                    b.remove_synapse(s)
        s1 = b.connect((0.0, 0.0), (1.0, 0.0), direction="uni")
        s2 = b.connect((0.0, 0.0), (0.0, 1.0), direction="uni")
        s2.weight = 2.0
        return b

    def test_actor_critic_updates_and_respects_actions(self):
        b = self.make_brain()
        w = Wanderer(b, type_name="actorcritic", seed=1)
        plug = next(p for p in w._wplugins if p.__class__.__name__ == "ActorCriticPlugin")
        start = b.get_neuron((0.0, 0.0))
        choices = w._gather_choices(start)
        syn, direction = plug.choose_next(w, start, choices)
        self.assertIn((syn, direction), choices)
        old_actor = plug.actor_w.clone()
        w.walk(max_steps=2, lr=0.0)
        print("actor weight after walk", plug.actor_w.item())
        self.assertFalse(torch.allclose(old_actor, plug.actor_w))
        self.assertIn("w_loss", w._learnables)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
