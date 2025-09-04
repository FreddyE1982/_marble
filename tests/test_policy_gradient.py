import os
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from marble.policy_gradient import PolicyGradientAgent


class TestPolicyGradient(unittest.TestCase):
    def test_policy_update_increases_rewarded_action(self) -> None:
        torch.manual_seed(0)
        agent = PolicyGradientAgent(state_dim=1, action_dim=2, lr=0.1, beta=0.0)
        states = torch.zeros(2, 1)
        actions = torch.tensor([0, 1])
        returns = torch.tensor([1.0, 0.0])

        init_probs = agent.action_probs(states)[0].detach().clone()
        loss = agent.step(states, actions, returns)
        new_probs = agent.action_probs(states)[0].detach()
        print("initial probs", init_probs.tolist(), "updated probs", new_probs.tolist(), "loss", loss)
        self.assertGreater(new_probs[0], init_probs[0])

    def test_constraint_penalty_discourages_action(self) -> None:
        torch.manual_seed(0)
        g1 = lambda a: (a == 1).float()
        agent = PolicyGradientAgent(
            state_dim=1, action_dim=2, lr=0.1, beta=0.0, lambdas=[5.0], constraints=[g1]
        )
        states = torch.zeros(2, 1)
        actions = torch.tensor([1, 1])
        returns = torch.tensor([1.0, 1.0])

        init_probs = agent.action_probs(states)[0].detach().clone()
        agent.step(states, actions, returns)
        new_probs = agent.action_probs(states)[0].detach()
        print("constraint initial", init_probs.tolist(), "constraint updated", new_probs.tolist())
        self.assertLess(new_probs[1], init_probs[1])


if __name__ == "__main__":
    unittest.main()
