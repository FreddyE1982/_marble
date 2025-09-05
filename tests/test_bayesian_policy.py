import torch
from marble.bayesian_policy import BayesianPolicy


def test_bayesian_policy_update_and_sample():
    torch.manual_seed(0)
    policy = BayesianPolicy(feat_dim=2, action_dim=2, noise_var=1.0)
    # True parameters for action 0 and 1
    phi0 = torch.tensor([1.0, 0.0])
    phi1 = torch.tensor([0.0, 1.0])
    policy.update(0, phi0, 1.0)
    policy.update(1, phi1, 0.0)
    samples = policy.sample(torch.tensor([0, 1]))
    scores = torch.tensor([
        torch.dot(phi0, samples[0]),
        torch.dot(phi1, samples[1]),
    ])
    best = int(torch.argmax(scores))
    assert best == 0
    mu0, _ = policy.get_posterior(0)
    assert torch.allclose(mu0, torch.tensor([0.5, 0.0]), atol=1e-4)
