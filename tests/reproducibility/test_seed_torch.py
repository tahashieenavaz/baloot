import torch
import baloot


def test_torch_shows_expected_behavior_with_seed():
    seed = 10023

    baloot.seed_torch(seed)
    adam = torch.randint(0, 100, (20,))

    baloot.seed_torch(seed)
    eve = torch.randint(0, 100, (20,))

    assert torch.equal(adam, eve)
