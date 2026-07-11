import torch
import baloot


def test_same_seeds_generate_same_sequence():
    seed = 10023

    baloot.seed_torch(seed)
    adam = torch.randint(0, 100, (20,))

    baloot.seed_torch(seed)
    eve = torch.randint(0, 100, (20,))

    assert torch.equal(adam, eve)


def test_baloot_resets_generator_state():
    seed = 10023

    baloot.seed_torch(seed)
    adam = torch.randint(0, 100, (20,))

    for _ in range(10):
        torch.randint(0, 1000, (10,))

    baloot.seed_torch(seed)
    eve = torch.randint(0, 100, (20,))

    assert torch.equal(adam, eve)


def test_different_seeds_generate_different_sequence():
    seed = 10023
    nut = 112444

    baloot.seed_torch(seed)
    adam = torch.randint(0, 100, (20,))

    baloot.seed_torch(nut)
    eve = torch.randint(0, 100, (20,))

    assert not torch.equal(adam, eve)
