import random
import numpy as np
import baloot


def test_seed_python_is_deterministic():
    baloot.seed_python(123)
    a = [random.random() for _ in range(3)]

    baloot.seed_python(123)
    b = [random.random() for _ in range(3)]

    assert a == b


def test_seed_numpy_is_deterministic():
    baloot.seed_numpy(123)
    a = np.random.rand(3)

    baloot.seed_numpy(123)
    b = np.random.rand(3)

    assert np.allclose(a, b)


def test_seed_torch_cpu_is_deterministic():
    import torch

    baloot.seed_torch(123)
    a = torch.rand(3)

    baloot.seed_torch(123)
    b = torch.rand(3)

    assert torch.equal(a, b)


def test_seed_torch_sets_flags():
    import torch

    baloot.seed_torch(0)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seed_everything_runs_without_error():
    baloot.seed_everything(999)
