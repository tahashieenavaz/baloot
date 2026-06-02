from baloot.helpers import load_torch


def coth(x):
    torch = load_torch()
    assert isinstance(
        x, torch.Tensor
    ), f"baloot.coth accepts torch.Tensor {type(x)} was given."
    return 1.0 / torch.tanh(x)
