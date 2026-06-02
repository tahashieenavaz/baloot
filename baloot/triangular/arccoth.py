from baloot.helpers import load_torch


def arccoth(x):
    torch = load_torch()
    assert isinstance(
        x, torch.Tensor
    ), f"baloot.arccoth accepts a torch.Tensor {type(x)} was given."
    return 0.5 * torch.log((x + 1.0) / (x - 1.0))
