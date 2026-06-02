from baloot.helpers import load_torch


def arccot(x):
    torch = load_torch()
    assert isinstance(
        x, torch.Tensor
    ), f"baloot.arccot accepts torch.Tensor {type(x)} was given."
    return (torch.pi / 2) - torch.arctan(x)
