from baloot.helpers import load_torch


def cot(x):
    torch = load_torch()
    assert isinstance(
        x, torch.Tensor
    ), f"baloot.cot accepts a torch.Tensor {type(x)} was given."
    return 1.0 / torch.tan(x)
