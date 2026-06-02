from baloot.helpers import load_torch


def arccot(x):
    torch = load_torch()
    assert isinstance(x, torch.Tesnor)
    return (torch.pi / 2) - torch.arctan(x)
