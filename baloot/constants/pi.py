from baloot.helpers import load_torch


def pi():
    torch = load_torch()
    return torch.tensor(torch.pi)
