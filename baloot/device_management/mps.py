from baloot.loaders import load_torch


def mps():
    torch = load_torch()
    return torch.device("mps")
