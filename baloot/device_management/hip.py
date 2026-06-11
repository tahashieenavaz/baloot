from baloot.loaders import load_torch


def hip():
    torch = load_torch()
    return torch.device("hip")
