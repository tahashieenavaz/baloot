from baloot.loaders import load_torch


def cuda():
    torch = load_torch()
    return torch.device("cuda")
