from baloot.loaders import load_torch


def cpu():
    torch = load_torch()
    return torch.device("cpu")
