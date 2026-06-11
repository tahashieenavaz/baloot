from baloot.loaders import load_torch


def xla():
    torch = load_torch()
    return torch.device("xla")
