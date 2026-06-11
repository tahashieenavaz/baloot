from baloot.loaders import load_torch


def xpu():
    torch = load_torch()
    return torch.device("xpu")
