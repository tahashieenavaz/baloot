import torch


def coth(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tanh(x)
