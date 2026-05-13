import torch


def cot(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tan(x)
