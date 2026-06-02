import torch


def arccoth(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log((x + 1.0) / (x - 1.0))
