import torch


def arccot(x: torch.Tensor) -> torch.Tensor:
    return (torch.pi / 2) - torch.arctan(x)
