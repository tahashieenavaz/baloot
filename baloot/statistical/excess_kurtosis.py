from baloot.typing import TorchTensor
from baloot.loaders import load_torch
from .statistical_moment import statistical_moment


def excess_kurtosis(x: TorchTensor, dim: int = -1) -> TorchTensor:
    torch = load_torch()
    return torch.tensor(3.0) - statistical_moment(x=x, dim=dim, n=4)
