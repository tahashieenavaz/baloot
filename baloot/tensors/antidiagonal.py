from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def antidiagonal(x: TorchTensor, dim1: int = -2, dim2: int = -1):
    torch = load_torch()
    return torch.diagonal(torch.fliplr(x), dim1=dim1, dim2=dim2)
