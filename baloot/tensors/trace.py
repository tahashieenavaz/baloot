from baloot.typing import TorchTensor


def trace(x: TorchTensor, dim1: int = -2, dim2: int = -1) -> TorchTensor:
    return x.diagonal(dim1=dim1, dim2=dim2).sum(dim=dim2)
