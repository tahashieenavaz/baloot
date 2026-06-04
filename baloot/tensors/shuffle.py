from typing import Any
from baloot.helpers import load_torch

TorchTensor = Any


def shuffle(x: TorchTensor, dim: int = -1) -> TorchTensor:
    torch = load_torch()
    random_permutations = torch.randperm(x.size(dim))
    return x.index_select(dim, random_permutations)
