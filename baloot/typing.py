from typing import Any, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

TorchTensor: TypeAlias = Any
TorchModule: TypeAlias = Any

if TYPE_CHECKING:
    from torch import Tensor as _Tensor
    from torch.nn import Module as _Module

    TorchTensor = _Tensor
    TorchModule = _Module
