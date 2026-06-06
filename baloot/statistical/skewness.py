from baloot.typing import TorchTensor
from .statistical_moment import statistical_moment


def skewness(x: TorchTensor, dim: int = -1):
    return statistical_moment(x=x, dim=dim, n=3)
