from typing import Union
from matplotlib import figure

HeightDataType = Union[int, None]


def size(width: int, height: HeightDataType) -> None:
    if height is None:
        height = width

    figure(figsize=(width, height))


def dpi(resolution) -> None:
    figure(dpi=resolution)
