import matplotlib.pyplot as plt
import numpy
from typing import Union, List, Optional

PlotInputDataType = Union[List[float], numpy.ndarray]


def plot(x: PlotInputDataType, y: Optional[PlotInputDataType] = None):
    if y is None:
        y = x
        x = list(range(len(y)))

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: x has {len(x)} elements, y has {len(y)}.")

    plt.plot(x, y)
    plt.show()
