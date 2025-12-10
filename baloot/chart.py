import matplotlib.pyplot as plt
import numpy
from typing import Union, List, Optional, Tuple

PlotInputDataType = Union[List[float], numpy.ndarray]


def plot(
    x: PlotInputDataType,
    y: Optional[PlotInputDataType] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    xtitle: Optional[str] = None,
    ytitle: Optional[str] = None,
    output_path: Optional[str] = None,
    color: str = "#2E86C1",
    alpha: float = 0.1,
):
    if y is None:
        y = x
        x = list(range(len(y)))

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: x has {len(x)} elements, y has {len(y)}.")

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axis = plt.subplots(figsize=figsize, dpi=100)
        axis.plot(x, y, color=color, lw=2.5, zorder=10)
        axis.fill_between(x, y, color=color, alpha=alpha, zorder=5)

        if title:
            axis.set_title(
                title, fontsize=16, weight="bold", pad=20, loc="left", color="#333"
            )

        if xtitle:
            axis.set_xlabel(xtitle, fontsize=12, labelpad=10, color="#555")

        if ytitle:
            axis.set_ylabel(ylabel, fontsize=12, labelpad=10, color="#555")

    plt.plot(x, y)
    plt.show()
