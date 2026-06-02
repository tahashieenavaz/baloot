def load_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required but not installed. Install it with `pip install torch`."
        ) from e
    return torch


def load_numpy():
    try:
        import numpy
    except ImportError as e:
        raise ImportError(
            "Numpy is required but not installed. Install it with `pip install numpy`."
        ) from e
    return numpy
