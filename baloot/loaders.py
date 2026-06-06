from typing import Literal

_torch_cache = None
_numpy_cache = None


def load_torch():
    global _torch_cache
    if _torch_cache is not None:
        return _torch_cache

    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required but not installed. Install it with `pip install torch`."
        ) from e
    _torch_cache = torch
    return torch


def load_numpy():
    global _numpy_cache
    if _numpy_cache is not None:
        _numpy_cache = numpy

    try:
        import numpy
    except ImportError as e:
        raise ImportError(
            "Numpy is required but not installed. Install it with `pip install numpy`."
        ) from e

    _numpy_cache = numpy
    return numpy


def load_pickle():
    import pickle

    return pickle


_LOADER_MAP = {"torch": load_torch, "numpy": load_numpy, "pickle": load_pickle}


def load(library: Literal["torch", "numpy", "pickle"]):
    if library not in _LOADER_MAP:
        raise ValueError("Library in load must be in [torch, numpy, pickle]")

    loader_callback = _LOADER_MAP.get(library)
    loader_callback()
