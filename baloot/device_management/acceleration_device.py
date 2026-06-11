from baloot.loaders import load_torch
from .cuda import cuda
from .hip import hip
from .mps import mps
from .xpu import xpu
from .cpu import cpu


def acceleration_device(return_all: bool = False):
    torch = load_torch()
    devices = []

    if torch.cuda.is_available():
        devices.append(cuda())

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(mps())

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append(xpu())

    if torch.version.hip is not None:
        devices.append(hip())

    try:
        import torch_xla.core.xla_model as xm

        xm.xla_device()
        devices.append(torch.device("xla"))
    except Exception:
        pass

    devices.append(cpu())

    if return_all:
        return devices

    return devices[0]
