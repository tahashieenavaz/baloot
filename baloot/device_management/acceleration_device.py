import torch


def acceleration_device(return_all=False):
    devices = []

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append(torch.device("xpu"))

    if torch.version.hip is not None:
        devices.append(torch.device("hip"))

    try:
        import torch_xla.core.xla_model as xm

        xm.xla_device()
        devices.append(torch.device("xla"))
    except Exception:
        pass

    devices.append(torch.device("cpu"))

    if return_all:
        return devices

    return devices[0]
