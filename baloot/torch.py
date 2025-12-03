from typing import Any


def parameter_count(model: Any) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if getattr(parameter, "requires_grad", False)
    )


def acceleration_device():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "To get the current acceleration device you must have torch installed. "
            "Install it with: `pip install torch`."
        ) from e

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
