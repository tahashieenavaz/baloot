import torch
import random
from typing import Any, Type, List, Union, Tuple


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


def randomly_replace_layers(
    model: torch.nn.Module,
    target_type: Union[Type[torch.nn.Module], Tuple[Type[torch.nn.Module], ...]],
    replacement_pool: List[Type[torch.nn.Module]],
) -> torch.nn.Module:
    for name, child in model.named_children():
        if isinstance(child, target_type):
            NewLayerClass = random.choice(replacement_pool)
            setattr(model, name, NewLayerClass())
        else:
            randomly_replace_layers(child, target_type, replacement_pool)

    return model
