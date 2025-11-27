def parameter_count(model) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def acceleration_device():
    try:
        import torch
    except ImportError:
        raise Exception(
            "In order to get current acceleration device you need to have torch installed. Install using `pip install torch`."
        )

    _device = "cpu"
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.mps.is_available():
        _device = "mps"
    return torch.device(_device)
