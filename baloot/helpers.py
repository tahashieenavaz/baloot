def load_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required but not installed. Install it with `pip install torch`."
        ) from e
    return torch
