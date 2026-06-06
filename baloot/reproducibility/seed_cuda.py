from baloot.loaders import load_torch


def seed_cuda(seed: int) -> None:
    torch = load_torch()

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
