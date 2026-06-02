from baloot.helpers import load_torch


def seed_torch(seed: int) -> None:
    torch = load_torch()
    seed = int(seed)
    torch.manual_seed(seed)
