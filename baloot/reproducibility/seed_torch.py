import torch


def seed_torch(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
