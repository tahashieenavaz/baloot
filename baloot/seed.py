def seed_torch(seed: int):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_numpy(seed: int):
    import numpy

    numpy.random.seed(seed)


def seed_python(seed: int):
    import random

    random.seed(seed)


def full_seed(seed: int):
    seed_python(seed)
    seed_numpy(seed)
    seed_torch(seed)
