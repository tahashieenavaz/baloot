import random
import torch
import numpy


def seed_torch(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_numpy(seed: int):
    numpy.random.seed(seed)


def seed_python(seed: int):
    random.seed(seed)


def seed_gymnasium(seed: int, environment: object):
    environment.reset(seed=seed)

    try:
        environment.action_space.seed(seed)
    except Exception:
        pass

    try:
        environment.observation_space.seed(seed)
    except Exception:
        pass


def seed_everything(seed: int):
    seed_python(seed)
    seed_numpy(seed)
    seed_torch(seed)
