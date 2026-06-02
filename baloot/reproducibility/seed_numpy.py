import numpy


def seed_numpy(seed: int) -> None:
    seed = int(seed)
    numpy.random.seed(seed)
