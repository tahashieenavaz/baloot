from baloot.helpers import load_numpy


def seed_numpy(seed: int) -> None:
    numpy = load_numpy()
    seed = int(seed)
    numpy.random.seed(seed)
