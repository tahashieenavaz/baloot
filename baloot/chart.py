import matplotlib.pyplot as plt


def plot(x, y=None):
    if y is None:
        y = x
        x = list(range(len(y)))

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: x has {len(x)} elements, y has {len(y)}.")

    plt.plot(x, y)
    plt.show()
