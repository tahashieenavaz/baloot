import matplotlib.pyplot as plt


def plot(x, y=None):
    if y is None:
        y = x
        x = list(range(len(y)))
    plt.plot(x, y)
    plt.show()
