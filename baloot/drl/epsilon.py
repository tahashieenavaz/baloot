import math


def linear_epsilon(step, k, start_eps, end_eps):
    fraction = min(step / k, 1.0)
    return start_eps + fraction * (end_eps - start_eps)


def cosine_epsilon(step, k, start_eps, end_eps):
    fraction = min(step / k, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * fraction))
    return end_eps + (start_eps - end_eps) * cosine
