import math


def linear_epsilon(
    step: int, total_steps: int, start_epsilon: float, end_epsilon: float
) -> float:
    fraction = min(step / total_steps, 1.0)
    delta = end_epsilon - start_epsilon
    return start_epsilon + fraction * delta


def cosine_epsilon(
    step: int, total_steps: int, start_epsilon: float, end_epsilon: float
) -> float:
    fraction = min(step / total_steps, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * fraction))
    delta = start_epsilon - end_epsilon
    return end_epsilon + delta * cosine


def exploring_cosine_epsilon(
    step: int,
    total_steps: int,
    start_epsilon: float,
    end_epsilon: float,
    ratio: float = 0.5,
    initial_amplitude: float = 0.1,
    frequency: int = 6,
    decay_rate: float = 3.0,
) -> float:
    fraction = min(step / total_steps, 1.0)

    if fraction <= ratio:
        scaled_fraction = fraction / ratio
        cosine = 0.5 * (1 + math.cos(math.pi * scaled_fraction))
        delta = start_epsilon - end_epsilon
        return end_epsilon + delta * cosine
    else:
        oscillation_fraction = (fraction - ratio) / (1.0 - ratio)
        current_amplitude = initial_amplitude * math.exp(
            -decay_rate * oscillation_fraction
        )
        return end_epsilon + current_amplitude * math.sin(
            oscillation_fraction * frequency * math.pi
        )
