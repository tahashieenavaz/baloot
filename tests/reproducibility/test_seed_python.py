import random
import baloot


def test_same_seed_generates_same_sequence():
    seed = 10023

    baloot.seed_python(seed)
    adam = [random.randint(0, 10) for _ in range(100)]

    baloot.seed_python(seed)
    eve = [random.randint(0, 10) for _ in range(100)]

    assert adam == eve


def test_different_seeds_produce_difference_sequences():
    seed = 10023
    nut = 123444

    baloot.seed_python(seed)
    adam = [random.randint(0, 10) for _ in range(100)]

    baloot.seed_python(nut)
    eve = [random.randint(0, 10) for _ in range(100)]

    assert adam != eve
