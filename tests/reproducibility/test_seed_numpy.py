import numpy
import baloot


def test_numpy_shows_expected_behavior_with_seed():
    seed = 10023

    baloot.seed_numpy(seed)
    adam = numpy.random.randint(0, 100, 20)

    baloot.seed_numpy(seed)
    eve = numpy.random.randint(0, 100, 20)

    assert numpy.array_equal(adam, eve)
