import random
import numpy as np
import baloot


def test_seed_python_is_deterministic():
    baloot.seed_python(123)
    a = [random.random() for _ in range(3)]

    baloot.seed_python(123)
    b = [random.random() for _ in range(3)]

    assert a == b


def test_seed_numpy_is_deterministic():
    baloot.seed_numpy(123)
    a = np.random.rand(3)

    baloot.seed_numpy(123)
    b = np.random.rand(3)

    assert np.allclose(a, b)


def test_seed_torch_cpu_is_deterministic():
    import torch

    baloot.seed_torch(123)
    a = torch.rand(3)

    baloot.seed_torch(123)
    b = torch.rand(3)

    assert torch.equal(a, b)


def test_seed_torch_sets_flags():
    import torch

    baloot.seed_torch(0)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


class DummySpace:
    def __init__(self):
        self.seed_value = None

    def seed(self, seed):
        self.seed_value = seed


class DummyEnv:
    def __init__(self, has_reset=True):
        self.reset_called_with = None

        if has_reset:
            self.reset = self._reset

        self.action_space = DummySpace()
        self.observation_space = DummySpace()

    def _reset(self, seed=None):
        self.reset_called_with = seed


def test_seed_gymnasium_calls_everything():
    env = DummyEnv()

    baloot.seed_gymnasium(42, env)

    assert env.reset_called_with == 42
    assert env.action_space.seed_value == 42
    assert env.observation_space.seed_value == 42


def test_seed_gym_observation_optional():
    env = DummyEnv()

    baloot.seed_gym(42, env, seed_observation=False)

    assert env.action_space.seed_value == 42
    assert env.observation_space.seed_value is None


def test_seed_gym_observation_enabled():
    env = DummyEnv()

    baloot.seed_gym(42, env, seed_observation=True)

    assert env.action_space.seed_value == 42
    assert env.observation_space.seed_value == 42


def test_seed_everything_runs_without_error():
    baloot.seed_everything(999)
