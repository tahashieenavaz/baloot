import builtins
import random
import sys
import numpy
import pytest
from helpers import _make_fake_torch


FAKE_TORCH = _make_fake_torch()
sys.modules["torch"] = FAKE_TORCH

from baloot import (
    acceleration_device,
    funnel,
    parameter_count,
    render_template,
    seed_everything,
    seed_numpy,
    seed_python,
    seed_torch,
)
import baloot.seed as seed_module


def test_parameter_count_counts_trainable_parameters():
    class Param:
        def __init__(self, count, requires_grad):
            self._count = count
            self.requires_grad = requires_grad

        def numel(self):
            return self._count

    class Model:
        def parameters(self):
            return [
                Param(2, True),
                Param(3, False),
                Param(5, True),
            ]

    assert parameter_count(Model()) == 7


def test_render_template_replaces_placeholders(tmp_path):
    template = tmp_path / "template.txt"
    output = tmp_path / "output.txt"
    template.write_text("Hello %name%!")

    assert render_template(template, output, name="World") is True
    assert output.read_text() == "Hello World!"


def test_render_template_returns_false_on_error(tmp_path):
    missing_template = tmp_path / "missing.txt"
    output = tmp_path / "output.txt"

    assert render_template(missing_template, output, name="X") is False


def test_funnel_saves_and_loads_objects(tmp_path):
    location = tmp_path / "data.pkl"
    payload = {"value": 42, "nested": [1, 2, 3]}

    assert funnel(location, payload) is True
    assert funnel(location) == payload


def test_funnel_returns_none_for_missing_file(tmp_path):
    assert funnel(tmp_path / "nope.pkl") is None


def test_acceleration_device_prefers_cuda():
    FAKE_TORCH.cuda.is_available = lambda: True
    FAKE_TORCH.mps.is_available = lambda: False

    assert acceleration_device() == "device:cuda"


def test_acceleration_device_uses_mps_when_cuda_missing():
    FAKE_TORCH.cuda.is_available = lambda: False
    FAKE_TORCH.mps.is_available = lambda: True

    assert acceleration_device() == "device:mps"


def test_acceleration_device_defaults_to_cpu():
    assert acceleration_device() == "device:cpu"


def test_acceleration_device_requires_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(Exception) as excinfo:
        acceleration_device()

    assert "need to have torch installed" in str(excinfo.value)


def test_seed_python_calls_random_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(random, "seed", lambda value: calls.append(value))

    seed_python(123)

    assert calls == [123]


def test_seed_numpy_calls_numpy_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(numpy.random, "seed", lambda value: calls.append(value))

    seed_numpy(456)

    assert calls == [456]


def test_seed_torch_sets_torch_state():
    seed_torch(7)

    assert FAKE_TORCH.manual_seed_calls == [7]
    assert FAKE_TORCH.cuda.manual_seed_all_calls == [7]
    assert FAKE_TORCH.backends.cudnn.deterministic is True
    assert FAKE_TORCH.backends.cudnn.benchmark is False


def test_seed_everything_calls_all_seeders(monkeypatch):
    calls = []

    monkeypatch.setattr(
        seed_module, "seed_python", lambda seed: calls.append(("python", seed))
    )
    monkeypatch.setattr(
        seed_module, "seed_numpy", lambda seed: calls.append(("numpy", seed))
    )
    monkeypatch.setattr(
        seed_module, "seed_torch", lambda seed: calls.append(("torch", seed))
    )
    monkeypatch.setattr(
        seed_module,
        "seed_gymnasium",
        lambda seed, env: calls.append(("gym", seed, env)),
    )

    fake_env = object()
    seed_everything(9, fake_env)

    assert ("python", 9) in calls
    assert ("numpy", 9) in calls
    assert ("torch", 9) in calls
    assert ("gym", 9, fake_env) in calls


def test_seed_everything_skips_gym_when_not_provided(monkeypatch):
    calls = []
    monkeypatch.setattr(seed_module, "seed_gymnasium", lambda *args: calls.append(args))

    seed_everything(10)

    assert calls == []
