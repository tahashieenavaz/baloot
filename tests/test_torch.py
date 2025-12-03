import pytest
import types
import baloot


class DummyParam:
    def __init__(self, n, requires_grad=True):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self):
        return self._n


class DummyModel:
    def parameters(self):
        return [
            DummyParam(10, True),
            DummyParam(5, False),  # ignored
            DummyParam(3, True),
        ]


def test_parameter_count_only_trainable():
    model = DummyModel()
    assert baloot.parameter_count(model) == 13


def test_acceleration_device_cpu(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
        device=lambda x: x,
    )

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert baloot.acceleration_device() == "cpu"


def test_acceleration_device_cuda(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
        device=lambda x: x,
    )

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert baloot.acceleration_device() == "cuda"


def test_acceleration_device_mps(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
        device=lambda x: x,
    )

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert baloot.acceleration_device() == "mps"


def test_acceleration_device_no_torch(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    with pytest.raises(ImportError):
        baloot.acceleration_device()
