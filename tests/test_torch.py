import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ACCELERATION_DEVICE_PATH = REPO_ROOT / "baloot" / "torch" / "acceleration_device.py"
PARAMETER_COUNT_PATH = REPO_ROOT / "baloot" / "torch" / "parameter_count.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@dataclass(frozen=True)
class FakeDevice:
    kind: str


def _fake_torch(cuda_available: bool, mps_available):
    calls = {"cuda": 0, "mps": 0, "device": []}

    def cuda_is_available():
        calls["cuda"] += 1
        return cuda_available

    backends = types.SimpleNamespace()

    if mps_available is not None:

        def mps_is_available():
            calls["mps"] += 1
            return mps_available

        backends.mps = types.SimpleNamespace(is_available=mps_is_available)

    def device(device_name):
        calls["device"].append(device_name)
        return FakeDevice(device_name)

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=cuda_is_available),
        backends=backends,
        device=device,
    )

    return torch_stub, calls


@pytest.fixture
def acceleration_device_module(monkeypatch):
    # Ensure import succeeds even when torch is not installed in the environment.
    bootstrap_torch, _ = _fake_torch(cuda_available=False, mps_available=None)
    monkeypatch.setitem(sys.modules, "torch", bootstrap_torch)

    return _load_module("test_acceleration_device_module", ACCELERATION_DEVICE_PATH)


@pytest.fixture(scope="module")
def parameter_count_function():
    module = _load_module("test_parameter_count_module", PARAMETER_COUNT_PATH)
    return module.parameter_count


def test_acceleration_device_prefers_cuda_and_short_circuits_mps(acceleration_device_module):
    mps_calls = {"count": 0}

    def mps_is_available():
        mps_calls["count"] += 1
        raise AssertionError("MPS should not be checked when CUDA is available")

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=mps_is_available)),
        device=lambda name: FakeDevice(name),
    )
    acceleration_device_module.torch = torch_stub

    device = acceleration_device_module.acceleration_device()

    assert device == FakeDevice("cuda")
    assert mps_calls["count"] == 0


def test_acceleration_device_uses_mps_when_cuda_is_unavailable(acceleration_device_module):
    torch_stub, calls = _fake_torch(cuda_available=False, mps_available=True)
    acceleration_device_module.torch = torch_stub

    device = acceleration_device_module.acceleration_device()

    assert device == FakeDevice("mps")
    assert calls == {"cuda": 1, "mps": 1, "device": ["mps"]}


def test_acceleration_device_returns_cpu_when_mps_is_unavailable(acceleration_device_module):
    torch_stub, calls = _fake_torch(cuda_available=False, mps_available=False)
    acceleration_device_module.torch = torch_stub

    device = acceleration_device_module.acceleration_device()

    assert device == FakeDevice("cpu")
    assert calls == {"cuda": 1, "mps": 1, "device": ["cpu"]}


def test_acceleration_device_returns_cpu_when_mps_backend_is_missing(acceleration_device_module):
    torch_stub, calls = _fake_torch(cuda_available=False, mps_available=None)
    acceleration_device_module.torch = torch_stub

    device = acceleration_device_module.acceleration_device()

    assert device == FakeDevice("cpu")
    assert calls == {"cuda": 1, "mps": 0, "device": ["cpu"]}


def test_acceleration_device_checks_cuda_before_mps(acceleration_device_module):
    events = []

    def cuda_is_available():
        events.append("cuda")
        return False

    def mps_is_available():
        events.append("mps")
        return True

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=cuda_is_available),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=mps_is_available)),
        device=lambda name: FakeDevice(name),
    )
    acceleration_device_module.torch = torch_stub

    device = acceleration_device_module.acceleration_device()

    assert device == FakeDevice("mps")
    assert events == ["cuda", "mps"]


class _Parameter:
    def __init__(self, size: int, requires_grad=True):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self.size


class _Model:
    def __init__(self, parameters):
        self._parameters = parameters

    def parameters(self):
        return iter(self._parameters)


def test_parameter_count_sums_only_trainable_parameters(parameter_count_function):
    model = _Model(
        [
            _Parameter(10, requires_grad=True),
            _Parameter(20, requires_grad=False),
            _Parameter(7, requires_grad=True),
        ]
    )

    assert parameter_count_function(model) == 17


def test_parameter_count_returns_zero_when_model_has_no_parameters(parameter_count_function):
    assert parameter_count_function(_Model([])) == 0


def test_parameter_count_skips_parameters_without_requires_grad(parameter_count_function):
    class MissingRequiresGrad:
        def numel(self):
            raise AssertionError("numel should not be called without requires_grad")

    model = _Model([MissingRequiresGrad(), _Parameter(4, requires_grad=True)])

    assert parameter_count_function(model) == 4


def test_parameter_count_does_not_call_numel_for_frozen_parameters(parameter_count_function):
    class FrozenParameter:
        requires_grad = False

        def numel(self):
            raise AssertionError("numel should not be called for frozen parameters")

    model = _Model([FrozenParameter(), _Parameter(3, requires_grad=True)])

    assert parameter_count_function(model) == 3


def test_parameter_count_calls_model_parameters_once(parameter_count_function):
    class CountingModel:
        def __init__(self):
            self.calls = 0
            self.values = [_Parameter(2, True), _Parameter(5, True)]

        def parameters(self):
            self.calls += 1
            return iter(self.values)

    model = CountingModel()

    assert parameter_count_function(model) == 7
    assert model.calls == 1


def test_parameter_count_propagates_model_parameters_errors(parameter_count_function):
    class BrokenModel:
        def parameters(self):
            raise RuntimeError("broken iterator")

    with pytest.raises(RuntimeError, match="broken iterator"):
        parameter_count_function(BrokenModel())


def test_parameter_count_propagates_numel_errors_for_trainable_parameters(parameter_count_function):
    class BrokenParameter:
        requires_grad = True

        def numel(self):
            raise ValueError("invalid tensor")

    model = _Model([BrokenParameter()])

    with pytest.raises(ValueError, match="invalid tensor"):
        parameter_count_function(model)
