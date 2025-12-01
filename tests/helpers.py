import types
import pytest


def _make_fake_torch(cuda_available: bool = False, mps_available: bool = False):
    module = types.ModuleType("torch")
    module.manual_seed_calls = []

    def manual_seed(seed):
        module.manual_seed_calls.append(seed)

    module.manual_seed = manual_seed

    cuda = types.SimpleNamespace()
    cuda.manual_seed_all_calls = []

    def manual_seed_all(seed):
        cuda.manual_seed_all_calls.append(seed)

    cuda.manual_seed_all = manual_seed_all
    cuda.is_available = lambda: cuda_available
    module.cuda = cuda

    mps = types.SimpleNamespace()
    mps.is_available = lambda: mps_available
    module.mps = mps

    module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=True)
    )
    module.device = lambda name: f"device:{name}"
    return module


@pytest.fixture(autouse=True)
def _reset_fake_torch():
    FAKE_TORCH.manual_seed_calls = []
    FAKE_TORCH.cuda.manual_seed_all_calls = []
    FAKE_TORCH.backends.cudnn.deterministic = False
    FAKE_TORCH.backends.cudnn.benchmark = True
    FAKE_TORCH.cuda.is_available = lambda: False
    FAKE_TORCH.mps.is_available = lambda: False
