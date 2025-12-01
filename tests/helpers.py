import types


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
