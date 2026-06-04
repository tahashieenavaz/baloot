from importlib.metadata import version

__version__ = version("baloot")

from .device_management import acceleration_device

from .parameters import parameter_count

from .file_management import render_template
from .file_management import funnel

from .reproducibility import seed_python
from .reproducibility import seed_torch
from .reproducibility import seed_numpy
from .reproducibility import seed_everything

from .triangular import cot
from .triangular import arccot
from .triangular import coth
from .triangular import arccoth

from .reinforcement_learning import soft_update
from .reinforcement_learning import soft_update_all
from .reinforcement_learning import hard_update
from .reinforcement_learning import hard_update_all

from .tensors import shuffle


def __getattr__(name):
    if name == "pi":
        from .constants import pi

        return pi()
    elif name == "e":
        from .constants import e

        return e()
    elif name == "ln2":
        from .constants import ln2

        return ln2()
    elif name == "ln10":
        from .constants import ln10

        return ln10()
    elif name == "golden":
        from .constants import golden

        return golden()
    elif name == "sqrt2":
        from .constants import sqrt2

        return sqrt2()
    elif name == "sqrt3":
        from .constants import sqrt3

        return sqrt3()
    elif name == "sqrt5":
        from .constants import sqrt5

        return sqrt5()
    elif name in {"c", "light", "speed-light"}:
        from .constants import c

        return c()

    raise AttributeError(name=name)
