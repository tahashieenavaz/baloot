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

from .constants import pi
from .constants import e
