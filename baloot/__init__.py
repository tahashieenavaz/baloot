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

from importlib.metadata import version

__version__ = version("baloot")
