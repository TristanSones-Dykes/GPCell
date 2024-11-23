# Standard Library Imports
from typing import Callable, Mapping, Union

# Third-Party Library Imports

# Direct Namespace Imports
from torch import Tensor
from pyro.contrib.gp.kernels import RBF, Cosine, Exponential, Isotropy, Product
from pyro.nn import PyroParam, PyroSample


# Internal Project Imports


# --- Pyro Types --- #
PyroOptimiser = Callable[..., Callable[..., Tensor]]
PyroKernel = Union[Isotropy, Product, RBF, Cosine, Exponential]
PyroPriors = Mapping[str, Union[PyroParam, PyroSample]]
