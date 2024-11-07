# Standard Library Imports
from typing import Callable, Mapping, Union

from numpy import float64, int32
from numpy.typing import NDArray
from pyro.contrib.gp.kernels import RBF, Cosine, Exponential, Isotropy, Product
from pyro.nn import PyroParam, PyroSample

# Third-Party Library Imports
# Direct Namespace Imports
from torch import Tensor

# Internal Project Imports


# --- Pyro Types --- #
PyroOptimiser = Callable[..., Callable[..., Tensor]]
PyroKernel = Union[Isotropy, Product, RBF, Cosine, Exponential]
PyroPriors = Mapping[str, Union[PyroParam, PyroSample]]


# --- GPflow Types --- #
Ndarray = NDArray[float64] | NDArray[int32]

# --- Shared Types --- #
