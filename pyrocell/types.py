# Standard Library Imports
from typing import Callable, Mapping, Union

# Third-Party Library Imports

# Direct Namespace Imports
from torch import Tensor
from pyro.contrib.gp.kernels import RBF, Cosine, Exponential, Isotropy, Product
from pyro.nn import PyroParam, PyroSample

from numpy import float64, int32
from numpy.typing import NDArray
from numpy.random import RandomState

# Internal Project Imports


# --- Pyro Types --- #
PyroOptimiser = Callable[..., Callable[..., Tensor]]
PyroKernel = Union[Isotropy, Product, RBF, Cosine, Exponential]
PyroPriors = Mapping[str, Union[PyroParam, PyroSample]]


# --- GPflow Types --- #
Ndarray = NDArray[float64] | NDArray[int32]
GPPriorTypes = Union[RandomState, NDArray[float64], NDArray[int32], float64, int32]
GPPriors = Mapping[str, GPPriorTypes]

# --- Shared Types --- #
