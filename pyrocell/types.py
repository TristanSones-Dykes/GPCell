# Standard Library Imports
from typing import Callable, TypeVar, Union, Mapping

# Third-Party Library Imports

# Direct Namespace Imports
from torch import Tensor

from numpy.typing import NDArray
from numpy import float64

from pyro.contrib.gp.kernels import Isotropy, Product, RBF, Cosine, Exponential
from pyro.nn import PyroParam, PyroSample
from pyro.distributions.constraints import greater_than
from pyro.infer import Trace_ELBO


# Internal Project Imports


# --- Pyro Types --- #
PyroOptimiser = Callable[..., Callable[..., Tensor]]
PyroKernel = Union[Isotropy, Product, RBF, Cosine, Exponential]
PyroPriors = Mapping[str, Union[PyroParam, PyroSample]]


# --- GPflow Types --- #


# --- Shared Types --- #
TensorLike = Union[Tensor, NDArray[float64]]
