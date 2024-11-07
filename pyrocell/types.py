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


# --- Shared Types --- #
TensorLike = Union[Tensor, NDArray[float64], NDArray[int32]]


# --- Shared Functions --- #
def check_types(type: type, values: list):
    """
    Check if the values are of the correct type.

    Parameters
    ----------
    type : type
        The type to check against.
    values : list
        List of values to check.
    """
    for value in values:
        if not isinstance(value, type):
            raise TypeError(f"Expected type {type}, got {type(value)}")
