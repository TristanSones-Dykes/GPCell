# Standard Library Imports
from typing import Mapping, Union
from abc import ABC, abstractmethod

# Third-Party Library Imports
from gpflow import Parameter
from gpflow.models import GPR

# Internal Project Imports

# for defining custom types
from numpy import float64, int32
from numpy.typing import NDArray
from numpy.random import RandomState


# ---------------------#
# --- Define types --- #
# ---------------------#

Ndarray = NDArray[float64] | NDArray[int32]
GPPriors = Mapping[str, Union[Parameter, RandomState, float, bool]]


class GPModel(ABC):
    """
    Gaussian Process constructor base class
    """

    def __init__(self, priors: GPPriors = {}):
        self.priors = priors

    @abstractmethod
    def __call__(self, X: Ndarray, y: Ndarray) -> GPR:
        pass
