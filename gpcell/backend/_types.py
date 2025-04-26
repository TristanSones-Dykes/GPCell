# Standard Library Imports
from abc import ABC, abstractmethod
from typing import (
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

# Third-Party Library Imports
from gpflow import Parameter
from gpflow.kernels import Kernel

# Internal Project Imports

# for defining custom types
from numpy import float64, int32
from numpy.typing import NDArray


# ---------------------#
# --- Define types --- #
# ---------------------#

# Convenience type aliases
Ndarray = NDArray[float64] | NDArray[int32]
Numeric = Union[float, int, float64, int32]

# Prior types
GPPrior = Mapping[str, Union[Parameter, Numeric, bool]]
"""
Type for Gaussian Process priors.

The keys are the paths to the priors, e.g `.kernel.lengthscales` or `.kernel.kernels[1].variance`.\n
Values:
    - `Parameter` used for transformed parameters e.g: Softplus.
    - `Numeric` used for numeric initial values.
"""
GPPriorFactory = Callable[..., GPPrior]
"""
Type for Gaussian Process prior factories.

Used for defining a pattern that will dynamically generate priors for each GPR model.
"""


class FixedPriorGen(ABC):
    """
    Abstract class for serving preset priors for Gaussian Process models.
    """

    @abstractmethod
    def __init__(self, prior_list: Sequence[GPPrior]) -> None:
        pass

    @abstractmethod
    def __call__(self, noise) -> GPPrior:
        pass


GPPriorTrainingFlag = (
    Mapping[str, bool]
    | Mapping[Tuple[int, str], bool]
    | Mapping[Union[str, Tuple[int, str]], bool]
)
"""
Type for setting which parameters of a kernel are trainable.\n
Members:
    - `str -> bool`: Set a global model parameter or a kernel parameter of a single kernel model.
    - `(int, str) -> bool`: Set a kernel parameter of a composite kernel model.
"""

# Kernel types
GPKernel = Union[Type[Kernel], Sequence[Type[Kernel]]]
"""
Type for Gaussian Process kernels.\n
Can be a single kernel or a list for a composite kernel. If a list is provided, an operator can be provided or `*` is used.
"""
GPOperator = Optional[Callable[[Kernel, Kernel], Kernel]]
"""
Type for operators that combine multiple kernels.

Examples for a product `*` operator:
    - `operator.mul` (what is used as default)
    - `gpflow.kernels.Product`
    - `lambda a, b: a * b`
"""
