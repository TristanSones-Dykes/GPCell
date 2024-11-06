# Standard Library Imports
from typing import Union, Tuple
from abc import ABC, abstractmethod

# Third-Party Library Imports
import torch
import numpy as np

# Direct Namespace Imports
from torch import Tensor
from numpy.typing import NDArray
from numpy import float64

TensorLike = Union[Tensor, NDArray[float64]]

# Internal Project Imports
# ------------------------
# (e.g., from ..utils import load_data, from ..gp import GaussianProcessBase)


class GaussianProcessBase(ABC):
    """
    Base class for Gaussian Process models.
    """

    @abstractmethod
    def __call__(
        self, X: TensorLike, full_cov: bool = False
    ) -> Tuple[TensorLike, TensorLike]:
        """
        Evaluate the Gaussian Process at the given input locations.

        Parameters
        ----------
        X : TensorLike
            Input locations to evaluate the Gaussian Process at.
        full_cov : bool
            Whether to return the full covariance matrix or just the diagonal.

        Returns
        -------
        Tuple[TensorLike, TensorLike]
            Mean and covariance of the Gaussian Process at the input locations.
        """
        pass
