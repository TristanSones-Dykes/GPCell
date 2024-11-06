# Standard Library Imports
from typing import Optional, Tuple
from abc import ABC, abstractmethod

# Third-Party Library Imports

# Direct Namespace Imports

# Internal Project Imports
from ..types import TensorLike


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

    @abstractmethod
    def fit(self, X: TensorLike, y: TensorLike, verbose: bool = False):
        """
        Fit the Gaussian Process to the input.

        Parameters
        ----------
        X : TensorLike
            Input locations of the data.
        y : TensorLike
            Output values of the data.
        """
        pass

    @abstractmethod
    def log_likelihood(self, y: Optional[TensorLike] = None) -> TensorLike:
        """
        Compute the log likelihood of the Gaussian Process against the given response data.
        If no response data is provided, the log likelihood of the fit data is returned.

        Parameters
        ----------
        y : Optional[TensorLike]
            Output values of the data.

        Returns
        -------
        TensorLike
            Log likelihood of the Gaussian Process.
        """
        pass

    @abstractmethod
    def test_plot(
        self,
        X: Optional[TensorLike] = None,
        y: Optional[TensorLike] = None,
        plot_sd: bool = False,
    ):
        """
        Plot the Gaussian Process model against the given data.

        Parameters
        ----------
        X : Optional[TensorLike]
            Input locations of the data.
        y : Optional[TensorLike]
            Output values of the data.
        plot_sd : bool
            Whether to plot the standard deviation of the Gaussian Process.
        """
        pass
