# Standard Library Imports
from typing import Optional, Tuple
from functools import reduce
import operator

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import sqrt

import gpflow.optimizers as optimizers
from gpflow.models import GPR
from gpflow.utilities import set_trainable

# Internal Project Imports
from pyrocell.gp import GaussianProcessBase
from pyrocell.gp.gpflow.backend.types import (
    Ndarray,
    GPPriorFactory,
    GPKernel,
    GPPriorTrainingFlag,
    GPOperator,
)
from pyrocell.gp.gpflow.backend.utils import _multiple_assign

# ------------------------------ #
# --- Gaussian Process class --- #
# ------------------------------ #


class GPRConstructor:
    """
    Gaussian Process Regression constructor
    """

    def __init__(
        self,
        kernels: GPKernel,
        prior_gen: GPPriorFactory,
        trainable: GPPriorTrainingFlag = {},
        operator: Optional[GPOperator] = operator.mul,
    ):
        """
        Defines the kernel as a single kernel or a composite kernel using given operator

        Parameters
        ----------
        kernels : Union[Kernel, List[Kernel]]
            Single kernel or list of kernels
        prior_gen : Callable[..., GPPriors]
            Function that generates the priors for the GP model
        trainable : Union[Mapping[str, bool], Mapping[Tuple[int, str], bool]], optional
            Dictionary to set trainable parameters, by default {}
        operator : Optional[Callable[[Kernel, Kernel], Kernel]], optional
            Operator to combine multiple kernels, by default None
        """
        match kernels:
            case type():
                self.kernel = kernels
            case list():
                assert operator is not None, ValueError(
                    "Operator must be provided for composite kernels"
                )
                self.kernel = lambda: reduce(operator, [k for k in kernels])
            case _:
                raise TypeError(f"Invalid kernel type: {type(kernels)}")

        self.prior_gen = prior_gen
        self.trainable = trainable

    def __call__(self, X: Ndarray, y: Ndarray) -> GPR:
        # create new kernel and define model
        kernel = self.kernel()
        model = GPR((X, y), kernel)

        prior_dict = {k: v(*args) for k, (v, args) in self.prior_gen.items()}
        print("Prior dict:")
        print(prior_dict["kernel.lengthscales"].transform)
        # assign priors and set trainable parameters
        _multiple_assign(model, prior_dict)
        print("Model parameters 1:")
        print(model.kernel.lengthscales.transform)
        for param, trainable in self.trainable.items():
            match param:
                case str(s):
                    obj = model
                    attrs = s.split(".")
                case (int(i), str(s)):
                    obj = model.kernel.kernels[i]  # type: ignore
                    attrs = s.split(".")

            for attr in attrs[:-1]:
                print(obj)
                obj = getattr(obj, attr)
            set_trainable(getattr(obj, attrs[-1]), trainable)

        print("Model parameters 2:")
        print(model.kernel.lengthscales.transform)

        return model


class GaussianProcess(GaussianProcessBase):
    """
    Gaussian Process model using GPflow.
    """

    def __init__(self, constructor: GPRConstructor):
        self.constructor = constructor
        """Regression with kernel for the Gaussian Process"""

    def __call__(
        self,
        X: Ndarray,
        full_cov: bool = False,
    ) -> Tuple[Ndarray, Ndarray]:
        """
        Evaluate the Gaussian Process on the input domain

        Parameters
        ----------
        X: Ndarray
            Input domain
        full_cov: bool
            Return full covariance matrix

        Returns
        -------
        Tuple[Tensor, Tensor]
            Mean and standard deviation

        """
        if not hasattr(self, "fit_gp"):
            raise ValueError("Model has not been fit yet.")

        fit_mean, fit_var = self.fit_gp.predict_y(X, full_cov=full_cov)
        return fit_mean.numpy(), fit_var.numpy()

    def fit(
        self,
        X: Ndarray,
        y: Ndarray,
        verbose: bool = False,
    ):
        """
        Fit the Gaussian Process model, saves the model and training values for later use if needed.

        Parameters
        ----------
        X: Ndarray
            Input domain
        y: Ndarray
            Target values
        verbose: bool
            Print training information

        Returns
        -------
        bool
            Success status
        """

        gp_reg = self.constructor(X, y)

        self.X, self.y = X, y
        opt = optimizers.Scipy()

        opt.minimize(
            gp_reg.training_loss,
            gp_reg.trainable_variables,  # type: ignore
            options=dict(maxiter=100),
        )

        if verbose:
            # print("Trained GP model:")
            print(gp_reg.parameters)

        self.log_posterior_density = gp_reg.log_posterior_density().numpy()

        res = gp_reg.predict_y(self.X, full_cov=False)
        self.mean = res[0].numpy()
        self.var = res[1].numpy()

        self.noise = gp_reg.likelihood.variance**0.5  # type: ignore
        self.fit_gp = gp_reg

    def log_likelihood(
        self,
        y: Optional[Ndarray] = None,
    ) -> Ndarray:
        """
        Calculates the log-marginal likelihood for the Gaussian process.
        If no target values are input, calculates log likelihood for data is was it on.

        Parameters
        ----------
        y: Optional[Ndarray]
            Observed target values

        Returns
        -------
        Tensor
            Log-likelihood
        """
        return self.log_posterior_density

    def test_plot(
        self,
        X_y: Optional[Tuple[Ndarray, Ndarray]] = None,
        plot_sd: bool = False,
    ):
        """
        Create a test plot of the fitted model on the training data

        Parameters
        ----------
        X_y: Optional[Tuple[Ndarray, Ndarray]]
            Input domain and target values
        plot_sd: bool
            Plot standard deviation
        """

        # check if fit_gp exists
        if not hasattr(self, "fit_gp"):
            raise AttributeError("Please fit the model first")

        # check if X is None
        if X_y is None:
            X, y = self.X, self.y
            mean, var = self.mean, self.var
        else:
            X, y = X_y
            mean, var = self(X, full_cov=False)
        std = sqrt(var) * 2

        # plot
        plt.plot(X, mean, zorder=1, c="k", label="Fit GP")
        plt.plot(X, y, zorder=1, c="b", label="True Data")

        if plot_sd:
            plt.plot(X, mean + std, zorder=0, c="r")
            plt.plot(X, mean - std, zorder=0, c="r")
