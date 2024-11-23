# Standard Library Imports
from typing import Optional, Tuple

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from tensorflow import Tensor, sqrt, multiply
import gpflow.optimizers as optimizers

# Internal Project Imports
from pyrocell.gp import GaussianProcessBase
from pyrocell.gp.gpflow.backend.types import Ndarray, GPModel

# ------------------------------ #
# --- Gaussian Process class --- #
# ------------------------------ #


class GaussianProcess(GaussianProcessBase):
    """
    Gaussian Process model using GPflow.
    """

    def __init__(self, model: GPModel):
        self.model = model
        """Regression with kernel for the Gaussian Process"""

    def __call__(
        self,
        X: Ndarray,
        full_cov: bool = False,
    ) -> Tuple[Tensor, Tensor]:
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

        return self.fit_gp.predict_y(X, full_cov=full_cov)

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

        try:
            gp_reg = self.model(X, y)

            self.X, self.y = X, y
            opt = optimizers.Scipy()

            opt_logs = opt.minimize(
                gp_reg.training_loss,
                gp_reg.trainable_variables,  # type: ignore
                options=dict(maxiter=100),
            )

            # if verbose:
            # print(gp_reg.parameters)

            res = gp_reg.predict_y(X, full_cov=False)
            self.mean = res[0].numpy()
            self.var = res[1].numpy()

            self.noise = gp_reg.likelihood.variance**0.5  # type: ignore
            self.fit_gp = gp_reg
        except:
            print("Error in fitting the model")

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

        raise NotImplementedError

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
        std = multiply(sqrt(var), 2.0)

        # plot
        plt.plot(X, mean, zorder=1, c="k", label="Fit GP")
        plt.plot(X, y, zorder=1, c="b", label="True Data")

        if plot_sd:
            plt.plot(X, mean + std, zorder=0, c="r")
            plt.plot(X, mean - std, zorder=0, c="r")
