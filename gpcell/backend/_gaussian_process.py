# Standard Library Imports
from typing import Optional, Tuple

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
import gpflow.optimizers as optimizers
from gpflow.posteriors import PrecomputeCacheType

# Internal Project Imports
from ._types import Ndarray
from ._gpr_constructor import GPRConstructor


NOCACHE = PrecomputeCacheType.NOCACHE


class GaussianProcess:
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

        fit_mean, fit_var = self.fit_gp.fused_predict_f(X, full_cov=full_cov)
        return fit_mean.numpy(), fit_var.numpy()  # type: ignore

    def fit(
        self,
        X: Ndarray,
        y: Ndarray,
        Y_var: bool = False,
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
        Y_var: bool
                Calculate variance of missing data
        verbose: bool
                Print training information

        Returns
        -------
        bool
                Success status
        """

        gp_reg = self.constructor(X, y)

        opt = optimizers.Scipy()

        opt.minimize(
            gp_reg.training_loss,
            gp_reg.trainable_variables,  # type: ignore
            options=dict(maxiter=100),
        )

        # Extract training values and model parameters
        self.log_posterior_density = gp_reg.log_posterior_density().numpy()  # type: ignore
        self.noise = gp_reg.likelihood.variance**0.5  # type: ignore

        if Y_var:
            self.Y_var = gp_reg.predict_y(X)[1].numpy()  # type: ignore

        # Keep the posterior for predictions
        self.fit_gp = gp_reg.posterior()

    def log_posterior(
        self,
        y: Optional[Ndarray] = None,
    ) -> Ndarray:
        """
        Compute the log posterior density of the model

        Parameters
        ----------
        y: Optional[Ndarray]
                Observed target values

        Returns
        -------
        Ndarray
                Log posterior density
        """
        return self.log_posterior_density

    def plot(
        self,
        plot_sd: bool = False,
        new_y: Optional[Ndarray] = None,
        new_label: str = "Altered data",
    ):
        """
        Plot the Gaussian Process model

        Parameters
        ----------
        plot_sd: bool
                Plot standard deviation
        new_y: Optional[Ndarray]
                New target values
        """
        if not hasattr(self, "fit_gp"):
            raise ValueError("Model has not been fit yet.")

        X = self.fit_gp.X_data
        y = self.fit_gp.Y_data
        fit_y = self.fit_gp.fused_predict_f(X)[0]

        plt.plot(X, y, label="True data")

        match new_y:
            case None:
                plt.plot(X, fit_y, label="Fit process")
            case _:
                try:
                    plt.plot(X, new_y, label=new_label)
                    plt.plot(
                        X,
                        fit_y,
                        label="Fit process",
                        color="k",
                        alpha=0.5,
                        linestyle="dashed",
                    )
                except Exception as e:
                    print("Problem with new data to plot: ", e)
