# Standard Library Imports
from typing import List, Optional, Tuple, override
from copy import deepcopy

# Third-Party Library Imports
import matplotlib.pyplot as plt
import pandas as pd

# Direct Namespace Imports
from numpy import float64, int32, max, zeros, mean, shape, nonzero
from numpy.typing import NDArray
from tensorflow import Tensor, sqrt, multiply

from gpflow.kernels import Kernel, SquaredExponential, Matern12, Cosine
from gpflow import Parameter
from gpflow.utilities import to_default_float, print_summary
from gpflow.models import GPR
import gpflow.optimizers as optimizers

import tensorflow_probability as tfp

# Internal Project Imports
from pyrocell.gp import GaussianProcessBase
from pyrocell.types import Ndarray, GPPriors

# -------------------------------- #
# --- Gaussian Process classes --- #
# -------------------------------- #


class GaussianProcess(GaussianProcessBase):
    """
    Gaussian Process model using GPflow.
    """

    def __init__(self, kernel: Kernel, priors: GPPriors):
        self.kernel = deepcopy(kernel)
        self.priors = priors
        """Kernel for the Gaussian Process"""

    def __call__(
        self,
        X: Ndarray,
        full_cov: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Evaluate the Gaussian Process on the input domain

        Parameters
        ----------
        X: Tensor
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
        X: Tensor
            Input domain
        y: Tensor
            Target values
        verbose: bool
            Print training information

        Returns
        -------
        bool
            Success status
        """

        gp_reg = GPR((X, y), kernel=self.kernel, mean_function=None)
        assign_priors(gp_reg.kernel, self.priors)

        self.X, self.y = X, y
        opt = optimizers.Scipy()

        opt_logs = opt.minimize(
            gp_reg.training_loss,
            gp_reg.trainable_variables,
            options=dict(maxiter=100),
        )

        self.mean, self.var = gp_reg.predict_y(X, full_cov=False)
        self.noise = gp_reg.likelihood.variance**0.5
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
        y: Optional[Tensor]
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
        X_y: Optional[Tuple[Tensor, Tensor]]
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


# -----------------------------------------#
# --- Gaussian Process Implementations --- #
# -----------------------------------------#


class OU(GaussianProcess):
    """
    Ornstein-Uhlenbeck process class
    """

    @override
    def __init__(self, priors: GPPriors):
        matern = Matern12()

        super().__init__(matern, priors)


class OUosc(GaussianProcess):
    """
    Ornstein-Uhlenbeck process with an oscillator (cosine) kernel
    """

    @override
    def __init__(self, ou_priors: GPPriors, osc_priors: GPPriors):
        matern = Matern12()
        osc = Cosine()

        super().__init__(matern * osc, ou_priors)


class NoiseModel(GaussianProcess):
    """
    Noise model class
    """

    @override
    def __init__(self, priors: GPPriors):
        kernel = SquaredExponential()

        super().__init__(kernel, priors)


# -------------------------------- #
# --- Gaussian Process helpers --- #
# -------------------------------- #


def detrend(
    X: NDArray[float64],
    y: NDArray[float64],
    detrend_lengthscale: float,
    verbose: bool = False,
) -> Tuple[NDArray[float64], NoiseModel]:
    """
    Detrend stochastic process using RBF process

    Parameters
    ----------
    X: Tensor
        Input domain
    y: Tensor
        Target values
    detrend_lengthscale: float
        Lengthscale of the detrending process
    verbose: bool
        Print information

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        detrended values and noise model
    """
    # create model and set priors
    detrend_priors = {
        "lengthscales": Parameter(
            to_default_float(7.1),
            transform=tfp.bijectors.Softplus(low=to_default_float(7.0)),
        ),
    }
    m = NoiseModel(detrend_priors)

    # fit model and extract mean
    m.fit(X, y)
    if verbose:
        print(f"Lengthscale: {m.fit_gp.kernel.lengthscales}")

    trend = m.mean

    # detrend and centre
    y_detrended = y - trend
    y_detrended = y_detrended - mean(y_detrended)

    return y_detrended, m


def background_noise(
    X: Ndarray,
    bckgd: Ndarray,
    bckgd_length: Ndarray,
    M: int,
    verbose: bool = False,
) -> Tuple[float64, List[NoiseModel]]:
    """
    Fit a background noise model to the data

    Parameters
    ----------
    X: Tensor
        Input domain
    bckgd: Tensor
        Background traces
    bckgd_length: Tensor
        Length of each background trace
    M: int
        Count of background regions
    verbose: bool
        Print information

    Returns
    -------
    Tuple[Tensor, list[NoiseModel]]
        Standard deviation of the overall noise, list of noise models
    """

    background_priors = {
        "lengthscales": Parameter(
            to_default_float(7.1),
            transform=tfp.bijectors.Softplus(low=to_default_float(7.0)),
        ),
    }

    std_array = zeros(M, dtype=float64)
    models = []

    for i in range(M):
        X_curr = X[: bckgd_length[i]]
        y_curr = bckgd[: bckgd_length[i], i, None]

        y_curr = y_curr - mean(y_curr)

        noise_model = NoiseModel(background_priors)
        noise_model.fit(X_curr, y_curr, verbose=verbose)

        std_array[i] = noise_model.noise
        models.append(noise_model)
    std = mean(std_array)

    print("Background noise model:")
    print(f"Standard deviation: {std}")

    return std, models


def assign_priors(kernel: Kernel, priors: GPPriors):
    """
    Assign priors to kernel hyperparameters

    Parameters
    ----------
    kernel: Kernel
        Kernel to assign priors to
    priors: GPPriors
        Priors for the kernel hyperparameters
    """
    for key, prior in priors.items():
        attribute = getattr(kernel, key)
        attribute.assign(prior)


# ------------------------ #
# --- GPflow Utilities --- #
# ------------------------ #


def load_data(
    path: str,
) -> Tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[int32],
    int,
    NDArray[float64],
    NDArray[int32],
    int,
]:
    """
    Loads experiment data from a csv file. This file must have:
    - Time (h) column
    - Cell columns, name starting with 'Cell'
    - Background columns, name starting with 'Background'

    :param str path: Path to the csv file.

    :return Tuple[NDArray[float64], NDArray[float64], NDArray[int32], int, NDArray[float64], NDArray[int32], int]: Split, formatted experimental data
    - time: time in hours
    - bckgd: background time-series data
    - bckgd_length: length of each background trace
    - M: count of background regions
    - y_all: cell time-series data
    - y_length: length of each cell trace
    - N: count of cell regions
    """
    df = pd.read_csv(path).fillna(0)
    data_cols = [col for col in df if col.startswith("Cell")]
    bckgd_cols = [col for col in df if col.startswith("Background")]
    time = df["Time (h)"].values[:, None]

    bckgd = df[bckgd_cols].values
    M = shape(bckgd)[1]

    bckgd_length = zeros(M, dtype=int32)

    for i in range(M):
        bckgd_curr = bckgd[:, i]
        bckgd_length[i] = max(nonzero(bckgd_curr))

    y_all = df[data_cols].values

    N = shape(y_all)[1]

    y_all = df[data_cols].values
    max(nonzero(y_all))

    y_length = zeros(N, dtype=int32)

    for i in range(N):
        y_curr = y_all[:, i]
        y_length[i] = max(nonzero(y_curr))

    return time, bckgd, bckgd_length, M, y_all, y_length, N
