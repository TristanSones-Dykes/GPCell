# Standard Library Imports
from typing import Callable, List, Optional, Tuple

# Third-party Library Imports
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Direct Namespace Imports
from torch import (
    Tensor,
    clone,
    eye,
    log,
    logdet,
    matmul,
    mean,
    no_grad,
    pi,
    sqrt,
    tensor,
    zeros,
)
from torch.linalg import LinAlgError, solve
from torch.optim import LBFGS

# --- Pyro Imports --- #
from pyro import clear_param_store
from pyro.contrib.gp.kernels import RBF, Cosine, Exponential, Product
from pyro.contrib.gp.models import GPRegression
from pyro.distributions.constraints import greater_than
from pyro.infer import Trace_ELBO
from pyro.nn import PyroParam, PyroSample  # noqa: F401

# Internal Project Imports
from ...types import PyroKernel, PyroOptimiser, PyroPriors
from .. import GaussianProcessBase

# -------------------------------- #
# --- Gaussian Process classes --- #
# -------------------------------- #


class GaussianProcess(GaussianProcessBase):
    """
    Gaussian Process class for fitting and evaluating parameters
    """

    def __init__(
        self,
        kernel: PyroKernel,
        optimiser: PyroOptimiser,
        variance_loc: List[str] = ["variance"],
    ):
        self.kernel = kernel
        """Kernel for the Gaussian Process"""
        self.optimiser = optimiser
        """Optimiser for the Gaussian Process"""
        self.variance_loc = variance_loc
        """Location of the variance parameter in the kernel"""

    def __call__(
        self, X: Tensor, full_cov: bool = False, noiseless: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Evaluate the Gaussian Process on the input domain

        Parameters
        ----------
        X: Tensor
            Input domain
        full_cov: bool
            Return full covariance matrix
        noiseless: bool
            Return noiseless predictions

        Returns
        -------
        Tuple[Tensor, Tensor]
            Mean and standard deviation
        """
        if not isinstance(X, Tensor):
            raise TypeError("Input domain must be a tensor")

        with no_grad():
            mean, cov = self.fit_gp(X, full_cov=full_cov, noiseless=noiseless)

        if not full_cov:
            var = cov.diag()
            return mean, var

        return mean, cov

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        loss_fn: Callable[..., Tensor],
        lr: float = 0.01,
        noise: Optional[Tensor] = None,
        jitter: float = 1.0e-5,
        num_steps: int = 1000,
        priors: PyroPriors = {},
        verbose: bool = False,
    ) -> bool:
        """
        Fit the Gaussian Process model, saves the model and training values for later use if needed.

        Parameters
        ----------
        X: Tensor
            Input domain
        y: Tensor
            Target values
        loss_fn: Callable[..., Tensor]
            Loss function
        lr: float
            Learning rate
        noise: Tensor
            Variance of Gaussian noise of the model
        jitter: float
            Positive term to help Cholesky decomposition
        num_steps: int
            Number of steps
        priors: PyroPriors
            Priors for the kernel parameters
        verbose: bool
            Print training information

        Returns
        -------
        bool
            Success status
        """
        if not isinstance(X, Tensor) or not isinstance(y, Tensor):
            raise TypeError("Input domain and target values must be tensors")

        clear_param_store()

        # check if kernel is class or object
        if isinstance(self.kernel, type):
            kernel = self.kernel(input_dim=1)
        else:
            kernel = self.kernel

        # set priors
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        # gaussian regression
        sgpr = GPRegression(X, y, kernel, noise=noise, jitter=jitter)
        optimiser = self.optimiser(sgpr.parameters(), lr=lr)

        # check if closure is needed
        if optimiser.__class__.__name__ == "LBFGS":

            def closure():
                optimiser.zero_grad()
                loss = loss_fn(sgpr.model, sgpr.guide)
                loss.backward()
                return loss
        else:
            closure = None  # type: ignore

        try:
            for i in range(num_steps):
                optimiser.zero_grad()
                loss = loss_fn(sgpr.model, sgpr.guide)
                loss.backward()
                optimiser.step(closure)

                # if verbose and (i % 100 == 0 or i == num_steps-1):
                #    print(f"lengthscale: {sgpr.kernel.lengthscale.item()}")
        except LinAlgError as e:
            print(f"Lapack error code: {e}")
            return False

        # display final trained parameters
        if verbose:
            for param in priors.keys():
                print(f"{param}: {getattr(sgpr.kernel, param).item()}")

        # calculate log posterior density
        with no_grad():
            self.loss = loss_fn(sgpr.model, sgpr.guide)

        # mean and variance
        with no_grad():
            res: Tuple[Tensor, Tensor] = sgpr(X, full_cov=True, noiseless=False)
            self.mean, self.cov = res
            self.var = self.cov.diag()

        self.X_true = X
        self.y_true = y
        self.fit_gp = sgpr

        return True

    def log_likelihood(self, y: Optional[Tensor] = None) -> Tensor:
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
        # no y input -> use fitted mean and cov
        if y is None:
            y, mean, K = clone(self.y_true), clone(self.mean), clone(self.cov)
        elif not isinstance(y, Tensor):
            raise TypeError("Target values must be tensors")
        else:
            # evaluate new mean and cov
            mean, K = self.fit_gp(self.X_true, y)

        with no_grad():
            # add noise variance to kernel diagonal
            noise = getattr(self.fit_gp.kernel, self.variance_loc[0])
            for i in range(1, len(self.variance_loc)):
                noise = getattr(noise, self.variance_loc[i])

            K_with_noise = K + noise.item() * eye(K.size(0))

            # residuals
            residual = y - mean

            # Compute terms for log-marginal likelihood
            term1 = -0.5 * matmul(residual.T, solve(K_with_noise, residual))
            term2 = -0.5 * logdet(K_with_noise)
            term3 = -0.5 * len(y) * log(tensor(2 * pi))

        return term1 + term2 + term3

    def test_plot(
        self,
        X: Optional[Tensor] = None,
        y_true: Optional[Tensor] = None,
        plot_sd: bool = False,
    ):
        """
        Create a test plot of the fitted model on the training data

        Parameters
        ----------
        X: Optional[Tensor]
            Input domain
        y_true: Optional[Tensor]
            True target values
        plot_sd: bool
            Plot standard deviation
        """
        # check if fit_gp exists
        if not hasattr(self, "fit_gp"):
            raise AttributeError("Please fit the model first")

        # default X and y_true values
        if X is None:
            X = self.X_true
            mean, std = self.mean, self.var.sqrt()
        if y_true is None:
            y_true = self.y_true

        # plot
        plt.plot(X, mean, zorder=1, c="k", label="Fit GP")
        if not plot_sd:
            plt.plot(X, mean + 2 * std, zorder=0, c="r")
            plt.plot(X, mean - 2 * std, zorder=0, c="r")

        if y_true is not None:
            plt.plot(X, y_true, zorder=0, c="b", label="True data")


# -----------------------------------------#
# --- Gaussian Process Implementations --- #
# -----------------------------------------#


class OU(GaussianProcess):
    """
    Ornstein-Uhlenbeck process class
    """

    def __init__(self, priors: PyroPriors):
        # create kernel
        kernel = Exponential(input_dim=1)
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        super().__init__(kernel, LBFGS)  # type: ignore


class OUosc(GaussianProcess):
    """
    Ornstein-Uhlenbeck process with an oscillator (cosine) kernel
    """

    def __init__(
        self,
        ou_priors: PyroPriors,
        osc_priors: PyroPriors,
    ):
        # create kernels
        ou_kernel = Exponential(input_dim=1)
        osc_kernel = Cosine(input_dim=1)

        # set priors
        for param, prior in ou_priors.items():
            setattr(ou_kernel, param, prior)
        for param, prior in osc_priors.items():
            setattr(osc_kernel, param, prior)

        # combine kernels
        kernel = Product(ou_kernel, osc_kernel)

        super().__init__(kernel, LBFGS, variance_loc=["kern0", "variance_map"])  # type: ignore


class NoiseModel(GaussianProcess):
    """
    Noise model class
    """

    def __init__(self, priors: PyroPriors):
        # create kernel
        kernel = RBF(input_dim=1)
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        super().__init__(kernel, LBFGS)  # type: ignore


# -------------------------------- #
# --- Gaussian Process helpers --- #
# -------------------------------- #


def detrend(
    X: Tensor, y: Tensor, detrend_lengthscale: float, verbose: bool = False
) -> Optional[Tuple[Tensor, NoiseModel]]:
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
    priors = {
        "lengthscale": PyroParam(
            tensor(detrend_lengthscale), constraint=greater_than(7.0)
        ),
    }

    noise_model = NoiseModel(priors)
    success = noise_model.fit(
        X, y, Trace_ELBO().differentiable_loss, num_steps=15, verbose=verbose
    )

    if not success:
        return None

    y_detrended = y - noise_model.mean

    return y_detrended, noise_model


def background_noise(
    X: Tensor, bckgd: Tensor, bckgd_length: Tensor, M: int, verbose: bool = False
) -> Tuple[Tensor, list[NoiseModel]]:
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
    priors = {
        "lengthscale": PyroParam(tensor(7.1), constraint=greater_than(0.0)),
    }

    std_tensor = zeros(M)
    models = []

    for i in range(M):
        X_curr = X[: bckgd_length[i]]
        y_curr = bckgd[: bckgd_length[i], i]

        m = NoiseModel(priors)
        success = m.fit(
            X_curr,
            y_curr,
            Trace_ELBO().differentiable_loss,
            num_steps=100,
            verbose=verbose,
        )

        if not success:
            raise RuntimeError(f"Failed to fit background noise model {i}")

        models.append(m)
        std_tensor[i] = sqrt(m.fit_gp.kernel.variance)

    std = mean(std_tensor)

    print("Background noise model:")
    print(f"Standard deviation: {std}")

    return std, models


# ---------------------- #
# --- Pyro Utilities --- #
# ---------------------- #


def load_data(path: str) -> tuple[Tensor, Tensor, Tensor, int, Tensor, Tensor, int]:
    """
    Loads experiment data from a csv file. This file must have:
    - Time (h) column
    - Cell columns, name starting with 'Cell'
    - Background columns, name starting with 'Background'

    Parameters
    ----------
    path: str
        Path to the csv file

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, int, Tensor, Tensor, int]
        Split, formatted experimental data
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
    time = torch.from_numpy(df["Time (h)"].values[:, None])

    bckgd = torch.from_numpy(df[bckgd_cols].values)
    M = bckgd.shape[1]

    bckgd_length = torch.zeros(M, dtype=torch.int32)

    for i in range(M):
        bckgd_curr = bckgd[:, i]
        bckgd_length[i] = torch.max(torch.nonzero(bckgd_curr))

    y_all = torch.from_numpy(df[data_cols].values)

    N = y_all.shape[1]

    y_length = torch.zeros(N, dtype=torch.int32)

    for i in range(N):
        y_curr = y_all[:, i]
        y_length[i] = torch.max(torch.nonzero(y_curr))

    return time, bckgd, bckgd_length, M, y_all, y_length, N
