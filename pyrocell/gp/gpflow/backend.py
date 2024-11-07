# Standard Library Imports
from typing import List, Optional, Tuple, override

# Third-Party Library Imports
import matplotlib.pyplot as plt
import pandas as pd

# Direct Namespace Imports
from numpy import float64, int32, max, zeros
from numpy.typing import NDArray
from gpflow.kernels import Kernel, SquaredExponential, Matern12, Cosine

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

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        """Kernel for the Gaussian Process"""

    def __call__(
        self,
        X: Ndarray,
        full_cov: bool = False,
    ) -> Tuple[Ndarray, Ndarray]:
        raise NotImplementedError

    def fit(
        self,
        X: Ndarray,
        y: Ndarray,
        verbose: bool = False,
    ):
        raise NotImplementedError

    def log_likelihood(
        self,
        y: Optional[Ndarray] = None,
    ) -> Ndarray:
        raise NotImplementedError

    def test_plot(
        self,
        X: Optional[Ndarray] = None,
        y: Optional[Ndarray] = None,
        plot_sd: bool = False,
    ):
        raise NotImplementedError


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
        assign_priors(matern, priors)

        super().__init__(matern)


class OUosc(GaussianProcess):
    """
    Ornstein-Uhlenbeck process with an oscillator (cosine) kernel
    """

    @override
    def __init__(self, ou_priors: GPPriors, osc_priors: GPPriors):
        matern = Matern12()
        assign_priors(matern, ou_priors)

        osc = Cosine()
        assign_priors(osc, osc_priors)

        super().__init__(matern * osc)


class NoiseModel(GaussianProcess):
    """
    Noise model class
    """

    @override
    def __init__(self, priors: GPPriors):
        kernel = SquaredExponential()
        assign_priors(kernel, priors)

        super().__init__(kernel)


# -------------------------------- #
# --- Gaussian Process helpers --- #
# -------------------------------- #


def detrend(
    X: NDArray[float64], y: NDArray[float64]
) -> Tuple[NDArray[float64], NoiseModel]:
    """
    Detrend stochastic process using RBF process
    """

    return zeros(y.shape[0]), NoiseModel({})


def background_noise(
    X: Ndarray,
    bckgd: Ndarray,
    bkcgd_length: Ndarray,
    M: int,
    verbose: bool = False,
) -> Tuple[NDArray[float64], List[NoiseModel]]:
    """
    Fit background noise model to the data
    """

    return zeros(M), [NoiseModel({})]


def assign_priors(kernel: Kernel, priors: GPPriors):
    """
    Assign priors to kernel hyperparameters
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
    M = bckgd.shape[1]

    bckgd_length = zeros(M, dtype=int32)

    for i in range(M):
        bckgd_curr = bckgd[:, i]
        bckgd_length[i] = max(bckgd_curr)

    y_all = df[data_cols].values

    N = y_all.shape[1]

    y_length = zeros(N, dtype=int32)

    for i in range(N):
        y_curr = y_all[:, i]
        y_length[i] = max(y_curr)

    return time, bckgd, bckgd_length, M, y_all, y_length, N
