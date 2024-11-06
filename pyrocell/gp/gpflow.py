# Standard Library Imports
from typing import Optional, Tuple

# Third-Party Library Imports
import matplotlib.pyplot as plt
import pandas as pd

# Direct Namespace Imports
from numpy import max, ndarray, zeros
from numpy import float64, int32
from numpy.typing import NDArray

# Internal Project Imports
from pyrocell.gp import GaussianProcessBase
from pyrocell.types import TensorLike, check_types


# -------------------------------- #
# --- Gaussian Process classes --- #
# -------------------------------- #


class GaussianProcess(GaussianProcessBase):
    """
    Gaussian Process model using GPflow.
    """

    def __call__(
        self,
        X: TensorLike,
        full_cov: bool = False,
    ) -> Tuple[TensorLike, TensorLike]:
        raise NotImplementedError

    def fit(
        self,
        X: TensorLike,
        y: TensorLike,
        verbose: bool = False,
    ):
        raise NotImplementedError

    def log_likelihood(
        self,
        y: Optional[TensorLike] = None,
    ) -> TensorLike:
        raise NotImplementedError

    def test_plot(
        self,
        X: Optional[TensorLike] = None,
        y: Optional[TensorLike] = None,
        plot_sd: bool = False,
    ):
        raise NotImplementedError


# -----------------------------------------#
# --- Gaussian Process Implementations --- #
# -----------------------------------------#


# -------------------------------- #
# --- Gaussian Process helpers --- #
# -------------------------------- #


def detrend(X: NDArray[float64], y: NDArray[float64]) -> NDArray[float64]:
    """
    Detrend stochastic process using RBF process
    """
    check_types(ndarray, [X, y])

    return zeros(y.shape[0])


def background_noise(
    X: TensorLike,
    bckgd: TensorLike,
    bkcgd_length: TensorLike,
    M: int,
    verbose: bool = False,
) -> Tuple[NDArray[float64], GaussianProcess]:
    """
    Fit background noise model to the data
    """
    check_types(ndarray, [X, bckgd, bkcgd_length])

    return zeros(M), GaussianProcess()


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
