# Standard Library Imports
from typing import List, Optional, Tuple, Type
from copy import deepcopy

# Third-Party Library Imports
import pandas as pd

# Direct Namespace Imports
from numpy import float64, int32, max, std, zeros, mean, shape, nonzero
from numpy.typing import NDArray

from gpflow.kernels import Kernel, SquaredExponential
from gpflow import Parameter
from gpflow.utilities import to_default_float
from gpflow.models import GPR
import gpflow.optimizers as optimizers

import tensorflow_probability as tfp

# Internal Project Imports
from pyrocell.types import Ndarray, GPPriors
from pyrocell.gp.gpflow.backend import GaussianProcess, NoiseModel


# -----------------------------------#
# --- Pyrocell utility functions --- #
# -----------------------------------#


def fit_models(
    X: Ndarray,
    Y: Ndarray,
    Y_lengths: NDArray[int32],
    model: Type[GaussianProcess],
    priors: GPPriors,
    preprocess: int = 0,
    verbose: bool = False,
) -> List[Optional[GaussianProcess]]:
    """
    Fit a Gaussian Process model to each trace

    Parameters
    ----------
    X: Ndarray
        Input domain
    Y: Ndarray
        Input traces
    Y_lengths: Ndarray
        Length of each trace
    model: GaussianProcess
        Gaussian Process model
    priors: GPPriors
        Priors for the kernel hyperparameters
    preprocess: int
        Preprocessing option (0: None, 1: Centre, 2: Standardise)
    verbose: bool
        Print information

    Returns
    -------
    List[GaussianProcess]]
        List of fitted models
    """
    models = []

    for i in range(len(Y_lengths)):
        X_curr = X[: Y_lengths[i]]
        y_curr = Y[: Y_lengths[i], i, None]

        if preprocess == 1:
            y_curr = y_curr - mean(y_curr)
        elif preprocess == 2:
            y_curr = (y_curr - mean(y_curr)) / std(y_curr)

        m = model(**priors)
        m.fit(X_curr, y_curr, verbose=verbose)

        models.append(m)

    return models


def detrend(
    X: NDArray[float64],
    Y: NDArray[float64],
    Y_lengths: NDArray[int32],
    detrend_lengthscale: float,
    verbose: bool = False,
) -> Tuple[List[Optional[NDArray[float64]]], List[Optional[NoiseModel]]]:
    """
    Detrend stochastic process using RBF process

    Parameters
    ----------
    X: NDArray[float64]
        Input domain
    Y: NDArray[float64]
        Input traces
    Y_lengths: NDArray[int32]
        Length of each trace
    detrend_lengthscale: float
        Lengthscale of the detrending process
    verbose: bool
        Print information

    Returns
    -------
    Tuple[List[NDArray[float64]], List[NoiseModel]]
        Detrended traces, list of fit models
    """

    """
    # create model and set priors
    detrend_priors = {
        "lengthscales": Parameter(
            to_default_float(7.1),
            transform=tfp.bijectors.Softplus(low=to_default_float(7.0)),
        ),
    }
    m = NoiseModel(detrend_priors)"""

    detrend_list: List[Optional[NDArray[float64]]] = [None] * len(Y_lengths)
    models: List[Optional[NoiseModel]] = [None] * len(Y_lengths)

    for i in range(len(Y_lengths)):
        X_curr = X[: Y_lengths[i]]
        y_curr = Y[: Y_lengths[i], i, None]

        # standardise
        y_curr = (y_curr - mean(y_curr)) / std(y_curr)

        k_trend = SquaredExponential()
        gpr = GPR(data=(X_curr, y_curr), kernel=k_trend, mean_function=None)

        gpr.kernel.lengthscales = Parameter(
            to_default_float(detrend_lengthscale + 0.1),
            transform=tfp.bijectors.Softplus(low=to_default_float(detrend_lengthscale)),
        )

        opt = optimizers.Scipy()
        opt_logs = opt.minimize(
            gpr.training_loss, gpr.trainable_variables, options=dict(maxiter=100)
        )

        y_trend, var = gpr.predict_f(X_curr)

        m = NoiseModel()
        m.X = X_curr
        m.y = y_curr
        m.mean, m.var = y_trend, var

        """
        # fit model and extract mean
        m.fit(X, y)
        if verbose:
            print(f"Lengthscale: {m.fit_gp.kernel.lengthscales}")

        trend = m.mean"""

        # detrend and centre
        y_detrended = y_curr - y_trend
        y_detrended = y_detrended - mean(y_detrended)

        m.fit_gp = deepcopy(gpr)

        detrend_list[i] = y_detrended
        models[i] = m

    return detrend_list, models


def background_noise(
    X: Ndarray,
    Y: Ndarray,
    Y_lengths: NDArray[int32],
    verbose: bool = False,
) -> Tuple[float64, List[NoiseModel]]:
    """
    Fit a background noise model to the data and return the standard deviation of the overall noise

    Parameters
    ----------
    X: Ndarray
        Input domain
    Y: Ndarray
        Input traces
    Y_lengths: NDArray[int32]
        Length of each trace
    verbose: bool
        Print information

    Returns
    -------
    Tuple[Tensor, list[NoiseModel]]
        Standard deviation of the overall noise, list of noise models
    """
    std_array = zeros(len(Y_lengths), dtype=float64)

    priors = {
        "lengthscale": Parameter(
            to_default_float(7.1),
            transform=tfp.bijectors.Softplus(low=to_default_float(7.0)),
        ),
    }
    models = fit_models(X, Y, Y_lengths, NoiseModel, priors, 1, verbose)

    for i in range(len(Y_lengths)):
        noise_model = models[i]

        if noise_model is None:
            continue

        std_array[i] = noise_model.noise

    std = mean(std_array)

    if verbose:
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
