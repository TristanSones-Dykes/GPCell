# Standard Library Imports
from typing import Callable, List, Sequence, Tuple, Type, Union

# Third-Party Library Imports
import pandas as pd

# Direct Namespace Imports
from numpy import float64, int32, max, std, zeros, mean, shape, nonzero, full
from numpy.typing import NDArray

from gpflow.kernels import Kernel
from gpflow import Parameter
from gpflow.utilities import to_default_float

import tensorflow_probability as tfp

# Internal Project Imports
from pyrocell.gp.gpflow.backend.types import Ndarray, GPPriors, GPModel
from pyrocell.gp.gpflow.backend import GaussianProcess
from pyrocell.gp.gpflow.models import NoiseModel


# -----------------------------------#
# --- Pyrocell utility functions --- #
# -----------------------------------#


def fit_models(
    X: Ndarray,
    Y: Ndarray,
    Y_lengths: NDArray[int32],
    model: Type[GPModel],
    priors: Union[GPPriors, List[GPPriors]],
    preprocess: int = 0,
    verbose: bool = False,
) -> List[GaussianProcess]:
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
    model: GPModel
        Model to fit
    priors: GPPriors | List[GPPriors]
        Priors for the kernel hyperparameters, or list of priors for each trace
    preprocess: int
        Preprocessing option (0: None, 1: Centre, 2: Standardise)
    verbose: bool
        Print information

    Returns
    -------
    List[GaussianProcess]]
        List of fitted models
    """
    if isinstance(priors, list):
        assert len(priors) == len(Y_lengths)
    else:
        priors = [priors] * len(Y_lengths)

    processes = []

    for i, prior in zip(range(len(Y_lengths)), priors):
        X_curr = X[: Y_lengths[i]]
        y_curr = Y[: Y_lengths[i], i, None]

        if preprocess == 1:
            y_curr = y_curr - mean(y_curr)
        elif preprocess == 2:
            y_curr = (y_curr - mean(y_curr)) / std(y_curr)

        gp_model = model(prior)
        m = GaussianProcess(gp_model)
        m.fit(X_curr, y_curr, verbose=verbose)

        processes.append(m)

    return processes


def fit_models_replicates(
    N: int,
    X: Ndarray,
    Y: Ndarray,
    Y_lengths: NDArray[int32],
    model: Type[GPModel],
    prior_gen: Union[Callable[..., GPPriors], List[Callable[..., GPPriors]]],
    preprocess: int = 0,
    verbose: bool = False,
) -> List[List[GaussianProcess]]:
    """
    Fit a Gaussian Process model to each trace N times

    Parameters
    ----------
    N: int
        Number of replicates
    X: Ndarray
        Input domain
    Y: Ndarray
        Input traces
    Y_lengths: Ndarray
        Length of each trace
    model: GPModel
        Model to fit
    priors: Callable[..., GPPriors] | List[Callable[..., GPPriors]]
        Function that generates priors for each replicate, or list of functions for each trace
    preprocess: int
        Preprocessing option (0: None, 1: Centre, 2: Standardise)
    verbose: bool
        Print information

    Returns
    -------
    List[List[GaussianProcess]]
        List of N fitted models for each trace
    """
    if isinstance(prior_gen, list):
        assert len(prior_gen) == len(Y_lengths)
        priors = prior_gen
    else:
        priors = [prior_gen] * len(Y_lengths)

    # preprocess data
    if preprocess == 1 or preprocess == 2:
        Y = Y.copy()

    for i in range(N):
        if preprocess == 1:
            Y[: Y_lengths[i], i, None] = Y[: Y_lengths[i], i, None] - mean(
                Y[: Y_lengths[i], i, None]
            )
        elif preprocess == 2:
            Y[: Y_lengths[i], i, None] = (
                Y[: Y_lengths[i], i, None] - mean(Y[: Y_lengths[i], i, None])
            ) / std(Y[: Y_lengths[i], i, None])

    # fit models
    processes = []
    for i in range(len(Y_lengths)):
        # select trace
        X_curr = X[: Y_lengths[i]]
        y_curr = Y[: Y_lengths[i], i, None]

        # replicate data
        replicate_X = pad_zeros([X_curr] * N)
        replicate_Y = pad_zeros([y_curr] * N)
        replicate_priors = [priors[i]() for _ in range(N)]
        replicate_lengths = full(N, Y_lengths[i], dtype=int32)

        # fit N models
        replicate_models = fit_models(
            replicate_X,
            replicate_Y,
            replicate_lengths,
            model,
            replicate_priors,
        )
        processes.append(replicate_models)

    return processes


def detrend(
    X: NDArray[float64],
    Y: NDArray[float64],
    Y_lengths: NDArray[int32],
    detrend_lengthscale: Union[float, int],
    verbose: bool = False,
) -> Tuple[List[NDArray[float64]], List[GaussianProcess]]:
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
    detrend_lengthscale: float | int
        Lengthscale of the detrending process, or integer portion of trace length
    verbose: bool
        Print information

    Returns
    -------
    Tuple[List[NDArray[float64]], List[GaussianProcess]]
        Detrended traces, list of fit models
    """
    if isinstance(detrend_lengthscale, float):
        priors = {
            "lengthscale": Parameter(
                to_default_float(detrend_lengthscale + 0.1),
                transform=tfp.bijectors.Softplus(
                    low=to_default_float(detrend_lengthscale)
                ),
            )
        }
        models = fit_models(X, Y, Y_lengths, NoiseModel, priors, 2, verbose)
    else:
        priors = []
        for i in range(len(Y_lengths)):
            lengthscale = Y_lengths[i] / detrend_lengthscale
            priors.append(
                {
                    "lengthscale": Parameter(
                        to_default_float(lengthscale + 0.1),
                        transform=tfp.bijectors.Softplus(
                            low=to_default_float(lengthscale)
                        ),
                    )
                }
            )

        models = fit_models(X, Y, Y_lengths, NoiseModel, priors, 2, verbose)

    detrended = []
    for m in models:
        y_curr = m.y
        y_trend = m.mean

        y_detrended = y_curr - y_trend
        y_detrended = y_detrended - mean(y_detrended)

        detrended.append(y_detrended)

    return detrended, models


def background_noise(
    X: Ndarray,
    Y: Ndarray,
    Y_lengths: NDArray[int32],
    verbose: bool = False,
) -> Tuple[float64, List[GaussianProcess]]:
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
    Tuple[Tensor, list[GaussianProcess]]
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


def pad_zeros(X: Sequence[Ndarray]) -> Ndarray:
    """
    Pad zeros to the end of each array in a list of arrays

    Parameters
    ----------
    X: List[Ndarray]
        List of arrays

    Returns
    -------
    Ndarray
        Padded array
    """
    max_length = max([len(x) for x in X])
    result_array = zeros((len(X), max_length))

    for i, x in enumerate(X):
        result_array[i, : len(x)] = x.reshape(-1)

    return result_array.T


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
    data_cols = [col for col in df if col.startswith("Cell")]  # type: ignore
    bckgd_cols = [col for col in df if col.startswith("Background")]  # type: ignore
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

    print(time.shape)
    print(y_all.shape)

    return time, bckgd, bckgd_length, M, y_all, y_length, N
