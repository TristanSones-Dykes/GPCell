# Standard Library Imports
from typing import Callable, List, Sequence, Tuple, Type, Union

# Third-Party Library Imports
import pandas as pd

# Direct Namespace Imports
from numpy import float64, nonzero, std, mean, max

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
    X: Sequence[Ndarray],
    Y: Sequence[Ndarray],
    model: Type[GPModel],
    priors: Union[Callable[..., GPPriors], Sequence[GPPriors]],
    preprocess: int = 0,
    verbose: bool = False,
) -> List[GaussianProcess]:
    """
    Fit a Gaussian Process model to each trace

    Parameters
    ----------
    X: List[Ndarray]
        List of input domains
    Y: List[Ndarray]
        List of input traces
    model: GPModel
        Model to fit
    priors: Callable[..., GPPriors] | List[GPPriors]
        Function that generates priors for each trace, or list of priors
    preprocess: int
        Preprocessing option (0: None, 1: Centre, 2: Standardise)
    verbose: bool
        Print information

    Returns
    -------
    List[GaussianProcess]
        List of fitted models
    """
    if callable(priors):
        prior_list = [priors() for _ in Y]
    elif hasattr(priors, "__iter__"):
        prior_list = priors

    processes = []

    for x_curr, y_curr, prior in zip(X, Y, prior_list):
        if preprocess == 1:
            y_curr = y_curr - mean(y_curr)
        elif preprocess == 2:
            y_curr = (y_curr - mean(y_curr)) / std(y_curr)

        gp_model = model(prior)
        m = GaussianProcess(gp_model)
        m.fit(x_curr, y_curr, verbose=verbose)

        processes.append(m)

    return processes


def fit_models_replicates(
    N: int,
    X: List[Ndarray],
    Y: List[Ndarray],
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
    X: List[Ndarray]
        List of input domains
    Y: List[Ndarray]
        List of input traces
    model: GPModel
        Model to fit
    prior_gen: Callable[..., GPPriors] | List[Callable[..., GPPriors]]
        Function that generates priors for each replicate, or list of functions for each trace
    preprocess: int
        Preprocessing option (0: None, 1: Centre, 2: Standardise)
    verbose: bool
        Print information

    Returns
    -------
    List[List[GaussianProcess]]
        List of N fitted processes for each trace
    """
    if isinstance(prior_gen, list):
        assert len(prior_gen) == len(Y)
        priors = prior_gen
    else:
        priors = [prior_gen] * len(Y)

    # preprocess data
    match preprocess:
        case 0:
            Y_processed = Y
        case 1:
            Y_processed = [y - mean(y) for y in Y]
        case 2:
            Y_processed = [(y - mean(y)) / std(y) for y in Y]
        case _:
            raise ValueError("Invalid preprocess option")

    GPs: List[List[GaussianProcess]] = []
    for i, (x, y) in enumerate(zip(X, Y_processed)):
        x_list, y_list = [x] * N, [y] * N
        GPs.append(fit_models(x_list, y_list, model, priors[i], 0, verbose))

    return GPs


def detrend(
    X: List[Ndarray],
    Y: List[Ndarray],
    detrend_lengthscale: Union[float, int],
    verbose: bool = False,
) -> Tuple[List[Ndarray], List[GaussianProcess]]:
    """
    Detrend stochastic process using RBF process

    Parameters
    ----------
    X: List[Ndarray]
        List of input domains
    Y: List[Ndarray]
        List of input traces
    detrend_lengthscale: float | int
        Lengthscale of the detrending process, or integer portion of trace length
    verbose: bool
        Print information

    Returns
    -------
    Tuple[List[Ndarray], List[GaussianProcess]]
        Detrended traces, list of fit models
    """
    # Set priors
    """"""
    match detrend_lengthscale:
        case int():
            priors = [
                {
                    "lengthscale": Parameter(
                        to_default_float(len(y) / detrend_lengthscale + 0.1),
                        transform=tfp.bijectors.Softplus(
                            low=to_default_float(len(y) / detrend_lengthscale)
                        ),
                    )
                }
                for y in Y
            ]
        case float():
            priors = [
                {
                    "lengthscale": Parameter(
                        to_default_float(detrend_lengthscale + 0.1),
                        transform=tfp.bijectors.Softplus(
                            low=to_default_float(detrend_lengthscale)
                        ),
                    )
                }
                for _ in Y
            ]
        case _:
            raise TypeError(
                f"Invalid type for detrend_lengthscale: {type(detrend_lengthscale)}"
            )

    # Standardise traces
    Y_standardised = [(y - mean(y)) / std(y) for y in Y]

    # Fit RBF models
    GPs = fit_models(
        X, Y_standardised, NoiseModel, priors, preprocess=0, verbose=verbose
    )

    # Detrend traces
    detrended = []
    for y, m in zip(Y_standardised, GPs):
        y_trend = m.mean
        y_detrended = y - y_trend
        y_detrended = y_detrended - mean(y_detrended)
        detrended.append(y_detrended)

    return detrended, GPs


def background_noise(
    X: List[Ndarray],
    Y: List[Ndarray],
    lengthscale: float,
    verbose: bool = False,
) -> Tuple[float64, List[GaussianProcess]]:
    """
    Fit a background noise model to the data and return the standard deviation of the overall noise

    Parameters
    ----------
    X: List[Ndarray]
        List of input domains
    Y: List[Ndarray]
        List of input traces
    lengthscale: float
        Lengthscale of the noise model
    verbose: bool
        Print information

    Returns
    -------
    Tuple[float64, List[GaussianProcess]]
        Standard deviation of the overall noise, list of noise models
    """
    std_array = []

    priors = [
        {
            "lengthscale": Parameter(
                to_default_float(lengthscale + 0.1),
                transform=tfp.bijectors.Softplus(low=to_default_float(lengthscale)),
            ),
        }
        for _ in Y
    ]
    models = fit_models(X, Y, NoiseModel, priors, preprocess=1, verbose=verbose)

    for noise_model in models:
        std_array.append(noise_model.fit_gp.likelihood.variance**0.5)  # type: ignore

    std = mean(std_array)

    if verbose:
        print("Background noise model:")
        print(f"Standard deviation: {std}")

    return std, models


def load_data(
    path: str, X_name: str, Y_name: str
) -> Tuple[
    List[Ndarray],
    List[Ndarray],
]:
    """
    Loads experiment data from a csv file. Taking domain name and trace prefix as input.

    :param str path: Path to the csv file.
    :param str X_name: Name of the domain column.
    :param str Y_name: Name of the trace column.

    :return: Tuple of domain and trace data.
    """
    df = pd.read_csv(path).fillna(0)

    # Extract domain and trace data
    Y_cols = [col for col in df if col.startswith(Y_name)]  # type: ignore
    Y_data = [df[col].to_numpy() for col in Y_cols]
    X = df[X_name].to_numpy()

    # Filter out zero traces and adjust domains
    X_data = []
    Y_data_filtered = []

    for y in Y_data:
        y_length = max(nonzero(y))
        X_data.append(X[:y_length].reshape(-1, 1))
        Y_data_filtered.append(y[:y_length].reshape(-1, 1))

    return X_data, Y_data_filtered
