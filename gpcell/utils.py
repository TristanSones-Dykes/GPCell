# Standard Library Imports
from typing import (
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
import operator

# Third-Party Library Imports
import pandas as pd
import tensorflow_probability as tfp

# Direct Namespace Imports
from numpy import float64, nonzero, std, mean, max

from gpflow import Parameter
from gpflow.kernels import RBF, Matern12, Cosine
from gpflow.utilities import to_default_float

# Internal Project Imports
from gpcell.backend import (
    Ndarray,
    GaussianProcess,
    GPRConstructor,
    GPKernel,
    GPPriorFactory,
    GPPriorTrainingFlag,
    GPOperator,
    GPPrior,
    Numeric,
)


# ---------------------------------#
# --- gpcell utility functions --- #
# ---------------------------------#


@overload
def fit_processes(
    X: Sequence[Ndarray],
    Y: Sequence[Ndarray],
    kernels: GPKernel,
    prior_gen: Union[GPPriorFactory, Sequence[GPPriorFactory]],
    replicates: Literal[1] = 1,
    trainable: GPPriorTrainingFlag = {},
    operator: Optional[GPOperator] = operator.mul,
    preprocess: int = 0,
    Y_var: bool = False,
    verbose: bool = False,
) -> List[List[GaussianProcess]]: ...


@overload
def fit_processes(
    X: Sequence[Ndarray],
    Y: Sequence[Ndarray],
    kernels: GPKernel,
    prior_gen: Union[GPPriorFactory, Sequence[GPPriorFactory]],
    replicates: int,
    trainable: GPPriorTrainingFlag = {},
    operator: Optional[GPOperator] = operator.mul,
    preprocess: int = 0,
    Y_var: bool = False,
    verbose: bool = False,
) -> Iterable[List[GaussianProcess]]: ...


def fit_processes(
    X: Sequence[Ndarray],
    Y: Sequence[Ndarray],
    kernels: GPKernel,
    prior_gen: GPPriorFactory | Sequence[GPPriorFactory],
    replicates: int = 1,
    trainable: GPPriorTrainingFlag = {},
    operator: Optional[GPOperator] = operator.mul,
    preprocess: int = 0,
    Y_var: bool = False,
    verbose: bool = False,
) -> List[List[GaussianProcess]] | Iterable[List[GaussianProcess]]:
    """
    Fit Gaussian Processes to the data according the the number of prior generators and replicates.

    N traces and M replicates will result in N * M models. If N generators are provided, each generator will
    be used for M replicates on the corresponding trace.

    If there is only one replicate, the output will be a list of `[[models]]`. If there are multiple replicates, the output
    will be an iterator of lists of models.

    Parameters
    ----------
    X: List[Ndarray]
            List of input domains
    Y: List[Ndarray]
            List of input traces
    kernels: GPKernel
            Kernel(s) in the model
    prior_gen: GPPriorFactory
            Function that generates priors for the model
    replicates: int
            Number of replicates to fit for each prior generator
    trainable: GPPriorTrainingFlag
            Dictionary to set trainable parameters (all are trainable by default)
    operator: Optional[GPOperator]
            Operator to combine multiple kernels,
    preprocess: int
            Preprocessing option (0: None, 1: Centre, 2: Standardise)
    Y_var: bool
            Calculate variance of missing data
    verbose: bool
            Print information

    Returns
    -------
    List[GaussianProcess]
            List of fitted models
    """
    # Generators and dimension checks
    match prior_gen:
        case gen if callable(gen):
            constructors = [
                GPRConstructor(kernels, gen, trainable, operator) for _ in Y
            ]
        case list():
            assert len(prior_gen) == len(Y), ValueError(
                f"Number of generators ({len(prior_gen)}) must match number of traces ({len(Y)})"
            )
            constructors = [
                GPRConstructor(kernels, gen, trainable, operator) for gen in prior_gen
            ]

    # Preprocess traces
    match preprocess:
        case 0:
            Y_processed = Y
        case 1:
            Y_processed = [(y - mean(y)) for y in Y]
        case 2:
            Y_processed = [(y - mean(y)) / std(y) for y in Y]

    # Fit processes
    match replicates:
        case 1:
            processes = [GaussianProcess(constructor) for constructor in constructors]
            for process, x, y in zip(processes, X, Y_processed):
                process.fit(x, y, Y_var, verbose)

            return [processes]

        case int(r) if r > 1:

            def iterate_processes():
                for x, y, constructor in zip(X, Y_processed, constructors):
                    new_processes = [
                        GaussianProcess(constructor) for _ in range(replicates)
                    ]
                    for i, process in enumerate(new_processes):
                        process.fit(x, y, Y_var, verbose)

                    yield new_processes

            return iterate_processes()
        case _:
            raise ValueError(f"Invalid number of replicates: {replicates}")


def detrend(
    X: Sequence[Ndarray],
    Y: Sequence[Ndarray],
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
    # Preprocess traces
    Y_standardised = [(y - mean(y)) / std(y) for y in Y]

    # Create prior generator(s)
    match detrend_lengthscale:
        case float():
            prior_gen = lambda: {  # noqa: E731
                "kernel.lengthscales": Parameter(
                    to_default_float(detrend_lengthscale + 0.1),
                    transform=tfp.bijectors.Softplus(
                        low=to_default_float(detrend_lengthscale)
                    ),
                ),
            }
        case int():
            prior_gen = [
                lambda: {
                    "kernel.lengthscales": Parameter(
                        to_default_float(len(y) / detrend_lengthscale + 0.1),
                        transform=tfp.bijectors.Softplus(
                            low=to_default_float(len(y) / detrend_lengthscale)
                        ),
                    ),
                }
                for y in Y
            ]

    # Fit detrending processes
    GPs = fit_processes(X, Y_standardised, RBF, prior_gen, verbose=verbose)[0]

    # Detrend traces
    detrended = []
    for x, y, m in zip(X, Y_standardised, GPs):
        y_trend = m(x)[0]
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
    Fit RBF processes to the input traces and calculate overall noise

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
            Overall noise, list of noise processes
    """

    # Create prior generator
    def prior_gen():
        return {
            "kernel.lengthscales": Parameter(
                to_default_float(lengthscale + 0.1),
                transform=tfp.bijectors.Softplus(low=to_default_float(lengthscale)),
            ),
        }

    # Preprocess traces
    Y_centred = [y - mean(y) for y in Y]

    # Fit noise processes
    processes = fit_processes(
        X, Y_centred, RBF, prior_gen, Y_var=True, verbose=verbose
    )[0]

    # Calculate noise
    std_array = [gp.noise for gp in processes]
    std = mean(std_array)

    if verbose:
        print("\nBackground noise results:")
        print(f"Standard deviation: {std}")

    return std, processes


def fit_ou(
    X: List[Ndarray],
    Y: List[Ndarray],
    lengthscale_prior: Callable[..., GPPrior] | List[Numeric],
    variance_prior: Callable[..., GPPrior] | List[Numeric],
    likelihood_variance: Callable[..., GPPrior] | List[Numeric],
    trainable_likelihood_variance: bool = True,
    K: int = 1,
) -> List[List[GaussianProcess]] | Iterable[List[GaussianProcess]]:
    """
    Fit Ornstein-Uhlenbeck process to the input trace

    Parameters
    ----------
    X: Ndarray
            Input domain
    Y: Ndarray
            Input trace
    lengthscale_prior: Callable[..., GPPrior]
            Prior for the lengthscale parameter
    variance_prior: Callable[..., GPPrior]
            Prior for the variance parameter
    likelihood_variance: Callable[..., GPPrior]
            Prior for the likelihood variance
    K: int
            Number of replicates

    Returns
    -------
    Tuple[List[float], List[float], List[Kernel], List[Kernel], List[float]]
            List of fitted parameters
    """
    # define priors
    prior_list = []
    for param in [lengthscale_prior, variance_prior, likelihood_variance]:
        if callable(param):
            prior_list.append([param for _ in range(len(Y))])
        else:
            prior_list.append(param)

    ou_priors = [
        lambda: {
            "kernel.lengthscales": lengthscale,
            "kernel.variance": variance,
            "likelihood.variance": likelihood,
        }
        for lengthscale, variance, likelihood in zip(*prior_list)
    ]
    ou_trainable = {"likelihood.variance": trainable_likelihood_variance}

    # fit OU processes
    return fit_processes(
        X, Y, Matern12, ou_priors, replicates=K, trainable=ou_trainable
    )


def fit_ou_osc(
    X: List[Ndarray],
    Y: List[Ndarray],
    ou_lengthscale_prior: Callable[..., GPPrior] | List[Numeric],
    ou_variance_prior: Callable[..., GPPrior] | List[Numeric],
    osc_lengthscale_prior: Callable[..., GPPrior] | List[Numeric],
    likelihood_variance: Callable[..., GPPrior] | List[Numeric],
    ou_trainable_variance: bool = True,
    osc_trainable_variance: bool = True,
    trainable_likelihood_variance: bool = True,
    K: int = 1,
) -> List[List[GaussianProcess]] | Iterable[List[GaussianProcess]]:
    """
    Fit Ornstein-Uhlenbeck * Cosine process to the input trace

    Parameters
    ----------
    X: Ndarray
            Input domain
    Y: Ndarray
            Input trace
    ou_lengthscale_prior: Callable[..., GPPrior]
            Prior for the OU lengthscale parameter
    ou_variance_prior: Callable[..., GPPrior]
            Prior for the OU variance parameter
    osc_lengthscale_prior: Callable[..., GPPrior]
            Prior for the cosine lengthscale parameter
    likelihood_variance: Callable[..., GPPrior]
            Prior for the likelihood variance
    ou_trainable_variance: bool
            Trainable variance for OU process
    osc_trainable_variance: bool
            Trainable variance for cosine process
    trainable_likelihood_variance: bool
            Trainable likelihood variance
    K: int
            Number of replicates

    Returns
    -------
    List[List[GaussianProcess]] | Iterable[List[GaussianProcess]]
            Singleton list or iterator of lists of fitted processes
    """
    # define priors
    prior_list = []
    for param in [
        ou_lengthscale_prior,
        ou_variance_prior,
        osc_lengthscale_prior,
        likelihood_variance,
    ]:
        if callable(param):
            prior_list.append([param for _ in range(len(Y))])
        else:
            prior_list.append(param)

    ou_osc_priors = [
        lambda: {
            "kernel.kernels[0].lengthscales": ou_lengthscale,
            "kernel.kernels[0].variance": ou_variance,
            "kernel.kernels[1].lengthscales": osc_lengthscale,
            "likelihood.variance": likelihood,
        }
        for ou_lengthscale, ou_variance, osc_lengthscale, likelihood in zip(*prior_list)
    ]
    ou_osc_trainable = {
        "likelihood.variance": trainable_likelihood_variance,
        (0, "variance"): ou_trainable_variance,
        (1, "variance"): osc_trainable_variance,
    }

    # fit OU_OSC processes
    return fit_processes(
        X,
        Y,
        [Matern12, Cosine],
        ou_osc_priors,
        replicates=K,
        trainable=ou_osc_trainable,
    )


def load_data(
    path: str, X_name: str, Y_name: str
) -> Tuple[
    List[Ndarray],
    List[Ndarray],
]:
    """
    Loads experiment data from a csv file. Taking domain name and trace prefix as input.

    Parameters
    ----------
    path: str
            Path to the csv file
    X_name: str
            Name of the domain column
    Y_name: str
            Prefix of the trace columns

    Returns
    -------
    Tuple[List[Ndarray], List[Ndarray]]
            List of domain data, list of trace data
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
