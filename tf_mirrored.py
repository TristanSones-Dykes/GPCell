# Standard Library Imports
from typing import List, Optional, Sequence, Union
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import partial

# Third-Party Library Imports
from joblib import Parallel, delayed
import numpy as np

# Direct Namespace Imports
from numpy import std, mean
from numpy.random import uniform
from gpflow.kernels import Matern12, Cosine
from gpflow.utilities import print_summary

# Internal Project Imports
from gpcell.utils import load_data, background_noise, detrend, fit_processes
from gpcell.backend import (
    GaussianProcess,
    GPRConstructor,
    GPKernel,
    GPPriorFactory,
    GPPriorTrainingFlag,
    GPOperator,
)

# ---------------------#
# --- Joblib setup --- #
# ---------------------#


def ou_prior(noise: np.float64):
    """Top-level OU prior generator."""
    return {
        "kernel.lengthscales": uniform(0.1, 2.0),
        "kernel.variance": uniform(0.1, 2.0),
        "likelihood.variance": noise**2,
    }


def ouosc_prior(noise: np.float64):
    """Top-level OU+Oscillator prior generator."""
    return {
        "kernel.kernels[0].lengthscales": uniform(0.1, 2.0),
        "kernel.kernels[0].variance": uniform(0.1, 2.0),
        "kernel.kernels[1].lengthscales": uniform(0.1, 4.0),
        "likelihood.variance": noise**2,
    }


# Worker function to fit all replicates for one cell.
def joblib_fit_trace_worker(
    index: int,
    constructor: GPRConstructor,
    replicates: int,
    Y_var: bool,
    verbose: bool,
    X: Sequence[np.ndarray],
    Y: Sequence[np.ndarray],
) -> List[GaussianProcess]:
    """
    Fits all replicates for the cell at the given index.
    """
    x = X[index]
    y = Y[index]
    models = []
    for _ in range(replicates):
        gp_model = GaussianProcess(constructor)
        gp_model.fit(x, y, Y_var, verbose)
        models.append(gp_model)
    return models


def joblib_fit_models(
    X: Sequence[np.ndarray],
    Y: Sequence[np.ndarray],
    kernels: GPKernel,
    prior_gen: Union[GPPriorFactory, Sequence[GPPriorFactory]],
    replicates: int = 1,
    trainable: GPPriorTrainingFlag = {},
    operator: Optional[GPOperator] = operator.mul,
    preprocess: int = 0,
    Y_var: bool = False,
    verbose: bool = False,
) -> List[List[GaussianProcess]]:
    """
    Fit GP models to each cell in parallel using Joblib.

    This function groups the replicates by cell. Each cell's data is passed once
    to the worker, and the worker fits all replicates for that cell.

    Parameters
    ----------
    X : Sequence[np.ndarray]
        List of input domains (one per cell).
    Y : Sequence[np.ndarray]
        List of cell traces (one per cell).
    kernels : GPKernel
        Kernel (or list of kernels) for the GP model.
    prior_gen : GPPriorFactory or Sequence[GPPriorFactory]
        A callable (or list of callables) that generate the priors for each model.
    replicates : int, optional
        Number of replicates to fit per cell (default is 1).
    trainable : GPPriorTrainingFlag, optional
        Dictionary to set which parameters are trainable (default is {}).
    operator : Optional[GPOperator], optional
        Operator to combine kernels (defaults to multiplication if not provided).
    preprocess : int, optional
        0: no preprocessing; 1: centre the trace; 2: standardise the trace (default is 0).
    Y_var : bool, optional
        Whether to calculate variance of missing data (default is False).
    verbose : bool, optional
        If True, prints information during fitting (default is False).

    Returns
    -------
    List[List[GaussianProcess]]
        A list (one per cell) of lists of fitted GP models (one per replicate).
    """
    # Build constructors for each trace.
    if callable(prior_gen):
        constructors = [
            GPRConstructor(kernels, prior_gen, trainable, operator) for _ in Y
        ]
    else:
        if len(prior_gen) != len(Y):
            raise ValueError(
                f"Number of generators ({len(prior_gen)}) must match number of traces ({len(Y)})"
            )
        constructors = [
            GPRConstructor(kernels, gen, trainable, operator) for gen in prior_gen
        ]

    # Preprocess traces if needed.
    if preprocess == 0:
        Y_processed = Y
    elif preprocess == 1:
        Y_processed = [y - mean(y) for y in Y]
    elif preprocess == 2:
        Y_processed = [(y - mean(y)) / std(y) for y in Y]
    else:
        Y_processed = Y

    # Use Joblib Parallel to run one task per cell.
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(joblib_fit_trace_worker)(
            i, constructors[i], replicates, Y_var, verbose, X, Y_processed
        )
        for i in range(len(Y))
    )
    return results  # type: ignore


# -----------------------#
# --- Threaded setup --- #
# -----------------------#


def parallel_fit_models(
    X: Sequence[np.ndarray],
    Y: Sequence[np.ndarray],
    kernels: GPKernel,
    prior_gen: GPPriorFactory | Sequence[GPPriorFactory],
    replicates: int = 1,
    trainable: GPPriorTrainingFlag = {},
    operator: Optional[GPOperator] = operator.mul,
    preprocess: int = 0,
    Y_var: bool = False,
    verbose: bool = False,
) -> List[List[GaussianProcess]]:
    """
    Fit GP models to each cell in parallel using threads.

    This function groups the replicates by cell, so that for each trace (cell),
    its replicates are fit within one task—minimizing data copying overhead.

    Parameters
    ----------
    X : Sequence[np.ndarray]
        List of input domains (one per cell).
    Y : Sequence[np.ndarray]
        List of cell traces (one per cell).
    kernels : GPKernel
        Kernel (or list of kernels) for the GP model.
    prior_gen : GPPriorFactory or Sequence[GPPriorFactory]
        A callable (or list of callables) that generate the priors for each model.
    replicates : int, optional
        Number of replicates to fit per cell (default is 1).
    trainable : GPPriorTrainingFlag, optional
        Dictionary to set which parameters are trainable (default is {}).
    operator : GPOperator, optional
        Operator to combine kernels (defaults to multiplication if not provided).
    preprocess : int, optional
        0: no preprocessing; 1: centre the trace; 2: standardise the trace (default is 0).
    Y_var : bool, optional
        Whether to calculate variance of missing data (default is False).
    verbose : bool, optional
        If True, prints information during fitting (default is False).

    Returns
    -------
    List[List[GaussianProcess]]
        A list of lists. Each outer list element corresponds to one cell; its inner list contains the fitted replicates.
    """
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

    # Define a helper function that fits all replicates for one cell.
    def fit_models_for_trace(
        x: np.ndarray, y: np.ndarray, constructor: GPRConstructor
    ) -> List[GaussianProcess]:
        models = []
        for _ in range(replicates):
            gp_model = GaussianProcess(constructor)
            gp_model.fit(x, y, Y_var, verbose)
            models.append(gp_model)
        return models

    results: List[List[GaussianProcess]] = [None] * len(Y)  # type: ignore
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(fit_models_for_trace, x, y, constructor): i
            for i, (x, y, constructor) in enumerate(zip(X, Y_processed, constructors))
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Fitting for trace {idx} raised an exception: {e}")
    return results


# ---------------------------------#
# --- Comparing the strategies --- #
# ---------------------------------#

if __name__ == "__main__":
    # --- Defining params --- #
    path, X_name, Y_name, background_name = (
        "data/hes/Hes1_example.csv",
        "Time (h)",
        "Cell",
        "Background",
    )
    verbose = True

    # --- Load data and preprocess --- #
    X, Y = load_data(path, X_name, Y_name)
    X_bckgd, bckgd = load_data(path, X_name, background_name)
    N, M = len(Y), len(bckgd)

    # background noise and detrending
    mean_noise, bckgd_GPs = background_noise(X_bckgd, bckgd, 7.0, verbose=verbose)
    noise_list = [mean_noise / std(y) for y in Y]
    Y_detrended, detrend_GPs = detrend(X, Y, 7.0, verbose=verbose)

    # --------------------------------#
    # --- Sequential and Threaded --- #
    # --------------------------------#

    ouosc_kernel = [Matern12, Cosine]
    ou_kernel = Matern12

    ou_priors = [
        lambda noise=noise: {
            "kernel.lengthscales": uniform(0.1, 2.0),
            "kernel.variance": uniform(0.1, 2.0),
            "likelihood.variance": noise**2,
        }
        for noise in noise_list
    ]
    ouosc_priors = [
        lambda noise=noise: {
            "kernel.kernels[0].lengthscales": uniform(0.1, 2.0),
            "kernel.kernels[0].variance": uniform(0.1, 2.0),
            "kernel.kernels[1].lengthscales": uniform(0.1, 4.0),
            "likelihood.variance": noise**2,
        }
        for noise in noise_list
    ]

    # set trainables
    ou_trainables = {"likelihood.variance": False}
    ouosc_trainables = {
        "likelihood.variance": False,
        (1, "variance"): False,
    }

    replicates = 10

    # --- Time the parallel strategy --- #
    start_parallel = time.perf_counter()
    ouosc_models_parallel = parallel_fit_models(
        X,
        Y_detrended,
        ouosc_kernel,
        ouosc_priors,
        trainable=ouosc_trainables,
        replicates=replicates,
    )
    ou_models_parallel = parallel_fit_models(
        X,
        Y_detrended,
        ou_kernel,
        ou_priors,
        trainable=ou_trainables,
        replicates=replicates,
    )
    end_parallel = time.perf_counter()

    # --- Time the sequential strategy --- #
    start_seq = time.perf_counter()
    ou_kernel = Matern12
    ouosc_kernel = [Matern12, Cosine]

    # fit processes
    ou_GPs = list(
        fit_processes(
            X,
            Y_detrended,
            ou_kernel,
            ou_priors,
            replicates=replicates,
            trainable=ou_trainables,
        )
    )
    ouosc_GPs = list(
        fit_processes(
            X,
            Y_detrended,
            ouosc_kernel,
            ouosc_priors,
            replicates=replicates,
            trainable=ouosc_trainables,
        )
    )
    end_seq = time.perf_counter()

    # ---------------#
    # --- Joblib --- #
    # ---------------#

    # Create pickle-able priors
    proc_ou_priors = [partial(ou_prior, noise) for noise in noise_list]
    proc_ouosc_priors = [partial(ouosc_prior, noise) for noise in noise_list]

    # --- Time the Joblib strategy --- #
    start_joblib = time.perf_counter()
    ouosc_models_joblib = joblib_fit_models(
        X,
        Y_detrended,
        ouosc_kernel,
        proc_ouosc_priors,
        trainable=ouosc_trainables,
        replicates=replicates,
        verbose=True,
    )
    ou_models_joblib = joblib_fit_models(
        X,
        Y_detrended,
        ou_kernel,
        proc_ou_priors,
        trainable=ou_trainables,
        replicates=replicates,
        verbose=True,
    )
    end_joblib = time.perf_counter()

    # --- Print summary --- #

    # parallel
    ouosc_parallel = ouosc_models_parallel[0]
    ou_parallel = ou_models_parallel[0]
    best_ouosc = min(ouosc_parallel, key=lambda m: m.log_posterior())  # type: ignore
    best_ou = min(ou_parallel, key=lambda m: m.log_posterior())  # type: ignore
    print_summary(best_ouosc.fit_gp)
    print_summary(best_ou.fit_gp)
    print("")

    # sequential
    ouosc_seq = ouosc_GPs[0]
    ou_seq = ou_GPs[0]
    best_ouosc_seq = min(ouosc_seq, key=lambda m: m.log_posterior())  # type: ignore
    best_ou_seq = min(ou_seq, key=lambda m: m.log_posterior())  # type: ignore
    print_summary(best_ouosc_seq.fit_gp)
    print_summary(best_ou_seq.fit_gp)
    print("")

    # --- Compare times --- #
    print(f"Joblib (Processes) fitting time: {end_joblib - start_joblib:.2f} seconds")
    print(f"Threaded fitting time: {end_parallel - start_parallel:.2f} seconds")
    print(f"Sequential fitting time: {end_seq - start_seq:.2f} seconds")
