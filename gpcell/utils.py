# Standard Library Imports
from typing import (
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Third-Party Library Imports
from joblib import Parallel, delayed, dump, load
from joblib.externals.loky import get_reusable_executor
import numpy as np
import pandas as pd
import tensorflow_probability as tfp

# Direct Namespace Imports
from numpy import float64, nonzero, std, mean, max
from gpflow import Parameter
from gpflow.kernels import RBF
from gpflow.utilities import to_default_float


# Internal Project Imports
from gpcell.backend import (
    GaussianProcess,
    GPRConstructor,
    _joblib_fit_memmap_worker,
    _simulate_replicate_mod9,
    _simulate_replicate_mod9_nodelay,
    Ndarray,
    GPKernel,
    GPPriorFactory,
    GPPriorTrainingFlag,
    GPOperator,
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
    mcmc: bool = False,
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
    mcmc: bool = False,
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
    mcmc: bool = False,
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
    mcmc: bool
            Use MCMC for inference
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
                GPRConstructor(kernels, gen, trainable, operator, mcmc=mcmc) for _ in Y
            ]
        case list():
            assert len(prior_gen) == len(Y), ValueError(
                f"Number of generators ({len(prior_gen)}) must match number of traces ({len(Y)})"
            )
            constructors = [
                GPRConstructor(kernels, gen, trainable, operator, mcmc=mcmc)
                for gen in prior_gen
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


def fit_processes_joblib(
    X: Sequence[Ndarray],
    Y: Sequence[Ndarray],
    kernels: GPKernel,
    prior_gen: Union[GPPriorFactory, Sequence[GPPriorFactory]],
    replicates: int = 1,
    trainable: GPPriorTrainingFlag = {},
    operator: Optional[GPOperator] = operator.mul,
    preprocess: int = 0,
    mcmc: bool = False,
    Y_var: bool = False,
    verbose: bool = False,
) -> List[List[GaussianProcess]]:
    """
    Fit Gaussian Processes to the data in parallel using Joblib.

    N traces and M replicates will result in N * M models.
    This version groups the replicates by cell: each job fits all replicates for one cell,
    thereby reducing data copying overhead.

    Parameters
    ----------
    X : Sequence[Ndarray]
        List of input domains (one per cell).
    Y : Sequence[Ndarray]
        List of input traces (one per cell).
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
    mcmc : bool, optional
        Whether to use MCMC for inference (default is False).
    Y_var : bool, optional
        Whether to calculate variance of missing data (default is False).
    verbose : bool, optional
        If True, prints information during fitting (default is False).

    Returns
    -------
    List[List[GaussianProcess]]
        A list (one per cell) of lists of fitted GP models (one per replicate).
    """
    # Determine the constructors for each trace.
    if callable(prior_gen):
        constructors = [
            GPRConstructor(kernels, prior_gen, trainable, operator, mcmc=mcmc)
            for _ in Y
        ]
    else:
        if len(prior_gen) != len(Y):
            raise ValueError(
                f"Number of generators ({len(prior_gen)}) must match number of traces ({len(Y)})"
            )
        constructors = [
            GPRConstructor(kernels, gen, trainable, operator, mcmc=mcmc)
            for gen in prior_gen
        ]

    # Check if Y is homogeneous (all traces are the same length).
    if all(len(y) == len(Y[0]) for y in Y):
        # set number of cores used and job batches pre-dispatched
        n_jobs = 14
        pre_dispatch = "all"
        batch_size = "auto"

        # create homogenous matrix of traces
        Y_mat = np.stack(Y, axis=1)

        # dump to memmap file
        folder = "./temp"
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "Y_memmap")
        dump(Y_mat, path)

        # load memmap file and delete matrix
        Y_processed = load(path, mmap_mode="r")

        # mitigate memory issues
        if len(Y) > 4 * n_jobs:
            pre_dispatch = "n_jobs"
            del Y_mat
        print(
            f"\nHomogenous traces: {len(Y)} cells, pre_dispatch: {pre_dispatch}, batch_size: {batch_size}"
        )

        # run workers
        result: List[List[GaussianProcess]] = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            pre_dispatch=pre_dispatch,
            batch_size=batch_size,  # type: ignore
            # verbose=1,
        )(
            delayed(_joblib_fit_memmap_worker)(
                i, X, Y_processed, Y_var, constructors[i], replicates
            )
            for i in range(len(Y))
        )  # type: ignore

        # Clean up
        os.remove(path)
        if len(Y) > 4 * n_jobs:
            get_reusable_executor().shutdown(wait=True)

        return result

    # Y is a list of arrays, so fit each trace in parallel.
    # Preprocess traces
    match preprocess:
        case 0:
            Y_processed = Y
        case 1:
            Y_processed = [(y - mean(y)) for y in Y]
        case 2:
            Y_processed = [(y - mean(y)) / std(y) for y in Y]

    def joblib_fit_trace_worker(
        index: int, constructor: GPRConstructor
    ) -> List[GaussianProcess]:
        x = X[index]
        y = Y_processed[index]
        models = []
        for _ in range(replicates):
            gp_model = GaussianProcess(constructor)
            gp_model.fit(x, y, Y_var, verbose)
            models.append(gp_model)
        return models

    # Run the worker function in parallel using Joblib.
    results: List[List[GaussianProcess]] = Parallel(
        n_jobs=11,
        backend="loky",  # , verbose=1
    )(delayed(joblib_fit_trace_worker)(i, constructors[i]) for i in range(len(Y)))  # type: ignore
    return results


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
        print("Background noise results:")
        print(f"Standard deviation: {std}")

    return std, processes


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
    Y_data = [df[col].to_numpy(dtype=float64) for col in Y_cols]
    X = df[X_name].to_numpy(dtype=float64)

    # Filter out zero traces and adjust domains
    X_data = []
    Y_data_filtered = []

    for y in Y_data:
        y_length = max(nonzero(y))
        X_data.append(X[: y_length + 1].reshape(-1, 1))
        Y_data_filtered.append(y[: y_length + 1].reshape(-1, 1))

    return X_data, Y_data_filtered


def save_sim(X: Ndarray, Y: Sequence[Ndarray], filename: str) -> None:
    """
    Saves the simulation data to a CSV file with rows = time points and
    columns = simulated cells (plus a 'Time' column).

    Parameters
    ----------
    X : list of ndarrays
        Each element is an array of time points (all are the same in your case).
    Y : list of ndarrays
        Each element is a (num_time_points, 1) array for one simulated cell.
    filename : str
        Path to the CSV file to create.
    """
    # Number of simulated cells (columns)
    n_series = len(Y)

    # Each Y[i] is shape (num_time_points, 1).
    # Stack them horizontally to get a (num_time_points, n_series) 2D array.
    data = np.hstack(Y)  # shape becomes (num_time_points, n_series)

    # Build a DataFrame with each column labeled "Cell 1", "Cell 2", etc.
    col_names = [f"Cell {i + 1}" for i in range(n_series)]
    df = pd.DataFrame(data, columns=col_names, dtype=float64)

    # Insert the time as its own column at the front
    df.insert(0, "Time", X)

    # Write to CSV (no index column, since we include time explicitly)
    df.to_csv(filename, index=False)


def load_sim(path: str) -> Tuple[Ndarray, Sequence[Ndarray]]:
    """
    Reads simulation data from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    Tuple[Sequence[Ndarray], Sequence[Ndarray]]
        - x - Sequence of time vectors.
        - data - Sequence of 2D arrays in which each column is a time series.
    """
    df = pd.read_csv(path)

    # Extract time and data
    x = df["Time"].to_numpy(dtype=float64)
    data = df.drop(columns="Time").to_numpy(dtype=float64)

    # Convert to list of ndarrays as 2D column vectors.
    data_list = [data[:, i].reshape(-1, 1).flatten() for i in range(data.shape[1])]

    return x, data_list


def gillespie_timing_mod9(
    N: int,
    par: np.ndarray,
    n_cells: int,
    mstart: int,
    pstart: int,
    out_names: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs Gillespie algorithm with delay processes in parallel.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
         2D array (totalreps x len(Output_Times)) of m values,
         2D array (totalreps x len(Output_Times)) of p values.
    """
    num_times = len(out_names)
    mout = np.zeros((n_cells, num_times), dtype=int)
    pout = np.zeros((n_cells, num_times), dtype=int)

    # Prepare arguments for each replicate.
    args = [(N, par, mstart, pstart, out_names) for _ in range(n_cells)]

    with ProcessPoolExecutor() as executor:
        # Submit each replicate as a separate process.
        futures = {
            executor.submit(_simulate_replicate_mod9, *arg): i
            for i, arg in enumerate(args)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                m_rep, p_rep = future.result()
                mout[i, :] = m_rep
                pout[i, :] = p_rep
                # if (i + 1) % 10 == 0:
                #     print(f"gillespie_timing_mod9: Replicate {i + 1} finished.")
            except Exception as e:
                print(
                    f"gillespie_timing_mod9: Replicate {i + 1} raised an exception: {e}"
                )

    return mout, pout


def gillespie_timing_mod9_nodelay(
    N: int,
    par: np.ndarray,
    totalreps: int,
    mstart: int,
    pstart: int,
    out_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs Gillespie algorithm without delay processes in parallel.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
         2D array (totalreps x len(Output_Times)) of m values,
         2D array (totalreps x len(Output_Times)) of p values.
    """
    num_times = len(out_times)
    mout = np.zeros((totalreps, num_times), dtype=int)
    pout = np.zeros((totalreps, num_times), dtype=int)

    args = [(N, par, mstart, pstart, out_times) for _ in range(totalreps)]

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_simulate_replicate_mod9_nodelay, *arg): i
            for i, arg in enumerate(args)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                m_rep, p_rep = future.result()
                mout[i, :] = m_rep
                pout[i, :] = p_rep
                # if (i + 1) % 10 == 0:
                #     print(f"gillespie_timing_mod9_nodelay: Replicate {i + 1} finished.")
            except Exception as e:
                print(
                    f"gillespie_timing_mod9_nodelay: Replicate {i + 1} raised an exception: {e}"
                )

    return mout, pout


def get_time_series(
    par1: np.ndarray,
    par2: np.ndarray,
    t_final: float,
    noise: float,
    n_cells: int,
    path: Optional[str] = None,
    mode: str = "x",
) -> Tuple[Ndarray, Sequence[Ndarray]]:
    """
    Recreates the MATLAB GetTimeSeries.m functionality in Python.

    Parameters
    ----------
    par1 : Sequence[Numeric]
        Parameter vector for simulation #1.
    par2 : Sequence[Numeric]
        Parameter vector for simulation #2.
    t_final : float
        Final simulation time (e.g., 1500).
    noise : float
        Noise level (e.g., sqrt(0.1)).
    n_cells : int
        Number of simulation replicates.
    path : str, optional
        Path to check for existing data files (default is None, no loading or writing).
    mode : str, optional
        Mode for saving data. `r` will read only/check for param combination, `w` will overwrite, `x` is for exclusive creation (default is "x")

    Returns
    -------
    Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]
        - x - Sequence of time vectors.
        - dataNORMED - Sequence of 2D arrays in which each column is a normalized time series.
    """
    modes = {"x", "r", "w"}

    # bound and type check params
    match mode, path:
        # --- no path => no file I/O --- #
        case _, None:
            pass

        # --- mode and path type checks --- #
        case mode, _ if not isinstance(mode, str):
            raise TypeError(f"Invalid mode type: {type(mode)}")
        case _, path if not isinstance(path, str):
            raise TypeError(f"Invalid path type: {type(path)}")

        # --- mode and path value checks --- #
        case mode, _ if mode not in modes:
            raise ValueError(f"Invalid mode: {mode}, must be in {modes}")
        case _, path if not path.endswith(".csv"):
            raise ValueError(f"Invalid file extension: {path}, must be .csv")

        # --- file I/O --- #
        case mode, path:
            exists = os.path.exists(path)
            match mode, exists:
                # --- w mode --- #
                case "w", _:
                    # remove file if it exists
                    if exists:
                        os.remove(path)
                    print(
                        f"Overwrite: File {'exists' if exists else "doesn't exist"} at {path}"
                    )
                # --- x and r modes --- #
                # --- path does not exist --- #
                case "r", False:
                    # read-only path provided but does not exist
                    raise FileNotFoundError(
                        f"Cannot read from non-existent file: {path}"
                    )
                case "x", False:
                    # exclusive-write path provided but does not exist
                    print(f"Exclusive: File does not exist at {path}, simulating")
                    pass
                # --- path exists --- #
                case mode, True:
                    print(f"Simulation data found at {path}, mode: {mode}")
                    return load_sim(path)
                case _:
                    raise ValueError(
                        f"Invalid combination of mode ({mode}) and path ({path})"
                    )
        case _:
            raise ValueError(f"Invalid combination of mode ({mode}) and path ({path})")

    # MATLAB: Output_Times = 5000:30:(5000+Tfinal);
    out_times = np.arange(5000, 5000 + t_final + 1, 30)

    # Initial conditions for m and p.
    N = 20
    mstart = 1 * N
    pstart = 30 * N

    # Run the simulations
    mout1, pout1 = gillespie_timing_mod9_nodelay(
        N, par1, n_cells, mstart, pstart, out_times
    )
    mout2, pout2 = gillespie_timing_mod9(N, par2, n_cells, mstart, pstart, out_times)

    # In MATLAB, x = Output_Times' and then x = (x-5000)/60.
    x = (out_times.astype(float) - 5000) / 60.0

    # Transpose p-values so that rows correspond to times.
    pout1, pout2 = pout1.T, pout2.T
    pout = np.hstack((pout1, pout2))

    # Create list of ndarrays as column vectors.
    sim_list = [pout[:, i] for i in range(pout.shape[1])]

    # Process the data
    samp = len(x)
    processed_list = []
    for y1 in sim_list:
        # Normalize the data.
        y_centred = y1 - mean(y1)
        y_std = std(y_centred)
        if y_std == 0:
            y_std = 1

        y_normal = y_centred / y_std

        # Add noise.
        SIGMA = np.diag((noise**2) * np.ones(samp))
        measerror = np.random.multivariate_normal(np.zeros(samp), SIGMA)
        y_processed = y_normal + measerror
        processed_list.append(y_processed.reshape(-1, 1))

    # Save the data to a CSV file.
    if path:
        save_sim(x, processed_list, path)

    return x, [y.flatten() for y in processed_list]
