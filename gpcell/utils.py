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
from math import log
import random
import operator

# Third-Party Library Imports
import numpy as np
import pandas as pd
import tensorflow_probability as tfp

# Direct Namespace Imports
from numpy import float64, nonzero, std, mean, max, zeros
from gpflow import Parameter
from gpflow.kernels import RBF
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
)
from gpcell.backend._types import Numeric


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


def gillespie_timing_mod9(
    N: int,
    par: Sequence[Numeric],
    totalreps: int,
    mstart: int,
    pstart: int,
    Output_Times: List[float],
):
    """
    Runs Gillespie algorithm with delay processes.

    Parameters
    ----------
    N: int
            Number of something (as in MATLAB code)
    par: List[float]
            List or tuple of parameters [P0, NP, MUM, MUP, ALPHAM, ALPHAP, tau]
    totalreps: int
            Number of simulation replicates
    mstart: int
            Initial m value
    pstart: int
            Initial p value
    Output_Times: List[float]
            List or array of output times at which to record (must be in increasing order)

    Returns
    -------
    Tuple[Ndarray, Ndarray]
            2D NumPy array (totalreps x len(Output_Times)) of m values, 2D NumPy array (totalreps x len(Output_Times)) of p values
    """
    # Unpack parameters (note: MATLAB uses 1-indexing; Python uses 0-indexing)
    P0 = par[0]
    NP = par[1]
    MUM = par[2]
    MUP = par[3]
    ALPHAM = par[4]
    ALPHAP = par[5]
    tau = par[6]

    # Initialize output arrays (using int type to mirror MATLAB's zeros)
    mout = zeros((totalreps, len(Output_Times)), dtype=int)
    pout = zeros((totalreps, len(Output_Times)), dtype=int)

    for rep in range(totalreps):
        j_t_next = 0  # Python index for Output_Times

        # Set initial values
        m = mstart
        p = pstart
        t = 0.0
        rlist = []  # list for delayed events

        # Compute initial propensities
        a1 = MUM * m
        a2 = MUP * p
        a3 = ALPHAP * m
        a4 = N * ALPHAM / (1 + (((p / N) / P0) ** NP))

        # Continue simulation until t reaches the last output time
        while t < Output_Times[-1]:
            a0 = a1 + a2 + a3 + a4
            r1 = random.random()
            r2 = random.random()
            dt = (1 / a0) * log(1 / r1)

            # Check if a delayed event is scheduled before the next dt
            if rlist and t <= rlist[0] <= (t + dt):
                # Process delayed event:
                mn = m + 1
                pn = p
                tn = rlist.pop(0)  # remove the first scheduled event
                # Update propensities that depend on m
                a1 = MUM * mn
                a3 = ALPHAP * mn
            else:
                # Determine which reaction occurs
                if r2 * a0 <= a1:
                    # Reaction: m decreases by one
                    mn = m - 1
                    pn = p
                    a1 = MUM * mn
                    a3 = ALPHAP * mn
                elif r2 * a0 <= a1 + a2:
                    # Reaction: p decreases by one
                    mn = m
                    pn = p - 1
                    a2 = MUP * pn
                    a4 = N * ALPHAM / (1 + (((pn / N) / P0) ** NP))
                elif r2 * a0 <= a1 + a2 + a3:
                    # Reaction: p increases by one
                    mn = m
                    pn = p + 1
                    a2 = MUP * pn
                    a4 = N * ALPHAM / (1 + (((pn / N) / P0) ** NP))
                else:
                    # Schedule a delayed event at t + tau
                    rlist.append(t + tau)
                    mn = m
                    pn = p
                tn = t + dt  # update time for immediate (non-delayed) events

            # Update time and state variables
            t = tn
            m = mn
            p = pn

            # Record the current state for any output times passed
            while j_t_next < len(Output_Times) and t > Output_Times[j_t_next]:
                mout[rep, j_t_next] = m
                pout[rep, j_t_next] = p
                j_t_next += 1

    return mout, pout


def gillespie_timing_mod9_nodelay(
    N: int,
    par: Sequence[Numeric],
    totalreps: int,
    mstart: int,
    pstart: int,
    Output_Times: List[float],
):
    """
    Runs Gillespie algorithm without delay processes.

    Parameters
    ----------
    N: int
            Number of something (as in MATLAB code)
    par: List[float]
            List or tuple of parameters [P0, NP, MUM, MUP, ALPHAM, ALPHAP, tau]
    totalreps: int
            Number of simulation replicates
    mstart: int
            Initial m value
    pstart: int
            Initial p value
    Output_Times: List[float]
            List or array of output times at which to record (must be in increasing order)

    Returns
    -------
    Tuple[Ndarray, Ndarray]
            2D NumPy array (totalreps x len(Output_Times)) of m values, 2D NumPy array (totalreps x len(Output_Times)) of p values
    """

    # Unpack parameters (adjusting for Python's 0-indexing)
    P0 = par[0]
    NP = par[1]
    MUM = par[2]
    MUP = par[3]
    ALPHAM = par[4]
    ALPHAP = par[5]
    tau = par[6]  # Although tau is provided, it is not used in the nodelay version

    # Initialize output arrays.
    mout = zeros((totalreps, len(Output_Times)), dtype=int)
    pout = zeros((totalreps, len(Output_Times)), dtype=int)

    for rep in range(totalreps):
        j_t_next = 0  # index for the next output time

        # Set initial values
        m = mstart
        p = pstart
        t = 0.0
        rlist = []  # In the nodelay version, this list remains unused.

        # Compute initial propensities.
        a1 = MUM * m
        a2 = MUP * p
        a3 = ALPHAP * m
        a4 = N * ALPHAM / (1 + (((p / N) / P0) ** NP))

        # Main simulation loop: run until time exceeds the last output time.
        while t < Output_Times[-1]:
            a0 = a1 + a2 + a3 + a4
            r1 = random.random()
            r2 = random.random()
            dt = (1 / a0) * log(1 / r1)

            # --- MATLAB: if numel(rlist)>0 && t<=rlist(1) && rlist(1)<=(t+dt) ---
            if rlist and t <= rlist[0] <= (t + dt):
                # In MATLAB:
                #    mn = m+1;
                #    pn = p;
                #    tn = rlist(1);
                #    rlist(1) = [];
                #    a1 = MUM * mn;
                #    a3 = ALPHAP * mn;
                m_new = m + 1
                p_new = p
                t_new = rlist.pop(
                    0
                )  # remove the first element (MATLAB's rlist(1) = [])
                a1 = MUM * m_new
                a3 = ALPHAP * m_new
            else:
                # --- MATLAB: if r2*a0<=a1 ---
                if r2 * a0 <= a1:
                    # Reaction: decrease m.
                    # MATLAB: mn = m-1; pn = p; a1 = MUM * (mn); a3 = ALPHAP * (mn);
                    m_new = m - 1
                    p_new = p
                    a1 = MUM * m_new
                    a3 = ALPHAP * m_new
                # --- MATLAB: elseif a1<=r2*a0 && r2*a0<=(a1+a2) ---
                elif a1 <= r2 * a0 <= (a1 + a2):
                    # Reaction: decrease p.
                    # MATLAB: mn = m; pn = p-1; a2 = MUP * pn; a4 = N * ALPHAM / (1 + ((pn/N)/P0)^NP);
                    m_new = m
                    p_new = p - 1
                    a2 = MUP * p_new
                    a4 = N * ALPHAM / (1 + (((p_new / N) / P0) ** NP))
                # --- MATLAB: elseif (a1+a2)<=r2*a0 && r2*a0<=(a1+a2+a3) ---
                elif (a1 + a2) <= r2 * a0 <= (a1 + a2 + a3):
                    # Reaction: increase p.
                    # MATLAB: mn = m; pn = p+1; a2 = MUP * pn; a4 = N * ALPHAM / (1 + ((pn/N)/P0)^NP);
                    m_new = m
                    p_new = p + 1
                    a2 = MUP * p_new
                    a4 = N * ALPHAM / (1 + (((p_new / N) / P0) ** NP))
                # --- MATLAB: else ---
                else:
                    # In the nodelay version the reaction increases m (instead of scheduling a delay)
                    # MATLAB: mn = m+1; pn = p; a1 = MUM * (mn); a3 = ALPHAP * (mn);
                    m_new = m + 1
                    p_new = p
                    a1 = MUM * m_new
                    a3 = ALPHAP * m_new
                # Update time for these immediate reactions.
                t_new = t + dt

            # Update state and time.
            t = t_new
            m = m_new
            p = p_new

            # Record state values if t has passed the scheduled output times.
            while j_t_next < len(Output_Times) and t > Output_Times[j_t_next]:
                mout[rep, j_t_next] = m
                pout[rep, j_t_next] = p
                j_t_next += 1

    return mout, pout


def get_time_series(par1, par2, Tfinal, Noise, totalreps):
    """
    Recreates the MATLAB GetTimeSeries.m functionality in Python.

    Parameters
    ----------
    par1 : array-like
        Parameter vector for simulation #1.
    par2 : array-like
        Parameter vector for simulation #2.
    Tfinal : int or float
        Final simulation time (in MATLAB, Tfinal=1500).
    Noise : float
        Noise level (e.g. sqrt(0.1)).
    totalreps : int
        Number of simulation replicates (CellNum in MATLAB).

    Returns
    -------
    x : ndarray
        Time vector (converted from MATLAB Output_Times).
    dataNORMED : ndarray
        A 2D array in which each column is a time series (with added noise).
    """
    # MATLAB: Nvec = [20]; and Output_Times = 5000:30:(5000+Tfinal);
    Nvec = [20]
    Output_Times = np.arange(5000, 5000 + Tfinal + 1, 30)  # step size 30

    # Run the Gillespie simulation for par1 using the "nodelay" version.
    N = Nvec[0]
    mstart = 1 * N
    pstart = 30 * N
    # Note: our Python simulation functions expect Output_Times as a list (or array)
    mout1, pout1 = gillespie_timing_mod9_nodelay(
        N, par1, totalreps, mstart, pstart, Output_Times.tolist()
    )

    # Run the Gillespie simulation for par2 using the delay version.
    mout2, pout2 = gillespie_timing_mod9(
        N, par2, totalreps, mstart, pstart, Output_Times.tolist()
    )

    # In MATLAB, x = Output_Times' (a column vector).
    x = Output_Times.astype(float)
    # Transpose the p-values so that rows correspond to times.
    data1 = np.transpose(pout1)
    data2 = np.transpose(pout2)
    # Concatenate the two simulation outputs horizontally.
    dataTOT = np.hstack((data1, data2))

    # Convert time: MATLAB does x = (x-5000)/60;
    x = (x - 5000) / 60.0

    # Add noise to each time series.
    samp = len(x)
    dataNORMED = np.zeros_like(dataTOT, dtype=float)
    for i in range(dataTOT.shape[1]):
        y1 = dataTOT[:, i].copy()
        # MATLAB: y1 = y1 - mean(y1);
        y1 = y1 - np.mean(y1)
        # Normalize y1.
        std_y1 = np.std(y1)
        if std_y1 == 0:
            std_y1 = 1  # avoid division by zero
        y1 = y1 / std_y1
        # Create a noise covariance: diag((Noise^2)*ones(1,samp))
        SIGMA = np.diag((Noise**2) * np.ones(samp))
        # Draw a noise vector from a multivariate normal.
        measerror = np.random.multivariate_normal(np.zeros(samp), SIGMA)
        # Add noise.
        y2 = y1 + measerror
        dataNORMED[:, i] = y2

    # convert to lists of numpy arrays
    n_series = dataNORMED.shape[1]
    X = [x] * n_series
    Y = [dataNORMED[:, i].reshape(-1, 1) for i in range(n_series)]

    return X, Y


def save_sim(X, Y, filename):
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
    # All elements of X are identical time arrays, so just use the first one.
    time = X[0]  # shape (num_time_points,)

    # Number of simulated cells (columns)
    n_series = len(Y)

    # Each Y[i] is shape (num_time_points, 1).
    # Stack them horizontally to get a (num_time_points, n_series) 2D array.
    data = np.hstack(Y)  # shape becomes (num_time_points, n_series)

    # Build a DataFrame with each column labeled "Cell 1", "Cell 2", etc.
    col_names = [f"Cell {i + 1}" for i in range(n_series)]
    df = pd.DataFrame(data, columns=col_names)

    # Insert the time as its own column at the front
    df.insert(0, "Time", time)

    # Write to CSV (no index column, since we include time explicitly)
    df.to_csv(filename, index=False)
