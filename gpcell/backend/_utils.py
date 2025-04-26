# Standard Library Imports
from typing import Tuple, Union

# Third-Party Library Imports
import numpy as np

# Direct Namespace Imports
from tensorflow_probability import distributions as tfd
from tensorflow import Module
from gpflow import Parameter
from numpy import empty, float64, inf, int32, ndarray, log
from numba import njit

# Internal Project Imports
from ._types import GPPrior, Numeric


# ----------------------------------------------#
# --- Gaussian Process Parameter Assignment --- #
# ----------------------------------------------#


def multiple_assign(module: Module, parameters: GPPrior) -> None:
    """
    Assigns parameters in a dictionary to the Module (tf.Module or gpflow.Module)

    Parameters
    ----------
    module : Module
        Model to assign parameters to
    parameters : Mapping[str, GPPrior]
        Dictionary of parameters to assign
    """
    for key, value in parameters.items():
        _set_parameter_by_key(module, key, value)


def _set_parameter_by_key(
    module: Module, key: str, value: Union[Parameter, Numeric, str]
):
    """
    Sets a parameter in a module by key

    Parameters
    ----------
    module : Module
        Module to set the parameter in
    key : str
        Key to the parameter
    value : GPPrior
        Value to set the parameter to
    """
    parts = key.split(".")
    target = module

    for i in range(len(parts) - 1):
        part = parts[i]
        if "[" in part and "]" in part:
            # Handle indexed attributes like "kernels[0]"
            attr_name, index = part.split("[")
            index = int(index.rstrip("]"))
            target = getattr(target, attr_name)[index]
        else:
            # Handle normal attributes
            target = getattr(target, part)

    # Finally, set the parameter
    match value:
        case Parameter():
            setattr(target, parts[-1], value)
        case tfd.Uniform():
            setattr(target, parts[-1], value)
        case tfd.LogNormal():
            setattr(target, parts[-1], value)
        case str():
            # MCMC constrained/unconstrained
            if parts[-1] == "prior_on":
                # print(f"Setting {key} to {value}")
                target = value
            else:
                raise ValueError(f"Invalid value type for {key}: {type(value)}")
        case _:
            # try block
            try:
                getattr(target, parts[-1]).assign(value)
            except AttributeError:
                # If the attribute is not a Parameter, assign it directly
                setattr(target, parts[-1], value)


# -----------------------------#
# --- Gillespie Simulation --- #
# -----------------------------#


@njit(cache=True)
def _simulate_replicate_mod9(
    N: int,
    par: ndarray,
    mstart: int,
    pstart: int,
    out_times: ndarray,  # best passed as a numpy array of float64
) -> Tuple[ndarray, ndarray]:
    """
    Runs the Gillespie algorithm with delays.

    Parameters
    ----------
    N : int
        The system size.
    par : np.ndarray
        Parameter vector:
            par[0] = P0
            par[1] = NP
            par[2] = MUM
            par[3] = MUP
            par[4] = ALPHAM
            par[5] = ALPHAP
            par[6] = tau  (unused in this nodelay version)
    mstart : int
        Initial value for m.
    pstart : int
        Initial value for p.
    out_times : np.ndarray
        Array of output times (must be sorted in increasing order).

    Returns
    -------
    mout : np.ndarray
        A 1D array (length = number of output times) of m values.
    pout : np.ndarray
        A 1D array (length = number of output times) of p values.
    """
    # Unpack parameters.
    P0 = par[0]
    NP = par[1]
    MUM = par[2]
    MUP = par[3]
    ALPHAM = par[4]
    ALPHAP = par[5]
    tau = par[6]

    num_times = out_times.shape[0]
    m_out = empty(num_times, dtype=int32)
    p_out = empty(num_times, dtype=int32)
    j_t_next = 0

    m = mstart
    p = pstart
    t = 0.0

    # Preallocate a large array for delayed events (circular buffer).
    max_delays = 200
    rlist = empty(max_delays, dtype=float64)
    rlist_start = 0  # Index of the first element in the queue.
    rlist_count = 0  # Number of elements currently in the buffer.

    # Initial propensities.
    a1_val = MUM * m
    a2_val = MUP * p
    a3_val = ALPHAP * m
    a4_val = N * ALPHAM / (1.0 + ((p / N) / P0) ** NP)

    while t < out_times[num_times - 1]:
        a0 = a1_val + a2_val + a3_val + a4_val
        r1 = np.random.random()
        r2 = np.random.random()
        dt = (1.0 / a0) * log(1.0 / r1)

        # Check for a delayed event.
        if rlist_count > 0:
            first_event = rlist[rlist_start]
        else:
            first_event = inf  # No event scheduled.

        # Check for a delayed event.
        if rlist_count > 0 and t <= first_event <= (t + dt):
            m_new = m + 1
            p_new = p
            t_new = first_event

            # Remove the first event by advancing the head.
            rlist_start = (rlist_start + 1) % max_delays
            rlist_count -= 1

            a1_val = MUM * m_new
            a3_val = ALPHAP * m_new
        else:
            if r2 * a0 <= a1_val:
                # m decreases.
                m_new = m - 1
                p_new = p
                a1_val = MUM * m_new
                a3_val = ALPHAP * m_new
            elif r2 * a0 <= a1_val + a2_val:
                # p decreases.
                m_new = m
                p_new = p - 1
                a2_val = MUP * p_new
                a4_val = N * ALPHAM / (1.0 + ((p_new / N) / P0) ** NP)
            elif r2 * a0 <= a1_val + a2_val + a3_val:
                # p increases.
                m_new = m
                p_new = p + 1
                a2_val = MUP * p_new
                a4_val = N * ALPHAM / (1.0 + ((p_new / N) / P0) ** NP)
            else:
                # Schedule a delayed event.
                # Compute the index at which to insert the new event.
                index = (rlist_start + rlist_count) % max_delays
                rlist[index] = t + tau
                rlist_count += 1

                m_new = m
                p_new = p
            t_new = t + dt

        t = t_new
        m = m_new
        p = p_new

        # Record the state if we've passed an output time.
        while j_t_next < num_times and t > out_times[j_t_next]:
            m_out[j_t_next] = m
            p_out[j_t_next] = p
            j_t_next += 1

    return m_out, p_out


@njit(cache=True)
def _simulate_replicate_mod9_nodelay(
    N: int,
    par: ndarray,
    mstart: int,
    pstart: int,
    out_times: ndarray,
) -> Tuple[ndarray, ndarray]:
    """
    Runs the Gillespie algorithm without delays.

    Parameters
    ----------
    N : int
        The system size.
    par : np.ndarray
        Parameter vector:
            par[0] = P0
            par[1] = NP
            par[2] = MUM
            par[3] = MUP
            par[4] = ALPHAM
            par[5] = ALPHAP
            par[6] = tau  (unused in this nodelay version)
    mstart : int
        Initial value for m.
    pstart : int
        Initial value for p.
    out_times : np.ndarray
        Array of output times (must be sorted in increasing order).

    Returns
    -------
    mout : np.ndarray
        A 1D array (length = number of output times) of m values.
    pout : np.ndarray
        A 1D array (length = number of output times) of p values.
    """
    num_times = out_times.shape[0]
    m_out = empty(num_times, dtype=int32)
    p_out = empty(num_times, dtype=int32)

    # Unpack parameters.
    P0 = par[0]
    NP = par[1]
    MUM = par[2]
    MUP = par[3]
    ALPHAM = par[4]
    ALPHAP = par[5]
    # tau = par[6] is not used in this version.

    j_t_next = 0  # Output time index.
    t = 0.0
    m = mstart
    p = pstart

    # Fixed-size delay array (will remain unused but required for code symmetry).
    max_delays = 10
    rlist = empty(max_delays, dtype=float64)
    rlist_count = 0

    a1 = MUM * m
    a2 = MUP * p
    a3 = ALPHAP * m
    a4 = N * ALPHAM / (1.0 + ((p / N) / P0) ** NP)

    # Run simulation until the final output time.
    while t < out_times[num_times - 1]:
        a0 = a1 + a2 + a3 + a4
        r1 = np.random.random()
        r2 = np.random.random()
        dt = (1.0 / a0) * log(1.0 / r1)

        if rlist_count > 0 and t <= rlist[0] and rlist[0] <= (t + dt):
            m_new = m + 1
            p_new = p
            t_new = rlist[0]
            for k in range(1, rlist_count):
                rlist[k - 1] = rlist[k]
            rlist_count -= 1
            a1 = MUM * m_new
            a3 = ALPHAP * m_new
        else:
            if r2 * a0 <= a1:
                m_new = m - 1
                p_new = p
                a1 = MUM * m_new
                a3 = ALPHAP * m_new
            elif r2 * a0 <= a1 + a2:
                m_new = m
                p_new = p - 1
                a2 = MUP * p_new
                a4 = N * ALPHAM / (1.0 + ((p_new / N) / P0) ** NP)
            elif r2 * a0 <= a1 + a2 + a3:
                m_new = m
                p_new = p + 1
                a2 = MUP * p_new
                a4 = N * ALPHAM / (1.0 + ((p_new / N) / P0) ** NP)
            else:
                m_new = m + 1
                p_new = p
                a1 = MUM * m_new
                a3 = ALPHAP * m_new
            t_new = t + dt

        t = t_new
        m = m_new
        p = p_new

        # Record the state for each output time passed.
        while j_t_next < num_times and t > out_times[j_t_next]:
            m_out[j_t_next] = m
            p_out[j_t_next] = p
            j_t_next += 1

    return m_out, p_out
