# Standard Library Imports
from typing import Mapping, Tuple

# Third-Party Library Imports
import tensorflow_probability as tfp

# Direct Namespace Imports
from tensorflow import Module
from gpflow import Parameter
from numpy import empty, float64, int32, ndarray, log, random
from numba import njit

# Internal Project Imports
from ._types import GPPrior


# ----------------------------------------------#
# --- Gaussian Process Parameter Assignment --- #
# ----------------------------------------------#


def multiple_assign(module: Module, parameters: Mapping[str, GPPrior]) -> None:
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


def _set_parameter_by_key(module: Module, key: str, value: GPPrior):
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
        case tfp.distributions.Uniform():
            setattr(target, parts[-1], value)
        case _:
            getattr(target, parts[-1]).assign(value)


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
    Runs one replicate of the Gillespie algorithm with delays.
    Uses a fixed-size array for delayed events.
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

    # Use a fixed-size array for delayed events.
    max_delays = 10
    rlist = empty(max_delays, dtype=float64)
    rlist_count = 0

    # Initial propensities.
    a1_val = MUM * m
    a2_val = MUP * p
    a3_val = ALPHAP * m
    a4_val = N * ALPHAM / (1.0 + ((p / N) / P0) ** NP)

    while t < out_times[num_times - 1]:
        a0 = a1_val + a2_val + a3_val + a4_val
        r1 = random.random()
        r2 = random.random()
        dt = (1.0 / a0) * log(1.0 / r1)

        # Check for a delayed event.
        if rlist_count > 0 and t <= rlist[0] and rlist[0] <= (t + dt):
            m_new = m + 1
            p_new = p
            t_new = rlist[0]
            # Shift the delay array to remove the first element.
            for k in range(1, rlist_count):
                rlist[k - 1] = rlist[k]
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
                if rlist_count < max_delays:
                    rlist[rlist_count] = t + tau
                    rlist_count += 1
                else:
                    # If rlist is full, overwrite the last element.
                    rlist[max_delays - 1] = t + tau
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
    Runs one replicate of the Gillespie algorithm without delays.
    Uses a fixed-size array for delayed events (unused in typical execution).
    """
    P0 = par[0]
    NP = par[1]
    MUM = par[2]
    MUP = par[3]
    ALPHAM = par[4]
    ALPHAP = par[5]
    # tau = par[6]  # Not used in the nodelay version.

    num_times = out_times.shape[0]
    m_out = empty(num_times, dtype=int32)
    p_out = empty(num_times, dtype=int32)
    j_t_next = 0

    m = mstart
    p = pstart
    t = 0.0

    # Fixed-size delay array (will remain unused but required for code symmetry).
    max_delays = 10
    rlist = empty(max_delays, dtype=float64)
    rlist_count = 0

    a1_val = MUM * m
    a2_val = MUP * p
    a3_val = ALPHAP * m
    a4_val = N * ALPHAM / (1.0 + ((p / N) / P0) ** NP)

    while t < out_times[num_times - 1]:
        a0 = a1_val + a2_val + a3_val + a4_val
        r1 = random.random()
        r2 = random.random()
        dt = (1.0 / a0) * log(1.0 / r1)

        if rlist_count > 0 and t <= rlist[0] and rlist[0] <= (t + dt):
            m_new = m + 1
            p_new = p
            t_new = rlist[0]
            for k in range(1, rlist_count):
                rlist[k - 1] = rlist[k]
            rlist_count -= 1
            a1_val = MUM * m_new
            a3_val = ALPHAP * m_new
        else:
            if r2 * a0 <= a1_val:
                m_new = m - 1
                p_new = p
                a1_val = MUM * m_new
                a3_val = ALPHAP * m_new
            elif r2 * a0 <= a1_val + a2_val:
                m_new = m
                p_new = p - 1
                a2_val = MUP * p_new
                a4_val = N * ALPHAM / (1.0 + ((p_new / N) / P0) ** NP)
            elif r2 * a0 <= a1_val + a2_val + a3_val:
                m_new = m
                p_new = p + 1
                a2_val = MUP * p_new
                a4_val = N * ALPHAM / (1.0 + ((p_new / N) / P0) ** NP)
            else:
                m_new = m + 1
                p_new = p
                a1_val = MUM * m_new
                a3_val = ALPHAP * m_new
            t_new = t + dt

        t = t_new
        m = m_new
        p = p_new

        while j_t_next < num_times and t > out_times[j_t_next]:
            m_out[j_t_next] = m
            p_out[j_t_next] = p
            j_t_next += 1

    return m_out, p_out
