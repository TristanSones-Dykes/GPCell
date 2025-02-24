# Standard Library Imports
import random
from typing import List, Mapping, Sequence, Tuple

# Third-Party Library Imports
import tensorflow_probability as tfp

# Direct Namespace Imports
from tensorflow import Module
from gpflow import Parameter
from numpy import array, ndarray, log

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


def _simulate_replicate_mod9(
    N: int,
    par: Sequence[int | float],
    mstart: int,
    pstart: int,
    out_times: List[float],
) -> Tuple[ndarray, ndarray]:
    """
    Runs one replicate of the Gillespie algorithm with delays.
    """
    # Unpack parameters.
    P0 = par[0]
    NP = par[1]
    MUM = par[2]
    MUP = par[3]
    ALPHAM = par[4]
    ALPHAP = par[5]
    tau = par[6]

    num_times = len(out_times)
    m_out = [0] * num_times
    p_out = [0] * num_times
    j_t_next = 0

    m = mstart
    p = pstart
    t = 0.0
    rlist: List[float] = []

    # Initial propensities.
    a1 = MUM * m
    a2 = MUP * p
    a3 = ALPHAP * m
    a4 = N * ALPHAM / (1 + (((p / N) / P0) ** NP))

    while t < out_times[-1]:
        a0 = a1 + a2 + a3 + a4
        r1 = random.random()
        r2 = random.random()
        dt = (1 / a0) * log(1 / r1)

        if rlist and t <= rlist[0] <= (t + dt):
            # Process delayed event.
            m_new = m + 1
            p_new = p
            t_new = rlist.pop(0)
            a1 = MUM * m_new
            a3 = ALPHAP * m_new
        else:
            if r2 * a0 <= a1:
                # m decreases.
                m_new = m - 1
                p_new = p
                a1 = MUM * m_new
                a3 = ALPHAP * m_new
            elif r2 * a0 <= a1 + a2:
                # p decreases.
                m_new = m
                p_new = p - 1
                a2 = MUP * p_new
                a4 = N * ALPHAM / (1 + (((p_new / N) / P0) ** NP))
            elif r2 * a0 <= a1 + a2 + a3:
                # p increases.
                m_new = m
                p_new = p + 1
                a2 = MUP * p_new
                a4 = N * ALPHAM / (1 + (((p_new / N) / P0) ** NP))
            else:
                # Schedule a delayed event.
                rlist.append(t + tau)
                m_new = m
                p_new = p
            t_new = t + dt

        t = t_new
        m = m_new
        p = p_new

        # Record the state for each output time passed.
        while j_t_next < num_times and t > out_times[j_t_next]:
            m_out[j_t_next] = m
            p_out[j_t_next] = p
            j_t_next += 1

    return array(m_out, dtype=int), array(p_out, dtype=int)


def _simulate_replicate_mod9_nodelay(
    N: int,
    par: Sequence[int | float],
    mstart: int,
    pstart: int,
    out_times: List[float],
) -> Tuple[ndarray, ndarray]:
    """
    Runs one replicate of the Gillespie algorithm without delays.
    """
    P0 = par[0]
    NP = par[1]
    MUM = par[2]
    MUP = par[3]
    ALPHAM = par[4]
    ALPHAP = par[5]
    tau = par[6]  # Not used in the nodelay version.  # noqa: F841

    num_times = len(out_times)
    m_out = [0] * num_times
    p_out = [0] * num_times
    j_t_next = 0

    m = mstart
    p = pstart
    t = 0.0
    rlist: List[float] = []  # remains unused.

    a1 = MUM * m
    a2 = MUP * p
    a3 = ALPHAP * m
    a4 = N * ALPHAM / (1 + (((p / N) / P0) ** NP))

    while t < out_times[-1]:
        a0 = a1 + a2 + a3 + a4
        r1 = random.random()
        r2 = random.random()
        dt = (1 / a0) * log(1 / r1)

        if rlist and t <= rlist[0] <= (t + dt):
            m_new = m + 1
            p_new = p
            t_new = rlist.pop(0)
            a1 = MUM * m_new
            a3 = ALPHAP * m_new
        else:
            if r2 * a0 <= a1:
                m_new = m - 1
                p_new = p
                a1 = MUM * m_new
                a3 = ALPHAP * m_new
            elif a1 <= r2 * a0 <= (a1 + a2):
                m_new = m
                p_new = p - 1
                a2 = MUP * p_new
                a4 = N * ALPHAM / (1 + (((p_new / N) / P0) ** NP))
            elif (a1 + a2) <= r2 * a0 <= (a1 + a2 + a3):
                m_new = m
                p_new = p + 1
                a2 = MUP * p_new
                a4 = N * ALPHAM / (1 + (((p_new / N) / P0) ** NP))
            else:
                m_new = m + 1
                p_new = p
                a1 = MUM * m_new
                a3 = ALPHAP * m_new
            t_new = t + dt

        t = t_new
        m = m_new
        p = p_new

        while j_t_next < num_times and t > out_times[j_t_next]:
            m_out[j_t_next] = m
            p_out[j_t_next] = p
            j_t_next += 1

    return array(m_out, dtype=int), array(p_out, dtype=int)
