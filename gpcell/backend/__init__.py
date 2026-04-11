# Standard Library Imports
from typing import List, Sequence

# Third-Party Library Imports
# Direct Namespace Imports
from numpy import ndarray

# Internal Project Imports
from ._types import (
    FixedPriorGen,
    GPKernel,
    GPOperator,
    GPPrior,
    GPPriorFactory,
    GPPriorTrainingFlag,
    Ndarray,
    Numeric,
)
from ._utils import _simulate_replicate_mod9, _simulate_replicate_mod9_nodelay
from .gaussian_process import GaussianProcess
from .gpr_constructor import GPRConstructor
from .kernels import TimeVaryingCosine
from .priors import (
    hes_ou_prior,
    hes_ouosc_prior,
    ou_hyperparameters,
    ou_trainables,
    ouosc_hyperparameters,
    ouosc_trainables,
    sim_ou_prior,
    sim_ouosc_prior,
)

__all__ = [
    "Ndarray",
    "Numeric",
    "GPKernel",
    "GPPriorFactory",
    "FixedPriorGen",
    "GPPriorTrainingFlag",
    "GPOperator",
    "GPPrior",
    "GPRConstructor",
    "GaussianProcess",
    "TimeVaryingCosine",
    "_simulate_replicate_mod9",
    "_simulate_replicate_mod9_nodelay",
    "hes_ou_prior",
    "hes_ouosc_prior",
    "ou_trainables",
    "ouosc_trainables",
    "sim_ou_prior",
    "sim_ouosc_prior",
    "ou_hyperparameters",
    "ouosc_hyperparameters",
]


def _joblib_fit_memmap_worker(
    i: int,
    X: Sequence[Ndarray],
    Y: ndarray,
    Y_var: bool,
    constructor: GPRConstructor,
    replicates: int,
) -> List[GaussianProcess]:
    """
    Worker function for fitting Gaussian Processes using Joblib with memory-mapped data.

    Parameters
    ----------
    i : int
        Index of the trace to fit.
    X : Sequence[Ndarray]
        List of input domains.
    Y : ndarray
        Memory-mapped array of input traces.
    Y_var : bool
        Whether to calculate variance of missing data.
    constructor : GPRConstructor
        Constructor for the GP model.
    replicates : int
        Number of replicates to fit.

    Returns
    -------
    List[GaussianProcess]
        List of fitted GP models.
    """
    # extract the data
    x = X[i]
    y = Y[i, :]

    # fit the models
    models = []
    for _ in range(replicates):
        gp_model = GaussianProcess(constructor)
        gp_model.fit(x, y, Y_var)
        models.append(gp_model)
    return models


def _joblib_nonhomog_fit_memmap_worker(
    i: int,
    X: Sequence[Ndarray],
    Y: ndarray,
    Y_var: bool,
    constructor: GPRConstructor,
    replicates: int,
    true_length: int | None = None,
) -> List[GaussianProcess]:
    """
    Worker function for fitting Gaussian Processes but on non-homogeneous data using Joblib with memory-mapped data.

    Parameters
    ----------
    i : int
        Index of the trace to fit.
    X : Sequence[Ndarray]
        List of input domains.
    Y : ndarray
        Memory-mapped array of input traces.
    Y_var : bool
        Whether to calculate variance of missing data.
    constructor : GPRConstructor
        Constructor for the GP model.
    replicates : int
        Number of replicates to fit.
    true_length : int
        True length of the data.

    Returns
    -------
    List[GaussianProcess]
        List of fitted GP models.
    """
    # extract the data, using passed true length
    x = X[i]
    y = Y[i, :true_length]

    # fit the models
    models = []
    for _ in range(replicates):
        gp_model = GaussianProcess(constructor)
        gp_model.fit(x, y, Y_var)
        models.append(gp_model)
    return models
