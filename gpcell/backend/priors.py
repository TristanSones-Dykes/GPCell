# Standard Library Imports

# Third-Party Library Imports

# Direct Namespace Imports
from numpy import float64, pi
from numpy.random import uniform

from tensorflow_probability import distributions as tfd
from gpflow.utilities import to_default_float as f64

# Internal Project Imports
from ._types import Numeric, GPPrior


# --- GPR prior generators --- #
def hes_ou_prior(noise: float64) -> GPPrior:
    """Top-level OU prior generator."""
    return {
        "kernel.lengthscales": uniform(0.1, 2.0),
        "kernel.variance": uniform(0.1, 2.0),
        "likelihood.variance": noise**2,
    }


def hes_ouosc_prior(noise: float64) -> GPPrior:
    """Top-level OU+Oscillator prior generator."""
    return {
        "kernel.kernels[0].lengthscales": uniform(0.1, 2.0),
        "kernel.kernels[0].variance": uniform(0.1, 2.0),
        "kernel.kernels[1].lengthscales": uniform(0.1, 4.0),
        "likelihood.variance": noise**2,
    }


def sim_ou_prior(noise: float64) -> GPPrior:
    """Top-level OU prior generator using MATLAB initial values."""
    return {
        # Use a fixed lengthscale of 1.0 (since log(1) = 0 in MATLAB initialization)
        "kernel.lengthscales": 1.0,
        # Set kernel variance (amplitude) to 0.5 (from log(0.5))
        "kernel.variance": 0.5,
        "likelihood.variance": noise**2,
    }


def sim_ouosc_prior(noise: float64) -> GPPrior:
    """Top-level OU+Oscillator prior generator using MATLAB initial values."""
    return {
        # For the first kernel (OU component), set variance and lengthscale
        "kernel.kernels[0].variance": 0.1,
        "kernel.kernels[0].lengthscales": 1.0,
        # For the oscillatory component, use a lengthscale corresponding to 2*pi/6
        "kernel.kernels[1].lengthscales": 2 * pi / 6,
        "likelihood.variance": noise**2,
    }


# --- MCMC prior generators --- #


def generate_ou_priors(noise: Numeric) -> GPPrior:
    return {
        "kernel.lengthscales": tfd.Uniform(low=f64(0.1), high=f64(2.0)),
        "kernel.variance": tfd.Uniform(low=f64(0.1), high=f64(2.0)),
        "likelihood.variance": noise**2,
    }


def generate_ouosc_priors(noise: Numeric) -> GPPrior:
    return {
        "kernel.kernels[0].lengthscales.prior": tfd.Uniform(
            low=f64(0.1), high=f64(2.0)
        ),
        "kernel.kernels[0].variance.prior": tfd.Uniform(low=f64(0.1), high=f64(2.0)),
        "kernel.kernels[1].lengthscales.prior": tfd.Uniform(
            low=f64(0.1), high=f64(4.0)
        ),
        "likelihood.variance": noise**2,
    }
