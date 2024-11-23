# Standard Library Imports

# Third-Party Library Imports

# Direct Namespace Imports
from gpflow.models import GPR
from gpflow.kernels import SquaredExponential, Matern12, Cosine

# Internal Project Imports
from pyrocell.gp.gpflow.backend.types import GPModel, Ndarray


# -----------------------------------------#
# --- Gaussian Process Implementations --- #
# -----------------------------------------#


class OU(GPModel):
    """
    Ornstein-Uhlenbeck process class
    """


class OUosc(GPModel):
    """
    Ornstein-Uhlenbeck process with an oscillator (cosine) kernel
    """


class NoiseModel(GPModel):
    """
    Noise model class
    """

    def __call__(self, X: Ndarray, y: Ndarray) -> GPR:
        kernel = SquaredExponential()
        model = GPR(data=(X, y), kernel=kernel)

        if self.priors.get("lengthscale", None) is not None:
            model.kernel.lengthscales = self.priors["lengthscale"]

        return model
