# Standard Library Imports

# Third-Party Library Imports

# Direct Namespace Imports
from gpflow import Parameter, set_trainable
from gpflow.models import GPR
from gpflow.kernels import SquaredExponential, Matern12, Cosine
from gpflow.utilities import print_summary

# Internal Project Imports
from pyrocell.gp.gpflow.backend.types import GPModel, Ndarray
from pyrocell.gp.gpflow.backend import SafeMatern12


# -----------------------------------------#
# --- Gaussian Process Implementations --- #
# -----------------------------------------#


class OU(GPModel):
    """
    Ornstein-Uhlenbeck process class
    """

    def __call__(self, X: Ndarray, y: Ndarray) -> GPR:
        kernel = Matern12()
        model = GPR(data=(X, y), kernel=kernel)

        if self.priors.get("lengthscale", None) is not None:
            model.kernel.lengthscales.assign(self.priors["lengthscale"])

        if self.priors.get("variance", None) is not None:
            model.kernel.variance.assign(self.priors["variance"])

        if self.priors.get("likelihood_variance", None) is not None:
            model.likelihood.variance.assign(self.priors["likelihood_variance"])

        if self.priors.get("train_likelihood", None) is not None:
            set_trainable(model.likelihood.variance, self.priors["train_likelihood"])  # type: ignore

        # print_summary(model)

        return model


class OUosc(GPModel):
    """
    Ornstein-Uhlenbeck process with an oscillator (cosine) kernel
    """

    def __call__(self, X: Ndarray, y: Ndarray) -> GPR:
        kernel = Matern12() + Cosine()
        model = GPR(data=(X, y), kernel=kernel)

        # Model Variance
        if self.priors.get("likelihood_variance", None) is not None:
            model.likelihood.variance.assign(self.priors["likelihood_variance"])

        if self.priors.get("train_likelihood", None) is not None:
            set_trainable(model.likelihood.variance, self.priors["train_likelihood"])

        # OU priors
        if self.priors.get("lengthscale", None) is not None:
            model.kernel.kernels[0].lengthscales.assign(self.priors["lengthscale"])

        if self.priors.get("variance", None) is not None:
            model.kernel.kernels[0].variance.assign(self.priors["variance"])

        # Cosine priors
        if self.priors.get("lengthscale_cos", None) is not None:
            model.kernel.kernels[1].lengthscales.assign(self.priors["lengthscale_cos"])

        if self.priors.get("train_osc_variance", None) is not None:
            set_trainable(
                model.kernel.kernels[1].variance, self.priors["train_osc_variance"]
            )

        # print_summary(model)

        return model


class NoiseModel(GPModel):
    """
    Noise model class
    """

    def __call__(self, X: Ndarray, y: Ndarray) -> GPR:
        kernel = SquaredExponential()
        model = GPR(data=(X, y), kernel=kernel)

        if self.priors.get("lengthscale", None) is not None:
            prior = self.priors["lengthscale"]
            if isinstance(prior, Parameter):
                model.kernel.lengthscales = prior
            else:
                model.kernel.lengthscales.assign(self.priors["lengthscale"])

        return model
