# Standard library imports
from typing import Callable, Dict, Optional, Tuple

# External library imports
import matplotlib.pyplot as plt

# External type imports
from torch import (
    Tensor,
    clone,
    tensor,
    no_grad,
    zeros,
    mean,
    sqrt,
    eye,
    matmul,
    logdet,
    log,
    pi,
)
from torch.linalg import solve
from torch.optim import LBFGS, Optimizer
from torch.linalg import LinAlgError

# --- Pyro-related imports --- #

# Gaussian processes
from pyro.contrib.gp.models import GPRegression
from pyro.contrib.gp.kernels import Isotropy, Cosine, RBF, Product, Exponential

# Primitives and utilities
from pyro.nn import PyroParam, PyroSample
from pyro.distributions.constraints import greater_than
from pyro.infer import Trace_ELBO
from pyro import clear_param_store

# Internal imports
from pyrocell.gp.kernels import Matern12


# -------------------------------- #
# --- Gaussian Process classes --- #
# -------------------------------- #


class GaussianProcess:
    """
    Gaussian Process class for fitting and evaluating parameters
    """

    def __init__(
        self, kernel: Isotropy, optimizer: Optimizer, variance_loc: str = ["variance"]
    ):
        self.kernel = kernel
        """Kernel for the Gaussian Process"""
        self.optimizer = optimizer
        """Optimizer for the Gaussian Process"""
        self.variance_loc = variance_loc
        """Location of the variance parameter in the kernel"""

    def __call__(
        self, X: Tensor, full_cov: bool = False, noiseless: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Evaluate the Gaussian Process on the input domain

        :param Tensor X: Input domain
        :param bool full_cov: Return full covariance matrix
        :param bool noiseless: Return noiseless predictions

        :return Tuple[Tensor, Tensor]: Mean and standard deviation
        """
        with no_grad():
            mean, cov = self.fit_gp(X, full_cov=full_cov, noiseless=noiseless)

        if not full_cov:
            var = cov.diag()
            return mean, var

        return mean, cov

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        loss_fn: Callable[..., Tensor],
        lr: float = 0.01,
        noise: Optional[Tensor] = None,
        jitter: float = 1.0e-5,
        num_steps: int = 1000,
        priors: Dict[str, PyroParam | PyroSample] = {},
        verbose: bool = False,
    ) -> bool:
        """
        Fit the Gaussian Process model, saves the model and training values for later use if needed.

        :param Tensor X: Input domain
        :param Tensor y: Target values
        :param Callable[..., Tensor] loss_fn: Loss function
        :param float lr: Learning rate
        :param Tensor noise: Variance of Gaussian noise of the model
        :param float jitter: Positive term to help Cholesky decomposition
        :param int num_steps: Number of steps
        :param dict[str, PyroParam | PyroSample] priors: Priors for the kernel parameters
        :param bool verbose: Print training information

        :return bool: Success status
        """
        clear_param_store()

        # check if kernel is class or object
        if isinstance(self.kernel, type):
            kernel = self.kernel(input_dim=1)
        else:
            kernel = self.kernel

        # set priors
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        # gaussian regression
        sgpr = GPRegression(X, y, kernel, noise=noise, jitter=jitter)
        optimizer = self.optimizer(sgpr.parameters(), lr=lr)

        # check if closure is needed
        if optimizer.__class__.__name__ == "LBFGS":

            def closure():
                optimizer.zero_grad()
                loss = loss_fn(sgpr.model, sgpr.guide)
                loss.backward()
                return loss
        else:
            closure = None

        try:
            for i in range(num_steps):
                optimizer.zero_grad()
                loss = loss_fn(sgpr.model, sgpr.guide)
                loss.backward()
                optimizer.step(closure)

                # if verbose and (i % 100 == 0 or i == num_steps-1):
                #    print(f"lengthscale: {sgpr.kernel.lengthscale.item()}")
        except LinAlgError as e:
            print(f"Lapack error code: {e}")
            return False

        # display final trained parameters
        if verbose:
            for param in priors.keys():
                print(f"{param}: {getattr(sgpr.kernel, param).item()}")

        # calculate log posterior density
        with no_grad():
            self.loss = loss_fn(sgpr.model, sgpr.guide)

        # mean and variance
        with no_grad():
            self.mean, self.cov = sgpr(X, full_cov=True, noiseless=False)
            self.var = self.cov.diag()

        self.X_true = X
        self.y_true = y
        self.fit_gp = sgpr

        return True

    def log_likelihood(self, y: Optional[Tensor] = None) -> Tensor:
        """
        Calculates the log-marginal likelihood for the Gaussian process.
        If no target values are input, calculates log likelihood for data is was it on.

        :param Tensor y: Observed target values (optional)
        :return Tensor: Log-likelihood
        """
        # no y input -> use fitted mean and cov
        if y is None:
            y, mean, K = clone(self.y_true), clone(self.mean), clone(self.cov)
        else:
            # evaluate new mean and cov
            mean, K = self.fit_gp(self.X_true, y)

        with no_grad():
            # add noise variance to kernel diagonal
            noise = getattr(self.fit_gp.kernel, self.variance_loc[0])
            for i in range(1, len(self.variance_loc)):
                noise = getattr(noise, self.variance_loc[i])

            K_with_noise = K + noise.item() * eye(K.size(0))

            # residuals
            residual = y - mean

            # Compute terms for log-marginal likelihood
            term1 = -0.5 * matmul(residual.T, solve(K_with_noise, residual))
            term2 = -0.5 * logdet(K_with_noise)
            term3 = -0.5 * len(y) * log(tensor(2 * pi))

        return term1 + term2 + term3

    def test_plot(
        self,
        X: Optional[Tensor] = None,
        y_true: Optional[Tensor] = None,
        plot_sd: bool = False,
    ):
        """
        Create a test plot of the fitted model on the training data

        :param Tensor X: Input domain
        :param Tensor y_true: True target values
        :param bool plot_sd: Plot standard deviation

        :return: None
        """
        # check if fit_gp exists
        if not hasattr(self, "fit_gp"):
            raise AttributeError("Please fit the model first")

        # default X and y_true values
        if X is None:
            X = self.X_true
            mean, std = self.mean, self.var.sqrt()

        if y_true is None:
            y_true = self.y_true

        # plot
        plt.plot(X, mean, zorder=1, c="k", label="Fit GP")
        if plot_sd:
            plt.plot(X, mean + 2 * std, zorder=0, c="r")
            plt.plot(X, mean - 2 * std, zorder=0, c="r")

        if y_true is not None:
            plt.plot(X, y_true, zorder=0, c="b", label="True data")


class OU(GaussianProcess):
    """
    Ornstein-Uhlenbeck process class
    """

    def __init__(self, priors: Dict[str, PyroParam | PyroSample]):
        # create kernel
        kernel = Exponential(input_dim=1)
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        super().__init__(kernel, LBFGS)


class OUosc(GaussianProcess):
    """
    Ornstein-Uhlenbeck process with an oscillator (cosine) kernel
    """

    def __init__(
        self,
        ou_priors: Dict[str, PyroParam | PyroSample],
        osc_priors: Dict[str, PyroParam | PyroSample],
    ):
        # create kernels
        ou_kernel = Exponential(input_dim=1)
        osc_kernel = Cosine(input_dim=1)

        # set priors
        for param, prior in ou_priors.items():
            setattr(ou_kernel, param, prior)
        for param, prior in osc_priors.items():
            setattr(osc_kernel, param, prior)

        # combine kernels
        kernel = Product(ou_kernel, osc_kernel)

        super().__init__(kernel, LBFGS, variance_loc=["kern0", "variance_map"])


class NoiseModel(GaussianProcess):
    """
    Noise model class
    """

    def __init__(self, priors: Dict[str, PyroParam | PyroSample]):
        # create kernel
        kernel = RBF(input_dim=1)
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        super().__init__(kernel, LBFGS)


# ------------------------------- #
# --- Gaussian Process helpers --- #
# ------------------------------- #


def detrend(
    X: Tensor, y: Tensor, detrend_lengthscale: float, verbose: bool = False
) -> Optional[Tuple[Tensor, GaussianProcess]]:
    """
    Detrend stochastic process using RBF process

    :param Tensor X: Input domain
    :param Tensor y: Target values
    :param float detrend_lengthscale: Lengthscale of the detrending process

    :return Tuple[Tensor, Tensor, Tensor]: mean, variance, detrended values or None
    """
    priors = {
        "lengthscale": PyroParam(
            tensor(detrend_lengthscale), constraint=greater_than(7.0)
        ),
    }

    noise_model = NoiseModel(priors)
    success = noise_model.fit(
        X, y, Trace_ELBO().differentiable_loss, num_steps=15, verbose=verbose
    )

    if not success:
        return None

    y_detrended = y - noise_model.mean

    return y_detrended, noise_model


def background_noise(
    X: Tensor, bckgd: Tensor, bckgd_length: Tensor, M: int, verbose: bool = False
) -> Tuple[Tensor, list[GaussianProcess]]:
    """
    Fit a background noise model to the data

    :param Tensor X: Input domain
    :param Tensor bckgd: Background traces
    :param Tensor bckgd_length: Length of each background trace
    :param int M: Count of background regions
    :param bool verbose: Print information

    :return: Standard deviation of the noise model, list of noise models
    """
    priors = {
        "lengthscale": PyroParam(tensor(7.1), constraint=greater_than(0.0)),
    }

    std_tensor = zeros(M)
    models = []

    for i in range(M):
        X_curr = X[: bckgd_length[i]]
        y_curr = bckgd[: bckgd_length[i], i]

        m = NoiseModel(priors)
        success = m.fit(
            X_curr,
            y_curr,
            Trace_ELBO().differentiable_loss,
            num_steps=100,
            verbose=verbose,
        )

        if not success:
            raise RuntimeError(f"Failed to fit background noise model {i}")

        models.append(m)
        std_tensor[i] = sqrt(m.fit_gp.kernel.variance)

    std = mean(std_tensor)

    print("Background noise model:")
    print(f"Standard deviation: {std}")

    return std, models
