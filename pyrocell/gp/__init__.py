# Standard library imports
from typing import Callable, Dict, Optional, Tuple

# External library imports
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# External type imports
from torch import Tensor
from torch.optim.optimizer import Optimizer

# Pyro-related imports
import pyro
from pyro.contrib.gp.models import GPRegression
import pyro.contrib.gp.kernels as pyro_kernels
from pyro.nn import PyroParam, PyroSample
from pyro.distributions import constraints

# Internal imports
from pyrocell.gp.kernels import Matern12


# -------------------------------- #
# --- Gaussian Process classes --- #
# -------------------------------- #

class GaussianProcess:
    """
    Gaussian Process class for fitting and evaluating parameters
    """
    def __init__(self, kernel: pyro_kernels.Isotropy, optimizer: Optimizer):
        self.kernel = kernel
        """Kernel for the Gaussian Process"""
        self.optimizer = optimizer
        """Optimizer for the Gaussian Process"""

    def __call__(self, X: Tensor, full_cov: bool = True, noiseless: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Evaluate the Gaussian Process on the input domain

        :param Tensor X: Input domain
        :param bool full_cov: Return full covariance matrix
        :param bool noiseless: Return noiseless predictions

        :return Tuple[Tensor, Tensor]: Mean and standard deviation
        """
        with torch.no_grad():
            mean, cov = self.fit_gp(X, full_cov=full_cov, noiseless=noiseless)

        if full_cov:
            var = cov.diag()

        return mean, var

    def fit(self, X: Tensor, y: Tensor, loss_fn: Callable[..., Tensor], lr: float = 0.01, noise: Optional[Tensor] = None, num_steps: int = 1000, priors: Dict[str, PyroParam | PyroSample] = {}, verbose: bool = False):
        """
        Fit the Gaussian Process model, saves the model and training values for later use if needed.

        :param Tensor X: Input domain
        :param Tensor y: Target values
        :param Callable loss_fn: Loss function
        :param float lr: Learning rate
        :param int num_steps: Number of steps
        :param Dict[str, PyroPrior] priors: Priors for the kernel parameters
        :param bool verbose: Print training information

        :return: None
        """
        pyro.clear_param_store()

        # check if kernel is class or object
        if isinstance(self.kernel, type):
            kernel = self.kernel(input_dim = 1)
        else:
            kernel = self.kernel

        # set priors
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        # gaussian regression
        sgpr = GPRegression(X, y, kernel, noise=noise, jitter=1.0e-5)
        optimizer = self.optimizer(sgpr.parameters(), lr=lr)

        if verbose:
            print(sgpr)
            print(optimizer)

        # check if closure is needed
        if optimizer.__class__.__name__ == "LBFGS":
            def closure():
                optimizer.zero_grad()
                loss = loss_fn(sgpr.model, sgpr.guide)
                loss.backward()
                return loss
        else:
            closure = None

        # training loop
        losses = []
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(sgpr.model, sgpr.guide)
            loss.backward()
            optimizer.step(closure)

            losses.append(loss.item())
            if verbose and (i % 100 == 0 or i == num_steps-1):
                print(f"lengthscale: {sgpr.kernel.lengthscale.item()}")

        # display final trained parameters
        if verbose:
            for param in priors.keys():
                print(f"{param}: {getattr(sgpr.kernel, param).item()}")

        # calculate log posterior density
        with torch.no_grad():
            self.loss = loss_fn(sgpr.model, sgpr.guide)

        # save data vals for test and plotting
        self.X_true = X
        self.y_true = y
        
        with torch.no_grad():
            self.mean, self.cov = sgpr(X, full_cov=True, noiseless=False)
            self.std = self.cov.diag().sqrt()

        self.fit_gp = sgpr
        self.params = sgpr.parameters()

    def test_plot(self, plot_sd: bool = False):
        """
        Create a test plot of the fitted model on the training data
        
        :return: None
        """
        # check if fit_gp exists
        if not hasattr(self, "fit_gp"):
            raise AttributeError("Please fit the model first")
        
        # plot
        plt.plot(self.X_true, self.mean, zorder=1, c='k')
        if plot_sd:
            plt.plot(self.X_true, self.mean + 2*self.std, zorder=0, c='r')
            plt.plot(self.X_true, self.mean - 2*self.std, zorder=0, c='r')

        plt.plot(self.X_true, self.y_true, zorder=0, c='b')


class OU(GaussianProcess):
    """
    Ornstein-Uhlenbeck process class
    """
    def __init__(self, priors: Dict[str, PyroParam | PyroSample]):
        # create kernel
        kernel = Matern12(input_dim = 1)
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        super().__init__(kernel, optim.LBFGS)


class OUosc(GaussianProcess):
    """
    Ornstein-Uhlenbeck process with an oscillator (cosine) kernel
    """
    def __init__(self, ou_priors: Dict[str, PyroParam | PyroSample], osc_priors: Dict[str, PyroParam | PyroSample]):
        # create kernels
        ou_kernel = Matern12(input_dim = 1)
        osc_kernel = pyro_kernels.Cosine(input_dim = 1)

        # set priors
        for param, prior in ou_priors.items():
            setattr(ou_kernel, param, prior)
        for param, prior in osc_priors.items():
            setattr(osc_kernel, param, prior)

        # combine kernels
        kernel = pyro_kernels.Product(ou_kernel, osc_kernel)

        super().__init__(kernel, optim.LBFGS)


# ------------------------------- #
# --- Gaussian Process helpers --- #
# ------------------------------- #

def noise_model(X: Tensor, y: Tensor, priors: Dict[str, PyroParam | PyroSample]) -> GaussianProcess:
    """
    Fit a squared exponential kernel to the data

    :param Tensor X: Input domain
    :param Tensor y: Target values
    :param Dict[str, PyroPrior] priors: Priors for the kernel parameters
    """
    process = GaussianProcess(pyro_kernels.RBF, optim.LBFGS)
    process.fit(X, y, pyro.infer.Trace_ELBO().differentiable_loss, priors=priors, num_steps=100)

    return process


def detrend(X: Tensor, y: Tensor, detrend_lengthscale: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Detrend stochastic process using RBF process

    :param Tensor X: Input domain
    :param Tensor y: Target values
    :param float detrend_lengthscale: Lengthscale of the detrending process

    :return Tuple[Tensor, Tensor, Tensor]: mean, variance, detrended values
    """
    priors = {
        "lengthscale": PyroParam(torch.tensor(detrend_lengthscale), constraint=constraints.greater_than(0.0)),
    }

    detrend_gp = noise_model(X, y, priors)
    mean, var = detrend_gp(X, full_cov=True, noiseless=False)

    y_detrended = y - mean
    y_detrended = y_detrended - torch.mean(y_detrended)

    return mean, var, y_detrended


def background_noise(X: Tensor, bckgd: Tensor, bckgd_length: Tensor, M: int, verbose: bool = False) -> Tuple[Tensor, list[GaussianProcess]]:
    """
    Fit a background noise model to the data

    :param Tensor X: Input domain
    :param Tensor bckgd: Background X-series data
    :param Tensor bckgd_length: Length of each background trace
    :param int M: Count of background regions
    :param bool verbose: Print information

    :return: Standard deviation of the noise model, list of noise models
    """
    priors = {
        "lengthscale": PyroParam(torch.tensor(7.1), constraint=constraints.greater_than(0.0)),
    }

    std_tensor = torch.zeros(M)
    models = []

    for i in range(M):
        X_curr = X[:bckgd_length[i]]
        y_curr = bckgd[:bckgd_length[i],i,None]  
        y_curr = y_curr - torch.mean(y_curr)

        # remove y-dim
        y_curr = y_curr.reshape(-1)
        
        m = noise_model(X_curr, y_curr, priors)

        models.append(m)
        std_tensor[i] = torch.pow(m.fit_gp.kernel.variance, 0.5)

    std = torch.mean(std_tensor)

    if verbose:
        print("Background noise model:")
        print(f"Standard deviation: {std}")

    return std, models