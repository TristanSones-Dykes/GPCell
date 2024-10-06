# Forwarding/lib imports
import pyro.distributions
import torch.distributions.constraints
from . import kernels

# Logic imports
import pyro
import pyro.contrib.gp.kernels
import pyro.nn
import pyro.distributions as dist
from pyro.nn.module import PyroSample, PyroParam
import torch
import matplotlib.pyplot as plt


class GaussianProcess:
    """
    Gaussian Process class for fitting and evaluating parameters
    """
    def __init__(self, kernel: pyro.contrib.gp.kernels.Isotropy, optimizer: torch.optim.Optimizer):
        self.kernel = kernel
        """Kernel for the Gaussian Process"""
        self.optimizer = optimizer
        """Optimizer for the Gaussian Process"""

    def fit(self, X: torch.Tensor, y: torch.Tensor, loss_fn: torch.nn.Module, lr: float = 0.01, num_steps: int = 1000, priors: dict = {}, verbose: bool = False):
        """
        Fit the Gaussian Process model, saves the model and training values for later use if needed.

        :param torch.Tensor X: Input domain
        :param torch.Tensor y: Target values
        :param torch.nn.Module loss_fn: Loss function
        :param float lr: Learning rate
        :param int num_steps: Number of steps
        :param dict priors: Priors for the kernel parameters
        :param bool verbose: Print training information

        :return: None
        """
        pyro.clear_param_store()

        # initialise kernel
        kernel = self.kernel(input_dim = 1)

        # set priors
        for param, prior in priors.items():
            setattr(kernel, param, prior)

        # gaussian regression
        sgpr = pyro.contrib.gp.models.GPRegression(X, y, kernel, jitter=1.0e-5)
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

        losses = []
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(sgpr.model, sgpr.guide)
            loss.backward()
            optimizer.step(closure)

            losses.append(loss.item())
            if verbose and (i % 100 == 0 or i == num_steps-1):
                print(f"lengthscale: {sgpr.kernel.lengthscale.item()}")
        print("Final lengthscale values...")
        print(f"lengthscale: {sgpr.kernel.lengthscale.item()}")
        
        # save data vals for test and plotting
        self.XRANGE = [X.min().item(), X.max().item()]
        self.NUM_POINTS = X.shape[0]
        self.X_true = X
        self.y_true = y

        self.fit_gp = sgpr

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the mean and standard deviation of the Gaussian Process

        :param torch.Tensor X: Input domain

        :return: mean, sd
        """
        with torch.no_grad():
            mean, cov = self.fit_gp(X, full_cov=True, noiseless=False)
            sd = cov.diag().sqrt()

        return mean, sd


    def test_plot(self, plot_sd: bool = False):
        """
        Create a test plot of the fitted model on the training data
        
        :return: None
        """
        # check if fit_gp exists
        if not hasattr(self, "fit_gp"):
            raise AttributeError("Please fit the model first")
        
        # plot
        Xtest = self.X_true 
        with torch.no_grad():
            mean, cov = self.fit_gp(Xtest, full_cov=True, noiseless=False)
            sd = cov.diag().sqrt()

        plt.plot(Xtest, mean, zorder=1, c='k')
        if plot_sd:
            plt.plot(Xtest, mean + 2*sd, zorder=0, c='r')
            plt.plot(Xtest, mean - 2*sd, zorder=0, c='r')

        plt.plot(self.X_true, self.y_true, zorder=0, c='b')


