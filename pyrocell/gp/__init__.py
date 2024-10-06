# Forwarding imports
from . import kernels

# Logic imports
import pyro.contrib
import pyro.nn
import torch
import pyro
import matplotlib.pyplot as plt
import numpy as np


class GaussianProcess:
    def __init__(self, kernel: pyro.contrib.gp.kernels.Isotropy, optimizer: torch.optim.Optimizer):
        self.kernel = kernel
        self.optimizer = optimizer

    def fit(self, X: torch.Tensor, y: torch.Tensor, loss_fn: torch.nn.Module, lr: float = 0.01, num_steps: int = 1000, priors: dict = {}, verbose: bool = False):
        # initialise kernel
        kernel = self.kernel(input_dim = 1)

        # set priors
        for param, prior in priors.items():
            setattr(kernel, param, pyro.nn.PyroSample(prior))

        # gaussian regression
        sgpr = pyro.contrib.gp.models.GPRegression(X, y, kernel, jitter=1.0e-5)
        optimizer = self.optimizer(sgpr.parameters(), lr=lr)

        # check if closure is needed
        if optimizer.__class__.__name__ == "LBFGS":
            print("Using LBFGS closure")
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
                print("lengthscale_map: {}   lengthscale: {}".format(
                    sgpr.kernel.lengthscale_map.detach().item(),
                    sgpr.kernel.lengthscale.item()))
        print("Final lengthscale values...")
        print("lengthscale_map: {}   lengthscale: {}".format(
            sgpr.kernel.lengthscale_map.detach().item(),
            sgpr.kernel.lengthscale.item()))
        
        # save data vals for test and plotting
        self.XRANGE = [X.min().item(), X.max().item()]
        self.NUM_POINTS = X.shape[0]
        self.X_true = X.numpy()
        self.y_true = y.numpy()

        self.fit_gp = sgpr

    def test_plot(self):
        # check if fit_gp exists
        if not hasattr(self, "fit_gp"):
            raise AttributeError("Please fit the model first")
        
        # plot
        Xtest = np.linspace(self.XRANGE[0], self.XRANGE[1], self.NUM_POINTS*2)[..., np.newaxis]
        Xtest = torch.from_numpy(Xtest)
        with torch.no_grad():
            mean, cov = self.fit_gp(Xtest, full_cov=True, noiseless=False)
            sd = cov.diag().sqrt()

        plt.plot(Xtest, mean, zorder=1, c='r')
        plt.scatter(self.X_true, self.y_true, zorder=0)