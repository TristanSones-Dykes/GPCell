from pyro.contrib.gp.kernels import Isotropy
import torch

# Isoptropic kernels
class Matern12(Isotropy):
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r = self._scaled_dist(X, Z)
        return self.variance * torch.exp(-r)

