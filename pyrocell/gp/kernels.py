from pyro.contrib.gp.kernels import Isotropy
import torch

# Isoptropic kernels
class Matern12(Isotropy):
    """
    Matern 1/2 kernel, non-smooth and non-differentiable
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        """
        :param int input_dim: Dimension of input
        :param torch.Tensor variance: Variance of the kernel
        :param torch.Tensor lengthscale: Lengthscale of the kernel
        :param list active_dims: Active dimensions
        """
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        """
        Compute the covariance matrix of kernel on inputs X and Z
        
        :param torch.Tensor X: Input tensor
        :param torch.Tensor Z: Input tensor
        :param bool diag: Return the diagonal of the covariance matrix
        """
        if diag:
            return self._diag(X)

        r = self._scaled_dist(X, Z)
        return self.variance * torch.exp(-r)

