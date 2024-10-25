# External library imports
from typing import List, Optional
from torch import Tensor, exp
from pyro.contrib.gp.kernels import Isotropy

# Isotropic kernels
class Matern12(Isotropy):
    """
    Matern 1/2 kernel, non-smooth and non-differentiable
    """
    def __init__(self, input_dim: int, variance: Optional[Tensor] = None, lengthscale: Optional[Tensor] = None, active_dims: Optional[List] = None):
        """
        :param int input_dim: Dimension of input
        :param Tensor variance: Variance of the kernel (optional)
        :param Tensor lengthscale: Lengthscale of the kernel (optional)
        :param list active_dims: Active dimensions (optional)
        """
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X: Tensor, Z: Optional[Tensor] = None, diag: bool = False):
        """
        Compute the covariance matrix of kernel on inputs X and Z
        
        :param Tensor X: Input tensor
        :param Tensor Z: Input tensor (optional)
        :param bool diag: Return the diagonal of the covariance matrix
        """
        if diag:
            return self._diag(X)

        r = self._scaled_dist(X, Z)
        return self.variance * exp(-r)

