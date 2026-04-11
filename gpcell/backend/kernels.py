# Standard Library Imports
import math

# Direct Namespace Imports
import gpflow

# Third-Party Library Imports
import tensorflow as tf
from gpflow.utilities import positive, to_default_float


class TimeVaryingCosine(gpflow.kernels.Kernel):
    """
    Cosine kernel with a linearly time-varying frequency.

    This non-stationary kernel uses the phase integral
    ``u(x) = beta_0 * x + 0.5 * beta_1 * x**2`` and defines covariance as
    ``variance * cos(2 * pi * (u(x) - u(x')))``. It is intended for chirp-like
    oscillatory traces where biological frequency can drift over time.

    Parameters
    ----------
    beta_0 : float, optional
        Base oscillation frequency. The parameter is constrained to be positive.
    beta_1 : float, optional
        Linear rate of frequency change.
    variance : float, optional
        Marginal variance of the kernel. The parameter is constrained to be positive.
    active_dims : slice or sequence of int, optional
        Input dimensions over which the kernel acts.
    name : str, optional
        Name passed to the GPflow kernel base class.
    """

    def __init__(
        self,
        beta_0: float = 1.0,
        beta_1: float = 0.0,
        variance: float = 1.0,
        active_dims=None,
        name: str | None = None,
    ):
        super().__init__(active_dims=active_dims, name=name)
        self.beta_0 = gpflow.Parameter(beta_0, transform=positive())
        self.beta_1 = gpflow.Parameter(beta_1)
        self.variance = gpflow.Parameter(variance, transform=positive())

    def K(self, X, X2=None):
        """
        Return the full covariance matrix.

        Parameters
        ----------
        X : tensor_like
            First input matrix.
        X2 : tensor_like, optional
            Second input matrix. If omitted, ``X`` is used.

        Returns
        -------
        tf.Tensor
            Covariance matrix evaluated at ``X`` and ``X2``.
        """
        if X2 is None:
            X2 = X

        u_X = self._phase_integral(X)
        u_X2 = self._phase_integral(X2)
        angular_frequency = tf.cast(2.0 * math.pi, u_X.dtype)
        phase_diff = u_X - tf.transpose(u_X2)
        return self.variance * tf.math.cos(angular_frequency * phase_diff)

    def K_diag(self, X):
        """
        Return the diagonal covariance.

        Parameters
        ----------
        X : tensor_like
            Input matrix.

        Returns
        -------
        tf.Tensor
            Vector containing the marginal variance for each input row.
        """
        variance = tf.cast(self.variance, dtype=X.dtype)
        return tf.fill(tf.shape(X)[:-1], variance)

    def Kdiag(self, X):
        """
        Return the diagonal covariance using the legacy GPflow method name.

        Parameters
        ----------
        X : tensor_like
            Input matrix.

        Returns
        -------
        tf.Tensor
            Vector containing the marginal variance for each input row.
        """
        return self.K_diag(X)

    def _phase_integral(self, X):
        """
        Evaluate the linear chirp phase integral.

        Parameters
        ----------
        X : tensor_like
            Input matrix.

        Returns
        -------
        tf.Tensor
            Phase integral values for each input row.
        """
        X = to_default_float(X)
        return self.beta_0 * X + to_default_float(0.5) * self.beta_1 * tf.math.square(X)
