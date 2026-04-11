import numpy as np
import pytest
import tensorflow as tf
from gpflow.kernels import Matern12, Product
from gpflow.models import GPR

from gpcell.backend import GPRConstructor, TimeVaryingCosine


def _empty_prior():
    """Return an empty prior mapping for constructor-only tests."""
    return {}


def test_time_varying_cosine_covariance_matches_phase_integral():
    """Verify the chirp kernel covariance matches the analytical phase integral."""
    kernel = TimeVaryingCosine(beta_0=0.5, beta_1=0.25, variance=2.0)
    X = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float64)

    covariance = kernel.K(X).numpy()
    u = 0.5 * X.numpy() + 0.5 * 0.25 * np.square(X.numpy())
    expected = 2.0 * np.cos(2.0 * np.pi * (u - u.T))

    np.testing.assert_allclose(covariance, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(kernel.Kdiag(X).numpy(), np.full(3, 2.0))


def test_gpr_constructor_builds_coregionalized_model():
    """Verify LMC construction compiles a finite-loss GPflow GPR model."""
    X = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    y = np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 0])])
    constructor = GPRConstructor(
        Matern12,
        _empty_prior,
        coregion_output_dim=2,
        coregion_rank=1,
    )

    model = constructor(X, y)
    loss = model.training_loss()

    assert isinstance(model, GPR)
    assert isinstance(model.kernel, Product)
    assert model.data[0].shape == (10, 2)
    assert model.data[1].shape == (10, 1)
    assert np.isfinite(loss.numpy())


def test_gpr_constructor_rejects_coregion_output_mismatch():
    """Verify LMC construction rejects mismatched multivariate output dimensions."""
    X = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    y = np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 0])])
    constructor = GPRConstructor(
        Matern12,
        _empty_prior,
        coregion_output_dim=3,
        coregion_rank=1,
    )

    with pytest.raises(ValueError, match="expected 3"):
        constructor(X, y)
