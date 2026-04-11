import numpy as np
from gpflow.kernels import SquaredExponential
from gpflow.models import SVGP
from sklearn.cluster import KMeans

from gpcell.backend import GaussianProcess, GPRConstructor


def _likelihood_prior():
    """Return a small observation-noise prior for deterministic SVGP tests."""
    return {"likelihood.variance": 0.01}


def test_gpr_constructor_builds_svgp_with_kmeans_inducing_points():
    """Verify SVGP construction initializes inducing points with KMeans centroids."""
    X = np.linspace(0.0, 4.0, 20).reshape(-1, 1)
    y = np.sin(X)
    constructor = GPRConstructor(
        SquaredExponential,
        _likelihood_prior,
        inference_method="SVGP",
        num_inducing=5,
        kmeans_random_state=7,
    )

    model = constructor(X, y)
    expected_Z = KMeans(n_clusters=5, random_state=7, n_init=10).fit(X).cluster_centers_

    assert isinstance(model, SVGP)
    np.testing.assert_allclose(
        np.sort(model.inducing_variable.Z.numpy(), axis=0),
        np.sort(expected_Z, axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isfinite(model.training_loss((X, y)).numpy())


def test_gaussian_process_svgp_fit_predicts_and_reduces_elbo_loss():
    """Verify SVGP fitting with Adam reduces loss and produces finite predictions."""
    X = np.linspace(0.0, 6.0, 40).reshape(-1, 1)
    y = np.sin(X)
    constructor = GPRConstructor(
        SquaredExponential,
        _likelihood_prior,
        inference_method="SVGP",
        num_inducing=8,
        svgp_batch_size=40,
        svgp_adam_steps=50,
        svgp_learning_rate=0.02,
        kmeans_random_state=3,
    )
    gp = GaussianProcess(constructor)

    gp.fit(X, y, Y_var=True)
    mean, variance = gp(X[:5])

    assert gp.training_loss_history[-1] < gp.training_loss_history[0]
    assert np.isfinite(gp.log_posterior())
    assert mean.shape == (5, 1)
    assert variance.shape == (5, 1)
    assert gp.Y_var.shape == (40, 1)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(variance))
