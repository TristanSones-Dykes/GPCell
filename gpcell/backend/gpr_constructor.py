# Standard Library Imports
import operator
from functools import reduce
from typing import Optional

# Third-Party Library Imports
import numpy as np

# Direct Namespace Imports
from gpflow.kernels import Coregion
from gpflow.likelihoods import Gaussian
from gpflow.models import GPMC, GPR, SVGP
from gpflow.utilities import set_trainable
from sklearn.cluster import KMeans

# Internal Project Imports
from ._types import GPKernel, GPOperator, GPPriorFactory, GPPriorTrainingFlag, Ndarray
from ._utils import multiple_assign


class GPRConstructor:
    """
    Gaussian Process Regression constructor
    """

    def __init__(
        self,
        kernels: GPKernel,
        prior_gen: GPPriorFactory,
        trainable: GPPriorTrainingFlag = {},
        operator: Optional[GPOperator] = operator.mul,
        mcmc: bool = False,
        coregion_output_dim: int | None = None,
        coregion_rank: int = 1,
        inference_method: str | None = None,
        num_inducing: int = 20,
        svgp_batch_size: int = 64,
        svgp_adam_steps: int = 1000,
        svgp_learning_rate: float = 0.01,
        kmeans_random_state: int = 0,
    ):
        """
        Defines the kernel as a single kernel or a composite kernel using given operator

        Parameters
        ----------
        kernels : Union[Kernel, List[Kernel]]
            Single kernel or list of kernels
        prior_gen : Callable[..., GPPriors]
            Function that generates the priors for the GP model
        trainable : Union[Mapping[str, bool], Mapping[Tuple[int, str], bool]], optional
            Dictionary to set trainable parameters, by default {}
        operator : Optional[Callable[[Kernel, Kernel], Kernel]], optional
            Operator to combine multiple kernels, by default None
        mcmc : bool, optional
            Use MCMC for inference, by default False
        coregion_output_dim : int, optional
            Number of coupled outputs to model with a Linear Model of Coregionalization.
            When provided, two-dimensional targets with shape ``(n_times, n_outputs)``
            are flattened and the input matrix is augmented with a categorical output
            index for ``gpflow.kernels.Coregion``.
        coregion_rank : int, optional
            Rank of the coregionalization matrix, by default 1.
        inference_method : str, optional
            Inference method to use. Supported values are ``"GPR"``, ``"GPMC"``,
            and ``"SVGP"``. If omitted, the value is inferred from ``mcmc`` for
            backwards compatibility.
        num_inducing : int, optional
            Number of inducing points to initialize with KMeans when using SVGP,
            by default 20.
        svgp_batch_size : int, optional
            Minibatch size used by Adam optimization for SVGP, by default 64.
        svgp_adam_steps : int, optional
            Number of Adam optimization steps for SVGP, by default 1000.
        svgp_learning_rate : float, optional
            Adam learning rate for SVGP, by default 0.01.
        kmeans_random_state : int, optional
            Random seed passed to KMeans for inducing point initialization, by default 0.
        """
        match kernels:
            case type():
                self.kernel = kernels
            case list():
                assert operator is not None, ValueError(
                    "Operator must be provided for composite kernels"
                )
                self.kernel = lambda: reduce(operator, [k() for k in kernels])
            case _:
                raise TypeError(f"Invalid kernel type: {type(kernels)}")

        self.prior_gen = prior_gen
        self.trainable = trainable
        self.mcmc = mcmc
        self.coregion_output_dim = coregion_output_dim
        self.coregion_rank = coregion_rank
        self.inference_method = self._resolve_inference_method(inference_method, mcmc)
        self.num_inducing = num_inducing
        self.svgp_batch_size = svgp_batch_size
        self.svgp_adam_steps = svgp_adam_steps
        self.svgp_learning_rate = svgp_learning_rate
        self.kmeans_random_state = kmeans_random_state

    def __call__(self, X: Ndarray, y: Ndarray) -> GPR | GPMC | SVGP:
        # create new kernel and define model
        kernel = self.kernel()
        if self.coregion_output_dim is not None:
            X, y = self.prepare_training_data(X, y)
            kernel.active_dims = [0]
            kernel = kernel * Coregion(
                output_dim=self.coregion_output_dim,
                rank=self.coregion_rank,
                active_dims=[1],
            )
        else:
            X, y = self.prepare_training_data(X, y)

        match self.inference_method:
            case "GPMC":
                likelihood = Gaussian()
                model = GPMC((X, y), kernel, likelihood)
            case "GPR":
                model = GPR((X, y), kernel)
            case "SVGP":
                likelihood = Gaussian()
                inducing_points = self._initialise_inducing_points(X)
                model = SVGP(
                    kernel,
                    likelihood,
                    inducing_points,
                    num_data=X.shape[0],
                )
            case _:
                raise ValueError(f"Invalid inference method: {self.inference_method}")

        prior_dict = self.prior_gen()

        # if MCMC, set priors on unconstrained space
        if self.inference_method == "GPMC" and len(prior_dict) == 4:
            # create (prior: "unconstrained") mapping
            constrain_map = {}
            for key, _ in prior_dict.items():
                # initialise path and remove prior if it exists in path
                attrs = key.split(".")
                if attrs[-1] == "prior":
                    attrs.pop()

                # add to constrain_map
                attrs.append("prior_on")
                new_key = ".".join(attrs)
                constrain_map[new_key] = "unconstrained"

            # set priors to unconstrained parameters
            multiple_assign(model, constrain_map)

        # assign priors
        multiple_assign(model, prior_dict)
        # print(model.likelihood.variance.prior_on)

        # set trainable parameters
        for param, trainable in self.trainable.items():
            match param:
                case str(s):
                    obj = model
                    attrs = s.split(".")
                case (int(i), str(s)):
                    obj = model.kernel.kernels[i]  # type: ignore
                    attrs = s.split(".")

            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            set_trainable(getattr(obj, attrs[-1]), trainable)

        return model

    def _resolve_inference_method(self, inference_method: str | None, mcmc: bool) -> str:
        """
        Resolve the inference method while preserving the legacy ``mcmc`` flag.

        Parameters
        ----------
        inference_method : str, optional
            Explicit inference method.
        mcmc : bool
            Legacy flag for MCMC inference.

        Returns
        -------
        str
            Canonical inference method name.
        """
        if inference_method is None:
            return "GPMC" if mcmc else "GPR"

        canonical = inference_method.upper()
        if canonical == "MCMC":
            canonical = "GPMC"
        if canonical not in {"GPR", "GPMC", "SVGP"}:
            raise ValueError(f"Invalid inference method: {inference_method}")
        if canonical == "SVGP" and mcmc:
            raise ValueError("SVGP inference cannot be combined with mcmc=True.")
        return canonical

    def _prepare_coregion_data(self, X: Ndarray, y: Ndarray) -> tuple[Ndarray, Ndarray]:
        """
        Convert multivariate trace data into the augmented format expected by Coregion.

        Parameters
        ----------
        X : Ndarray
            One-dimensional time input with shape ``(n_times,)`` or ``(n_times, 1)``.
        y : Ndarray
            Multivariate targets with shape ``(n_times, n_outputs)``.

        Returns
        -------
        tuple[Ndarray, Ndarray]
            Augmented inputs with ``[time, output_index]`` columns and flattened targets.
        """
        X_array = np.asarray(X, dtype=float).reshape(-1, 1)
        y_array = np.asarray(y, dtype=float)

        if y_array.ndim != 2:
            raise ValueError("Coregion models require y with shape (n_times, n_outputs).")

        n_times, n_outputs = y_array.shape
        if X_array.shape[0] != n_times:
            raise ValueError("X and y must contain the same number of time points.")
        if n_outputs != self.coregion_output_dim:
            raise ValueError(
                f"y has {n_outputs} outputs, expected {self.coregion_output_dim}."
            )

        times = np.repeat(X_array, n_outputs, axis=0)
        output_index = np.tile(np.arange(n_outputs, dtype=float), n_times).reshape(-1, 1)
        X_augmented = np.hstack([times, output_index])
        y_flat = y_array.reshape(-1, 1)
        return X_augmented, y_flat

    def prepare_training_data(self, X: Ndarray, y: Ndarray) -> tuple[Ndarray, Ndarray]:
        """
        Prepare training arrays for the configured model structure.

        Parameters
        ----------
        X : Ndarray
            Input locations.
        y : Ndarray
            Target values.

        Returns
        -------
        tuple[Ndarray, Ndarray]
            Rank-two arrays ready for the configured GPflow model.
        """
        if self.coregion_output_dim is not None:
            return self._prepare_coregion_data(X, y)
        return self._prepare_model_data(X, y)

    def _prepare_model_data(self, X: Ndarray, y: Ndarray) -> tuple[Ndarray, Ndarray]:
        """
        Convert model inputs to rank-two NumPy arrays.

        Parameters
        ----------
        X : Ndarray
            Input locations.
        y : Ndarray
            Target values.

        Returns
        -------
        tuple[Ndarray, Ndarray]
            Rank-two input and target arrays.
        """
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float)

        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        return X_array, y_array

    def _initialise_inducing_points(self, X: Ndarray) -> Ndarray:
        """
        Initialize SVGP inducing points with KMeans centroids.

        Parameters
        ----------
        X : Ndarray
            Rank-two model input array.

        Returns
        -------
        Ndarray
            Initial inducing point locations.
        """
        X_array = np.asarray(X, dtype=float)
        n_inducing = min(self.num_inducing, X_array.shape[0])
        if n_inducing < 1:
            raise ValueError("SVGP requires at least one inducing point.")
        if n_inducing == X_array.shape[0]:
            return X_array.copy()

        kmeans = KMeans(
            n_clusters=n_inducing,
            random_state=self.kmeans_random_state,
            n_init=10,
        )
        return kmeans.fit(X_array).cluster_centers_
