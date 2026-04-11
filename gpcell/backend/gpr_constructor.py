# Standard Library Imports
import operator
from functools import reduce
from typing import Optional

# Third-Party Library Imports
import numpy as np

# Direct Namespace Imports
from gpflow.kernels import Coregion
from gpflow.likelihoods import Gaussian
from gpflow.models import GPMC, GPR
from gpflow.utilities import set_trainable

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

    def __call__(self, X: Ndarray, y: Ndarray) -> GPR | GPMC:
        # create new kernel and define model
        kernel = self.kernel()
        if self.coregion_output_dim is not None:
            X, y = self._prepare_coregion_data(X, y)
            kernel.active_dims = [0]
            kernel = kernel * Coregion(
                output_dim=self.coregion_output_dim,
                rank=self.coregion_rank,
                active_dims=[1],
            )

        match self.mcmc:
            case True:
                likelihood = Gaussian()
                model = GPMC((X, y), kernel, likelihood)
            case False:
                model = GPR((X, y), kernel)
            case _:
                raise ValueError("Invalid MCMC value")

        prior_dict = self.prior_gen()

        # if MCMC, set priors on unconstrained space
        if self.mcmc and len(prior_dict) == 4:
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
