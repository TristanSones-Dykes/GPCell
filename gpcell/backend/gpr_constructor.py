# Standard Library Imports
from typing import Optional
from functools import reduce
import operator

# Third-Party Library Imports

# Direct Namespace Imports
from gpflow.models import GPR, GPMC
from gpflow.likelihoods import Gaussian
from gpflow.utilities import set_trainable

# Internal Project Imports
from ._types import Ndarray, GPKernel, GPPriorFactory, GPPriorTrainingFlag, GPOperator
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

    def __call__(self, X: Ndarray, y: Ndarray) -> GPR | GPMC:
        # create new kernel and define model
        kernel = self.kernel()

        match self.mcmc:
            case True:
                likelihood = Gaussian()
                model = GPMC((X, y), kernel, likelihood)
            case False:
                model = GPR((X, y), kernel)
            case _:
                raise ValueError("Invalid MCMC value")

        # assign priors
        prior_dict = self.prior_gen()
        multiple_assign(model, prior_dict)

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
