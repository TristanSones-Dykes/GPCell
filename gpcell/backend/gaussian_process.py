# Standard Library Imports
from typing import List, Optional, Tuple, cast

# Third-Party Library Imports
import matplotlib.pyplot as plt
import tensorflow as tf

# Direct Namespace Imports
from gpflow.optimizers import Scipy, SamplingHelper
from gpflow.models import GPR, GPMC
from gpflow.posteriors import PrecomputeCacheType
from gpflow.utilities import to_default_float as f64, parameter_dict, print_summary

from tensorflow_probability import mcmc
from tensorflow import Tensor

# Internal Project Imports
from ._types import Ndarray
from .gpr_constructor import GPRConstructor


NOCACHE = PrecomputeCacheType.NOCACHE


class GaussianProcess:
    """
    Gaussian Process model using GPflow.
    """

    def __init__(self, constructor: GPRConstructor):
        self.constructor = constructor
        """Regression with kernel for the Gaussian Process"""

    def __call__(
        self,
        X: Ndarray,
        full_cov: bool = False,
    ) -> Tuple[Ndarray, Ndarray]:
        """
        Evaluate the Gaussian Process on the input domain

        Parameters
        ----------
        X: Ndarray
                Input domain
        full_cov: bool
                Return full covariance matrix

        Returns
        -------
        Tuple[Tensor, Tensor]
                Mean and standard deviation

        """
        if not hasattr(self, "fit_gp"):
            raise ValueError("Model has not been fit yet.")

        fit_mean, fit_var = self.fit_gp.fused_predict_f(X, full_cov=full_cov)
        return fit_mean.numpy(), fit_var.numpy()  # type: ignore

    def fit(
        self,
        X: Ndarray,
        y: Ndarray,
        Y_var: bool = False,
        verbose: bool = False,
    ):
        """
        Fit the Gaussian Process model, saves the model and training values for later use if needed.

        Parameters
        ----------
        X: Ndarray
                Input domain
        y: Ndarray
                Target values
        Y_var: bool
                Calculate variance of missing data
        verbose: bool
                Print training information

        Returns
        -------
        bool
                Success status
        """
        gp_reg = self.constructor(X, y)

        opt = Scipy()
        opt.minimize(
            gp_reg.training_loss,
            gp_reg.trainable_variables,  # type: ignore
            options=dict(maxiter=100),
        )

        # Extract training values and model parameters
        self.log_posterior_density = gp_reg.log_posterior_density().numpy()  # type: ignore
        self.noise = gp_reg.likelihood.variance**0.5  # type: ignore

        # For plotting variance (lose detail on GPR.posterior)
        if Y_var:
            self.Y_var = gp_reg.predict_y(X)[1].numpy()  # type: ignore

        match gp_reg:
            case GPR():
                # Keep the posterior for predictions
                self.fit_gp = gp_reg.posterior()
            case GPMC():
                self._run_mcmc(gp_reg)

    def _run_mcmc(
        self,
        model: GPMC,
        sampler: str = "hmc",
        num_samples: int = 3000,
        num_burnin_steps: int = 3000,
        step_size: float = 0.01,
        num_leapfrog_steps: int = 5,
        target_accept: float = 0.9,
    ):
        # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
        sampler_helper = SamplingHelper(
            model.log_posterior_density, model.trainable_parameters
        )

        # define model
        match sampler.lower():
            case "hmc":
                integrator = mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=sampler_helper.target_log_prob_fn,
                    step_size=f64(step_size),
                    num_leapfrog_steps=num_leapfrog_steps,
                )
                kernel = mcmc.SimpleStepSizeAdaptation(
                    inner_kernel=integrator,
                    num_adaptation_steps=int(0.8 * num_burnin_steps),
                    target_accept_prob=f64(target_accept),
                    adaptation_rate=0.01,
                )

                # set up the chain
                @tf.function(reduce_retracing=True)
                def run_chain_fn():
                    return mcmc.sample_chain(
                        num_results=num_samples,
                        num_burnin_steps=num_burnin_steps,
                        current_state=sampler_helper.current_state,
                        kernel=kernel,
                        trace_fn=lambda _, pkr: (
                            pkr.inner_results.is_accepted,
                            pkr.inner_results.accepted_results.target_log_prob,
                        ),
                    )

                # run chain and extract samples
                self.samples, (_, self.log_prob_chain) = run_chain_fn()  # type: ignore
            case "nuts":
                integrator = mcmc.NoUTurnSampler(
                    target_log_prob_fn=sampler_helper.target_log_prob_fn,
                    step_size=f64(step_size),
                    max_tree_depth=5,
                )
                kernel = mcmc.DualAveragingStepSizeAdaptation(
                    inner_kernel=integrator,
                    num_adaptation_steps=int(0.8 * num_burnin_steps),
                    target_accept_prob=f64(target_accept),
                )

                # set up the chain
                @tf.function(reduce_retracing=True)
                def run_chain_fn():
                    return mcmc.sample_chain(
                        num_results=num_samples,
                        num_burnin_steps=num_burnin_steps,
                        current_state=sampler_helper.current_state,
                        kernel=kernel,
                        trace_fn=lambda _, pkr: (
                            pkr.inner_results.is_accepted,
                            pkr.inner_results.accepted_results.target_log_prob,
                        ),
                    )

                self.samples, (_, self.log_prob_chain) = run_chain_fn()  # type: ignore
            case _:
                raise ValueError(f"Invalid sampler: {sampler}")

        # take post-constrained samples
        self.parameter_samples = sampler_helper.convert_to_constrained_values(
            self.samples
        )

        # map parameter names to indices
        self.param_to_name = {
            param: name for name, param in parameter_dict(model).items()
        }
        self.name_to_index = {
            self.param_to_name[param]: i
            for i, param in enumerate(model.trainable_parameters)
        }
        self.trainable_parameters = model.trainable_parameters

    def log_posterior(
        self,
    ) -> Ndarray:
        """
        Compute the log posterior density of the model

        Returns
        -------
        Ndarray
            Log posterior density
        """
        return self.log_posterior_density

    def plot_gpr(
        self,
        plot_sd: bool = False,
        new_y: Optional[Ndarray] = None,
        new_label: str = "Altered data",
    ):
        """
        Plot the Gaussian Process model

        Parameters
        ----------
        plot_sd: bool
                Plot standard deviation
        new_y: Optional[Ndarray]
                New target values
        """
        if not hasattr(self, "fit_gp"):
            raise ValueError("Model has not been fit yet.")

        X = cast(Tensor, self.fit_gp.X_data)
        y = self.fit_gp.Y_data
        fit_y = self.fit_gp.fused_predict_f(X)[0]

        plt.plot(X, y, label="True data")

        match new_y:
            case None:
                plt.plot(X, fit_y, label="Fit process")
            case _:
                try:
                    plt.plot(X, new_y, label=new_label)
                    plt.plot(
                        X,
                        fit_y,
                        label="Fit process",
                        color="k",
                        alpha=0.5,
                        linestyle="dashed",
                    )
                except Exception as e:
                    print("Problem with new data to plot: ", e)

    def plot_samples(self, hyperparameters: List[str] | str):
        """
        Plot the samples from the MCMC chain

        Parameters
        ----------
        hyperparameters: List[str]
                List of hyperparameters to plot
        """
        # plot samples
        match hyperparameters:
            case str():
                plt.plot(
                    self.parameter_samples[self.name_to_index[hyperparameters]],
                )
                plt.xlabel("Sampler iteration")
                plt.ylabel(f"{hyperparameters} value")
            case list():
                for param_name in hyperparameters:
                    plt.plot(
                        self.parameter_samples[self.name_to_index[param_name]],
                        label=param_name,
                    )
                plt.legend()
                plt.xlabel("Sampler iteration")
                plt.ylabel("hyperparameter value")
            case _:
                raise ValueError(
                    "hyperparameters must be a string or a list of strings"
                )

    def plot_posterior_marginal(self, hyperparameter: str, constrained: bool = False):
        """
        Plot the posterior marginal of a hyperparameter

        Parameters
        ----------
        hyperparameter: str
                Hyperparameter to plot
        """
        # plot posterior marginal
        plt.hist(
            self.parameter_samples[self.name_to_index[hyperparameter]],
            bins=30,
        )

        # plot constrained posterior marginal
        if constrained:
            plt.hist(
                self.samples[self.name_to_index[hyperparameter]],
                bins=30,
            )
