# Standard Library Imports
from typing import Iterable, List, Sequence, Tuple
from functools import partial
from collections.abc import Callable

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import (
    argmax,
    ceil,
    isnan,
    concatenate,
    log,
    mean,
    sqrt,
    # stack,
    std,
    array,
    linspace,
    zeros,
    max,
    argsort,
    zeros_like,
    float64,
)
from numpy.random import uniform, multivariate_normal
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from arviz import rhat, ess, convert_to_dataset, plot_trace

from gpflow.kernels import White, Matern12, Cosine, Kernel
# from gpflow.utilities import print_summary

# Internal Project Imports
from gpcell.backend import (
    GaussianProcess,
    GPPriorFactory,
    GPPriorTrainingFlag,
    Ndarray,
    hes_ou_prior,
    hes_ouosc_prior,
    ou_trainables,
    ouosc_trainables,
    ou_hyperparameters,
    ouosc_hyperparameters,
)
from .utils import (
    load_data,
    fit_processes,
    background_noise,
    detrend,
    fit_processes_joblib,
)


# Main class
class OscillatorDetector:
    def __init__(
        self, X: Sequence[Ndarray], Y: Sequence[Ndarray], N: int, *args, **kwargs
    ):
        # default arguments
        default_kwargs = {
            "verbose": False,
            "plots": [],
            "set_noise": None,
            "joblib": False,
            "cache": False,
            "ou_prior_gen": hes_ou_prior,
            "ouosc_prior_gen": hes_ouosc_prior,
            "ou_trainables": ou_trainables,
            "ouosc_trainables": ouosc_trainables,
        }
        print("\nStarting Oscillator Detector...\n")
        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
            else:
                print(f"Overriding default value for {key}")

        # unpack arguments
        self.verbose = kwargs["verbose"]
        self.plots = set(kwargs["plots"])
        self.set_noise = kwargs["set_noise"]
        self.joblib = kwargs["joblib"]
        self.ou_prior = kwargs["ou_prior_gen"]
        self.ouosc_prior = kwargs["ouosc_prior_gen"]
        self.ou_trainables = kwargs["ou_trainables"]
        self.ouosc_trainables = kwargs["ouosc_trainables"]

        # validate arguments
        if not all(
            [
                x
                in {
                    "background",
                    "detrend",
                    "BIC",
                    "LLR",
                    "periods",
                    "MCMC",
                    "MCMC_marginal",
                }
                for x in self.plots
            ]
        ):
            raise ValueError("Invalid plot type(s) selected")

        # store data
        self.X = X
        self.Y = Y
        self.N = N

        if self.set_noise is None:
            self.X_bckgd = kwargs["X_bckgd"]
            self.bckgd = kwargs["bckgd"]
            self.M = kwargs["M"]

        if self.verbose:
            if self.set_noise is None:
                print(
                    f"\nLoaded data with {self.N} cells and {self.M} background noise models"
                )
            else:
                print(
                    f"\nLoaded data with {self.N} cells, noise set to {self.set_noise}"
                )

            print(f"Plots: {'on' if self.plots else 'off'}")

        # preprocessing
        # --- background noise --- #
        if self.set_noise is None:
            print("\nFitting background noise...")
            self.mean_noise, self.bckgd_GPs = background_noise(
                self.X_bckgd, self.bckgd, 7.0, verbose=self.verbose
            )
            self.noise_list = [self.mean_noise / std(y) for y in self.Y]
        else:
            self.noise_list: List[float64] = [self.set_noise for _ in range(self.N)]

        # --- detrend data --- #
        if kwargs.get("detrend", True):
            self.Y_detrended, self.detrend_GPs = detrend(
                self.X, self.Y, 7.0, verbose=self.verbose
            )
        else:
            self.Y_detrended = self.Y

        # generate plots
        pre_plots = {"background", "detrend"}
        for plot in pre_plots.intersection(self.plots):
            self.generate_plot(plot)

    @classmethod
    def from_file(
        cls,
        path: str,
        X_name: str,
        background_name: str,
        Y_name: str,
        params: dict | None = None,
    ):
        """
        Initialize the Oscillator Detector from a csv file.

        Parameters
        ----------
        path : str
            Path to the csv file
        X_name : str
            Name of the column containing the time points
        background_name : str
            Name of the column containing the background noise models
        Y_name : str
            Name of the column containing the cell traces
        """
        # load data
        X, Y = load_data(path, X_name, Y_name)
        N = len(Y)

        # check for kwargs and use set noise if provided
        match params:
            case dict():
                if "set_noise" in params:
                    return cls(X, Y, N, **params)
            case _:
                params = {}

        X_bckgd, bckgd = load_data(path, X_name, background_name)
        M = len(bckgd)
        params["X_bckgd"], params["bckgd"], params["M"] = X_bckgd, bckgd, M

        return cls(X, Y, N, **params)

    def __str__(self):
        # create a summary of the models and data
        out = f"Oscillator Detector with {self.N} cells and {self.M} background noise models\n"
        return out

    def fit(self, methods: str | List[str] | None = None, *args, **kwargs):
        """
        Extensible fitting method that runs a list of fitting routines in order.
        The available methods are "BIC", "BOOTSTRAP", and "MCMC". If only a single method (or a list)
        is provided, the dependency graph is used to ensure that required methods (e.g., BIC for BOOTSTRAP)
        are run first.

        Parameters
        ----------
        methods : Union[str, List[str]], optional
            A single fitting method or list of methods to run (default is "BIC").
        kwargs : dict
            Additional keyword arguments to pass to the fitting
        """
        # Allow methods to be a single string or a list.
        if methods is None:
            methods = ["BIC"]
        elif isinstance(methods, str):
            methods = [methods]

        # Convert all method names to uppercase.
        methods = [m.upper() for m in methods]

        # Initialize persistent fit status if not already present.
        if not hasattr(self, "_fit_status"):
            self._fit_status = set()

        # Define dependency graph: each key maps to a list of methods that must run first.
        dependencies = {
            "BIC": [],
            "BOOTSTRAP": ["BIC"],
            "MCMC": [],
        }

        # Define prior generators
        match self.joblib:
            case True:
                # prior is either callable or list of callables
                match self.ou_prior:
                    case Callable():
                        ou_priors = [
                            partial(self.ou_prior, noise) for noise in self.noise_list
                        ]
                    case list():
                        ou_priors = [
                            partial(prior, noise)
                            for prior, noise in zip(self.ou_prior, self.noise_list)
                        ]
                    case _:
                        raise ValueError(
                            "ou_prior must be a callable or list of callables"
                        )

                match self.ouosc_prior:
                    case Callable():
                        ouosc_priors = [
                            partial(self.ouosc_prior, noise)
                            for noise in self.noise_list
                        ]
                    case list():
                        ouosc_priors = [
                            partial(prior, noise)
                            for prior, noise in zip(self.ouosc_prior, self.noise_list)
                        ]
                    case _:
                        raise ValueError(
                            "ouosc_prior must be a callable or list of callables"
                        )
            case False:
                ou_priors = [
                    lambda noise=noise: {
                        "kernel.lengthscales": uniform(0.1, 2.0),
                        "kernel.variance": uniform(0.1, 2.0),
                        "likelihood.variance": noise**2,
                    }
                    for noise in self.noise_list
                ]
                ouosc_priors = [
                    lambda noise=noise: {
                        "kernel.kernels[0].lengthscales": uniform(0.1, 2.0),
                        "kernel.kernels[0].variance": uniform(0.1, 2.0),
                        "kernel.kernels[1].lengthscales": uniform(0.1, 4.0),
                        "likelihood.variance": noise**2,
                    }
                    for noise in self.noise_list
                ]
            case _:
                raise ValueError("joblib must be a boolean")

        # Recursive helper function to run a method and its dependencies.
        def run_method(method: str):
            if method not in dependencies:
                raise ValueError(f"Unknown fitting method: {method}")
            # Run all dependencies first.
            for dep in dependencies[method]:
                if dep not in self._fit_status:
                    run_method(dep)
            # Now run the method if it hasn't been run yet.
            if method not in self._fit_status:
                if method == "BIC":
                    self._fit_BIC(
                        ou_priors,
                        ou_trainables,
                        ouosc_priors,
                        ouosc_trainables,
                        **kwargs,
                    )
                elif method == "BOOTSTRAP":
                    self._fit_bootstrap(
                        ou_priors,
                        ou_trainables,
                        ouosc_priors,
                        ouosc_trainables,
                        **kwargs,
                    )
                elif method == "MCMC":
                    self._fit_mcmc(
                        ou_priors,
                        ou_trainables,
                        ouosc_priors,
                        ouosc_trainables,
                        **kwargs,
                    )
                # Mark this method as completed.
                self._fit_status.add(method)

        # Run each requested method (which will trigger dependencies as needed).
        for method in methods:
            run_method(method)

    def _fit_BIC(
        self,
        ou_priors: GPPriorFactory | Sequence[GPPriorFactory],
        ou_trainables: GPPriorTrainingFlag,
        ouosc_priors: GPPriorFactory | Sequence[GPPriorFactory],
        ouosc_trainables: GPPriorTrainingFlag,
        *args,
        **kwargs,
    ):
        """
        Fit background noise and trend models, adjust data and fit OU and OU+Oscillator models
        """
        # --- classify using BIC --- #
        if self.verbose:
            print("\nFitting BIC...")

        # fit OU and OU+Oscillator models
        ou_GPs, ouosc_GPs = self._fit_ou_ouosc(
            self.X,
            self.Y_detrended,
            ou_priors,
            ou_trainables,
            ouosc_priors,
            ouosc_trainables,
            10,
        )

        # calculate LLR and BIC
        (
            self.LLRs,
            self.BIC_diffs,
            self.k_ou_list,
            self.k_ouosc_list,
            self.periods,
        ) = self._calc_gpr_bic_llr(self.X, self.Y_detrended, ou_GPs, ouosc_GPs)

        if self.verbose:
            print(
                "\nNumber of cells counted as oscillatory (BIC method): {0}/{1}".format(
                    sum(array(self.BIC_diffs) > 3), len(self.BIC_diffs)
                )
            )

        # generate plot
        if "BIC" in self.plots:
            self.generate_plot("BIC")

        return

    def _fit_bootstrap(
        self,
        ou_priors: Sequence[GPPriorFactory],
        ou_trainables: GPPriorTrainingFlag,
        ouosc_priors: Sequence[GPPriorFactory],
        ouosc_trainables: GPPriorTrainingFlag,
        *args,
        **kwargs,
    ):
        # --- classification using synthetic cells --- #
        if self.verbose:
            print(f"\nGenerating synthetic data for {self.N} cells...")

        # extract kwargs
        set_noise = kwargs.get("set_noise", None)

        K = 10
        self.synth_LLRs = []
        detrend_kernels = [GP_d.fit_gp.kernel for GP_d in self.detrend_GPs]

        # for each cell, make K synthetic cells
        for i in range(self.N):
            X = self.X[i]

            # set noise for automated testing
            if set_noise is not None:
                noise = set_noise
            else:
                noise = self.noise_list[i]

            # configure synthetic cell kernel
            k_se = detrend_kernels[i]
            k_ou = self.k_ou_list[i]
            k_white = White(variance=noise**2)  # type: ignore
            k_synth = k_se + k_ou + k_white

            # generate and detrend synthetic cell
            synths = [
                multivariate_normal(zeros(len(X)), k_synth(X)).reshape(-1, 1)  # type: ignore
                for _ in range(K)
            ]
            synths_detrended, _ = detrend([X for _ in range(K)], synths, 7.0)

            # fit OU and OU+Oscillator models
            ou_GPs, ouosc_GPs = self._fit_ou_ouosc(
                [X for _ in range(K)],
                synths_detrended,
                ou_priors[i],
                ou_trainables,
                ouosc_priors[i],
                ouosc_trainables,
                K,
            )

            # calculate LLR
            LLRs, _, _, _, _ = self._calc_gpr_bic_llr(
                [X for _ in range(K)], synths_detrended, ou_GPs, ouosc_GPs
            )

            self.synth_LLRs.extend(LLRs)

        if "LLR" in self.plots:
            self.generate_plot("LLR")

        # --- classification --- #

        if self.verbose:
            print("Classifying cells...")

        LLR_array = array(self.LLRs)
        LLR_synth_array = array(self.synth_LLRs)

        # filter nan's from model fitting
        nan_inds = isnan(LLR_array)
        LLR_array = LLR_array[~nan_inds]
        LLR_synth_array = LLR_synth_array[~isnan(LLR_synth_array)]

        # LLRs can be tiny and just negative - this just sets them to zero
        LLR_array[LLR_array < 0] = 0
        LLR_synth_array[LLR_synth_array < 0] = 0
        LLR_combined = concatenate((LLR_array, LLR_synth_array), 0)
        self.periods = array(self.periods)[~nan_inds]

        upper = max(LLR_combined)
        lower1 = min(LLR_combined)
        lower = upper - 0.9 * (upper - lower1)
        grid = linspace(lower, upper, 20)

        piest = zeros_like(grid)

        for i, cutoff in enumerate(grid):
            num = sum(LLR_array < cutoff) / len(LLR_array)
            denom = sum(LLR_synth_array < cutoff) / len(LLR_synth_array)
            piest[i] = num / denom

        xx = linspace(lower, upper, 100)
        cs = CubicSpline(grid, piest)
        yy = cs(xx)
        piGUESS1 = yy[0]

        # retain indices to restore order
        LLR_Idx = argsort(LLR_array)
        LLR_array_sorted = LLR_array[LLR_Idx]

        q1 = zeros_like(LLR_array_sorted)
        for i, thresh in enumerate(LLR_array_sorted):
            q1[i] = (
                piGUESS1
                * (sum(LLR_synth_array >= thresh) / len(LLR_synth_array))
                / (sum(LLR_array_sorted >= thresh) / len(LLR_array_sorted))
            )

        q_vals = q1[argsort(LLR_Idx)]
        self.osc_filt = q_vals < 0.05

        if self.verbose:
            print(
                "Number of cells counted as oscillatory (full method): {0}/{1}".format(
                    sum(self.osc_filt), len(self.osc_filt)
                )
            )

        # generate plots
        if "LLR" in self.plots:
            self.generate_plot("LLR")
        if "periods" in self.plots:
            self.generate_plot("periods")

    def _fit_mcmc(
        self,
        ou_priors: Sequence[GPPriorFactory],
        ou_trainables: GPPriorTrainingFlag,
        ouosc_priors: Sequence[GPPriorFactory],
        ouosc_trainables: GPPriorTrainingFlag,
    ):
        # --- fit GPs using MCMC and categorise --- #
        if self.verbose:
            print("\nFitting MCMC...")

        ouosc_kernel = [Matern12, Cosine]

        # fit OU+Oscillator models
        self.ouosc_GPs = fit_processes_joblib(
            self.X,
            self.Y_detrended,
            ouosc_kernel,
            ouosc_priors,
            replicates=3,
            trainable=ouosc_trainables,
            mcmc=True,
            verbose=self.verbose,
        )

        # pool and check convergence
        self.beta_samples = []
        for i, gp_list in enumerate(self.ouosc_GPs):
            # extract list of sample objects for cell
            sample_list = [gp.samples for gp in gp_list]

            # get beta samples
            beta_idx_list = [
                gp.name_to_index[".kernel.kernels[1].lengthscales"] for gp in gp_list
            ]
            sample_list = array(
                [gp.samples[beta_idx] for gp, beta_idx in zip(gp_list, beta_idx_list)]
            )

            # calculate r_hat and ess for unconstrained space
            r_hat, ess_val = rhat(sample_list), ess(sample_list)
            print(f"Unconstrained {i + 1} r_hat: {r_hat}, ess: {ess_val}")
            self.beta_samples.append(concatenate(sample_list, axis=0))

            # # calculate r_hat and ess for constrained space
            # sample_list = array(
            #     gp.parameter_samples[beta_idx]
            #     for gp, beta_idx in zip(gp_list, beta_idx_list)
            # )
            # r_hat, ess_val = rhat(sample_list), ess(sample_list)
            # print(f"Constrained {i + 1} r_hat: {r_hat}, ess: {ess_val}")

            # check convergence
            # if r_hat < 1.1 and ess_val > 200:
            #     self.pooled_samples.append(
            #         concatenate(sample_list, axis=0).reshape(-1, K)
            #     )
            # else:
            #     print(
            #         f"Cell {i + 1} has not converged (r_hat: {r_hat}, ess: {ess_val})"
            #     )
            #     self.pooled_samples.append(zeros((K, 1)))

        print(f"Shape of beta samples: {self.beta_samples[0].shape}")

        # calculate savage-dickey
        self.bf_list = []
        for i, gp_list in enumerate(self.ouosc_GPs):
            # prior distribution
            prior_dict = gp_list[0].constructor.prior_gen()
            prior_dist = prior_dict["kernel.kernels[1].lengthscales.prior"]

            # posterior samples
            posterior_samples = self.beta_samples[i]

            # compute prior and posterior mass
            eps = 0.2
            prior_mass = float(prior_dist.cdf(eps) - prior_dist.cdf(0.0))  # type: ignore
            posterior_mass = float(mean(posterior_samples < eps))
            print(
                f"Cell {i + 1} prior mass: {prior_mass}, posterior mass: {posterior_mass}"
            )
            # compute Bayes factor
            bf = posterior_mass / prior_mass

            # # compute prior and posterior density
            # prior_density = float(prior_dist.prob(0.0))
            # kde = gaussian_kde(posterior_samples)
            # posterior_density = float(kde.evaluate(0.0))
            # print(
            #     f"Cell {i + 1} prior density: {prior_density}, posterior density: {posterior_density}"
            # )
            # # compute Bayes factor
            # bf = posterior_density / prior_density

            self.bf_list.append(bf)

        # print Bayes factors
        for i, bf in enumerate(self.bf_list):
            print(f"Cell {i + 1} Bayes factor: {bf}")

        # plot
        if "MCMC" in self.plots:
            self.generate_plot("MCMC")
        if "MCMC_marginal" in self.plots:
            self.generate_plot("MCMC_marginal")

    def _fit_ou_ouosc(
        self,
        X: Sequence[Ndarray],
        Y: Sequence[Ndarray],
        ou_priors: GPPriorFactory | Sequence[GPPriorFactory],
        ou_trainables: GPPriorTrainingFlag,
        ouosc_priors: GPPriorFactory | Sequence[GPPriorFactory],
        ouosc_trainables: GPPriorTrainingFlag,
        K: int,
        mcmc: bool = False,
    ):
        ou_kernel = Matern12
        ouosc_kernel = [Matern12, Cosine]

        # fit processes
        match self.joblib:
            case True:
                ou_GPs = fit_processes_joblib(
                    X,
                    Y,
                    ou_kernel,
                    ou_priors,
                    replicates=K,
                    trainable=ou_trainables,
                    mcmc=mcmc,
                    verbose=self.verbose,
                )
                ouosc_GPs = fit_processes_joblib(
                    X,
                    Y,
                    ouosc_kernel,
                    ouosc_priors,
                    replicates=K,
                    trainable=ouosc_trainables,
                    mcmc=mcmc,
                    verbose=self.verbose,
                )
            case False:
                ou_GPs = fit_processes(
                    X,
                    Y,
                    ou_kernel,
                    ou_priors,
                    replicates=K,
                    trainable=ou_trainables,
                    mcmc=mcmc,
                )
                ouosc_GPs = fit_processes(
                    X,
                    Y,
                    ouosc_kernel,
                    ouosc_priors,
                    replicates=K,
                    trainable=ouosc_trainables,
                    mcmc=mcmc,
                )
            case _:
                raise ValueError("joblib must be a boolean")

        return ou_GPs, ouosc_GPs

    def _calc_gpr_bic_llr(
        self,
        X: Sequence[Ndarray],
        Y: Sequence[Ndarray],
        ou_GPs: List[List[GaussianProcess]] | Iterable[List[GaussianProcess]],
        ouosc_GPs: List[List[GaussianProcess]] | Iterable[List[GaussianProcess]],
    ) -> Tuple[List[float], List[float], List[Kernel], List[Kernel], List]:
        """
        Calculate the LLR and BIC for each cell

        Parameters
        ----------
        X : Sequence[Ndarray]
            Input domain
        Y : Sequence[Ndarray]
            Target values
        ou_GPs : List[List[GaussianProcess]]
            List of OU GPs
        ouosc_GPs : List[List[GaussianProcess]]
            List of OU+Oscillator GPs
        """
        # calculate LLR and BIC
        LLRs = []
        BIC_diffs = []
        k_max_ou_list = []
        k_max_ouosc_list = []
        periods = []

        for i, (x, y, ou, ouosc) in enumerate(zip(X, Y, ou_GPs, ouosc_GPs)):
            ou_LL = [gp.log_posterior() for gp in ou]
            ouosc_LL = [gp.log_posterior() for gp in ouosc]

            # filter nan's from model fitting
            ou_nan, ouosc_nan = isnan(ou_LL), isnan(ouosc_LL)
            if ou_nan.any() or ouosc_nan.any():
                print(f"Cell {i + 1} has NaNs in the model fitting")

                ou_all, ouosc_all = ou_nan.all(), ouosc_nan.all()
                if ou_all or ouosc_all:
                    # select fitting function
                    match self.joblib:
                        case True:
                            f = fit_processes_joblib
                        case False:
                            f = fit_processes

                    attempts = 0
                    # refit model until no NaNs
                    while ou_all or ouosc_all:
                        attempts += 1
                        # fit 10 replicates of each model
                        if ou_all:
                            ou = next(
                                iter(
                                    f(
                                        [x],
                                        [y],
                                        Matern12,
                                        [
                                            lambda noise=self.noise_list[
                                                i
                                            ]: self.ou_prior(noise)
                                        ],
                                        replicates=10,
                                        trainable=self.ou_trainables,
                                    )
                                )
                            )

                        if ouosc_all:
                            ouosc = next(
                                iter(
                                    f(
                                        [x],
                                        [y],
                                        [Matern12, Cosine],
                                        [
                                            lambda noise=self.noise_list[
                                                i
                                            ]: self.ouosc_prior(noise)
                                        ],
                                        replicates=10,
                                        trainable=self.ouosc_trainables,
                                    )
                                )
                            )

                        # extract and test for NaNs
                        if ou_all:
                            ou_LL = [gp.log_posterior() for gp in ou]
                            ou_nan = isnan(ou_LL)
                        if ouosc_all:
                            ouosc_LL = [gp.log_posterior() for gp in ouosc]
                            ouosc_nan = isnan(ouosc_LL)
                        ou_all, ouosc_all = ou_nan.all(), ouosc_nan.all()

                        # if reached 5 attempts, print and break
                        if (ou_all or ouosc_all) and attempts >= 5:
                            print(
                                f"Cell {i + 1} has NaNs in the model fitting after {attempts} attempts"
                            )
                            ou_LL = zeros(10)
                            ou_nan = isnan(ou_LL)
                            ouosc_LL = zeros(10)
                            ouosc_nan = isnan(ouosc_LL)
                            break

            ou_LL = array(ou_LL)[~ou_nan]
            ouosc_LL = array(ouosc_LL)[~ouosc_nan]

            # take process with highest posterior density
            max_ou_ll = max(ou_LL)
            max_ouosc_ll = max(ouosc_LL)
            k_max_ou = ou[argmax(ou_LL)].fit_gp.kernel
            k_max_ouosc = ouosc[argmax(ouosc_LL)].fit_gp.kernel
            k_ouosc = ouosc[0].fit_gp.kernel

            # print_summary(ouosc[argmax(ouosc_LL)].fit_gp, fmt="notebook")

            # calculate LLR and BIC
            LLR = 100 * 2 * (max_ouosc_ll - max_ou_ll) / len(y)
            BIC_OUosc = -2 * max_ouosc_ll + 3 * log(len(y))
            BIC_OU = -2 * max_ou_ll + 2 * log(len(y))
            BIC_diff = BIC_OU - BIC_OUosc

            # BIC_OUosc = -2 * max_ouosc_ll / len(x)
            # BIC_OU = -2 * max_ou_ll / len(x)
            # BIC_diff = BIC_OU - BIC_OUosc
            # BIC_diff *= 100  # un-normalising

            # calculate period
            cov_ou_osc = k_ouosc(x).numpy()[0, :]  # type: ignore
            peaks, _ = find_peaks(cov_ou_osc, height=0)

            if len(peaks) != 0:
                period = x[peaks[0]]
            else:
                period = 0

            LLRs.append(LLR)
            BIC_diffs.append(BIC_diff)
            k_max_ou_list.append(k_max_ou)
            k_max_ouosc_list.append(k_max_ouosc)
            periods.append(period)

        return LLRs, BIC_diffs, k_max_ou_list, k_max_ouosc_list, periods

    def generate_plot(self, target: str):
        """
        Plot the data and save to class fields

        Parameters
        ----------
        target : str
            Type of plot to generate
        """

        # Dictionary to hold the plot attributes for the class
        plot_attributes = {
            "background": "background_plot",
            "detrend": "detrend_plot",
            "BIC": "bic_plot",
            "LLR": "llr_plot",
            "periods": "periods_plot",
            "MCMC": "mcmc_plot",
            "MCMC_marginal": "marginal_plot",
        }

        if target not in plot_attributes:
            raise ValueError(f"Unknown target: {target}")

        plot_size = int(15 / 5)
        fig = None  # Initialize the figure

        if target == "background":
            M = min(self.M, 12)
            dim = int(ceil(sqrt(M)))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, (x_bckgd, y_bckgd, m) in enumerate(
                zip(self.X_bckgd, self.bckgd, self.bckgd_GPs)
            ):
                if i == M:
                    break

                y_mean, y_var = m(self.X_bckgd[i])
                deviation = sqrt(m.Y_var) * 2
                y_bckgd_centred = y_bckgd - mean(y_bckgd)

                plt.subplot(dim, dim, i + 1)
                plt.plot(x_bckgd, y_mean, zorder=1, c="k", label="Fit GP")
                plt.plot(x_bckgd, y_bckgd_centred, zorder=0, c="b", label="True Data")
                plt.plot(x_bckgd, y_mean + deviation, zorder=1, c="r")
                plt.plot(x_bckgd, y_mean - deviation, zorder=1, c="r")
                plt.title(f"Background {i + 1}")

        elif target == "detrend":
            row = min(self.N, 12)
            dim = int(ceil(sqrt(row)))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, (x, y, y_detrended, m) in enumerate(
                zip(self.X, self.Y, self.Y_detrended, self.detrend_GPs)
            ):
                if i == row:
                    break

                y_standardised = (y - mean(y)) / std(y)
                y_trend = m(x)[0]

                plt.subplot(dim, dim, i + 1)
                plt.plot(x, y_detrended, label="Detrended")
                plt.plot(x, y_standardised, label="True Data")
                plt.plot(
                    x, y_trend, label="Trend", color="k", alpha=0.5, linestyle="dashed"
                )
                plt.title(f"Cell {i + 1}")

        elif target == "BIC":
            fig = plt.figure(figsize=(12 / 2.54, 6 / 2.54))

            cutoff = 3
            plt.hist(self.BIC_diffs, bins=linspace(-20, 20, 40), label="BIC")  # type: ignore
            plt.plot([cutoff, cutoff], [0, 2], "r--", label="Cutoff")
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of experimental cells")

        elif target == "LLR":
            fig = plt.figure(figsize=(20 / 2.54, 10 / 2.54))

            plt.subplot(1, 2, 1)
            plt.hist(self.LLRs, bins=linspace(0, 40, 40))  # type: ignore
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of experimental cells")

            plt.subplot(1, 2, 2)
            plt.hist(self.synth_LLRs, bins=linspace(0, 40, 40))  # type: ignore
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of synthetic non-oscillatory OU cells")

        elif target == "MCMC":
            # deal with sometimes not fitting OU GPs
            row, col = min(self.N, 12), 2
            not_ou = False
            if not hasattr(self, "ou_GPs"):
                ou_GPs = [0 for _ in range(row)]
                not_ou, row = True, row // 2
            else:
                ou_GPs = self.ou_GPs  # type: ignore

            print(
                f"Number of cells: {self.N}, Number of rows: {row}, Number of cols: {col}"
            )

            fig = plt.figure(figsize=(2.5 * plot_size * col, plot_size * row))
            param = ouosc_hyperparameters[-1]

            for i, (x, y, ou, ouosc) in enumerate(
                zip(self.X, self.Y_detrended, ou_GPs, self.ouosc_GPs)
            ):
                # if only fit OU+Oscillator, plot in both columns
                if not_ou:
                    plt.subplot(row, col, i + 1)
                    plt.title(f"Cell {i + 1}")
                    for gp in ouosc:
                        gp.plot_samples(param)
                    continue

                plt.subplot(row, col, 2 * i + 1)
                ou[0].plot_samples(ou_hyperparameters)  # type: ignore
                plt.title(f"OU cell {i + 1}")

                plt.subplot(row, col, 2 * i + 2)
                ouosc[0].plot_samples(ouosc_hyperparameters)
                plt.title(f"OU+Oscillator cell {i + 1}")

        elif target == "MCMC_marginal":
            # plot the marginal posterior distributions of the OU+Oscillator model
            row, col = min(self.N, 12), 3
            fig = plt.figure(figsize=(2 * plot_size * col, plot_size * row))

            for i, ouosc in enumerate(self.ouosc_GPs):
                # select first subplot of row, add ylabel
                plt.subplot(row, col, 3 * i + 1)
                plt.ylabel(f"Cell {i + 1} Count")

                for j in range(3):
                    plt.subplot(row, col, 3 * i + j + 1)

                    # if first iteration, add title
                    if i == 0:
                        plt.title(f"{ouosc_hyperparameters[j]}")
                    # if last iteration, add xlabel
                    if i == row - 1:
                        plt.xlabel(f"{ouosc_hyperparameters[j]}")

                    # plot marginal posteriors
                    ouosc[0].plot_posterior_marginal(ouosc_hyperparameters[j])
                    ouosc[0].plot_posterior_marginal(ouosc_hyperparameters[j], True)

        elif target == "periods":
            if self.periods is None:
                raise ValueError("Periods have not been calculated yet")

            fig = plt.figure(figsize=(12 / 2.54, 6 / 2.54))

            plt.hist(self.periods[self.osc_filt], bins=linspace(0, 10, 20))  # type: ignore
            plt.title("Periods of passing cells")
            plt.xlabel("Period (hours)")
            plt.ylabel("Frequency")

        plt.legend()
        plt.tight_layout()

        # Save figure to the corresponding class field
        field_name = plot_attributes[target]
        setattr(self, field_name, fig)

        # Close the plot to avoid displaying it immediately
        plt.close(fig)
