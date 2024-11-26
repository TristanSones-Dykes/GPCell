# Standard Library Imports
from math import ceil, sqrt
from typing import List, Optional, cast

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from pyro.nn.module import PyroSample
from pyro.distributions import Uniform
from pyro.infer import Trace_ELBO
from torch import Tensor, mean, no_grad, std, tensor

# Internal Project Imports
from pyrocell.gp import GaussianProcessBase
from pyrocell.gp.pyro.backend import (
    NoiseModel,
    load_data,
    background_noise,
    detrend,
    OU,
    OUosc,
)


class OscillatorDetector:
    def __init__(self, path: Optional[str] = None):
        """
        Initialize the Oscillator Detector
        If a path is provided, load data from the csv file

        Parameters
        ----------
        backend : str
            Backend library for the Gaussian Process models
        path : str | None
            Path to the csv file
        """

        if path is not None:
            (
                self.time,
                self.bckgd,
                self.bckgd_length,
                self.M,
                self.y_all,
                self.y_length,
                self.N,
            ) = load_data(path)
        self.allowed = set(["background", "detrend"])

    def __str__(self):
        # create a summary of the models and data
        out = f"Oscillator Detector with {self.N} cells and {self.M} background noise models\n"
        if hasattr(self, "bckgd_std"):
            # display overall std and fitted models
            out += f"\nBackground noise std: {self.bckgd_std}"
            out += f"\nBackground noise models: {self.bckgd_models}\n"

        if hasattr(self, "model_detrend"):
            # display detrended data and models
            out += f"\nDetrended noise models: {self.model_detrend}"

        return out

    def load_data(self, path: str):
        """
        Load data from a csv file


        Parameters
        ----------
        path : str
            Path to the csv file
        """
        (
            self.time,
            self.bckgd,
            self.bckgd_length,
            self.M,
            self.y_all,
            self.y_length,
            self.N,
        ) = load_data(path)

    def fit_models(self, *args, **kwargs):
        """
        Fit background noise and trend models, adjust data and fit OU and OU+Oscillator models

        Parameters
        ----------
        **kwargs
            Named arguments for the fitting process and administrative options
        """
        # default arguments
        default_kwargs = {"verbose": False, "plots": [], "jitter": 1.0e-5}
        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        # unpack arguments
        verbose = kwargs["verbose"]
        plots = set(kwargs["plots"])
        jitter = kwargs["jitter"]

        # check if plots are valid and print data
        if not plots.issubset(self.allowed):
            raise ValueError(f"Invalid plot type(s) selected: {plots - self.allowed}")
        if verbose:
            print(
                f"Loaded data with {self.N} cells and {self.M} background noise models"
            )
            print(f"Plots: {'on' if plots else 'off'}")
            print("\n")
            print("Fitting background noise...")

        # --- data preprocessing --- #

        # centre all data
        for i in range(self.N):
            y_curr = self.y_all[: self.y_length[i], i]
            y_curr -= mean(y_curr)
        for i in range(self.M):
            y_curr = self.bckgd[: self.bckgd_length[i], i]
            y_curr -= mean(y_curr)

        # --- background noise --- #
        self.bckgd_std, self.bckgd_models = background_noise(
            self.time, self.bckgd, self.bckgd_length, self.M, verbose=verbose
        )

        # plot and start next step
        if "background" in plots:
            self.plot("background")
        if verbose:
            print("\nDetrending and denoising cell data...")

        # --- detrend and denoise cell data --- #
        self.model_detrend: List[Optional[NoiseModel]] = [None] * self.N
        self.y_detrend: List[Optional[Tensor]] = [None] * self.N
        self.noise_detrend: List[Optional[Tensor]] = [None] * self.N
        self.LLR_list: List[Optional[Tensor]] = [None] * self.N
        self.OU_LL: List[Optional[Tensor]] = [None] * self.N
        self.OUosc_LL: List[Optional[Tensor]] = [None] * self.N

        ou_priors = {
            "lengthscale": PyroSample(Uniform(tensor(0.1), tensor(2.0))),
            "variance": PyroSample(Uniform(tensor(0.1), tensor(2.0))),
            # "lengthscale" : PyroParam(tensor(1.0), constraint=greater_than(0.0)),
            # "variance" : PyroParam(tensor(1.0), constraint=greater_than(0.0))
        }
        osc_priors = {
            "lengthscale": PyroSample(Uniform(tensor(0.1), tensor(4.0)))
            # "lengthscale" : PyroParam(tensor(1.0), constraint=greater_than(0.0))
        }

        # loop through and fit models
        for i in range(self.N):
            # reference input data
            X_curr = self.time[: self.y_length[i]]
            y_curr = self.y_all[: self.y_length[i], i]

            # normalise and reshape inplace
            noise = self.bckgd_std / std(y_curr)
            y_curr = y_curr / std(y_curr)

            # detrend
            res = detrend(X_curr, y_curr, 7.1, verbose=verbose)

            # skip if failed
            if res is None:
                continue
            y_detrended, noise_model = res

            # fit OU, OU+Oscillator
            ou = OU(ou_priors)
            success = ou.fit(
                X_curr,
                y_detrended,
                Trace_ELBO().differentiable_loss,
                verbose=verbose,
                jitter=jitter,
            )
            if success:
                self.OU_LL[i] = ou.log_posterior()

            ouosc = OUosc(ou_priors, osc_priors)
            success = ouosc.fit(
                X_curr,
                y_detrended,
                Trace_ELBO().differentiable_loss,
                verbose=verbose,
                jitter=jitter,
            )
            if success:
                self.OUosc_LL[i] = ouosc.log_posterior()

            self.y_detrend[i] = y_detrended
            self.model_detrend[i] = noise_model
            self.noise_detrend[i] = noise

        if "detrend" in plots:
            self.plot("detrend")

    def plot(self, target: str):
        """
        Plot the data

        Parameters
        ----------
        target : str
            String describing plot type
        """
        plot_size = 15 / 5
        if target == "background":
            # generate grid for background models
            dim = ceil(sqrt(self.M))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, m in enumerate(self.bckgd_models):
                plt.subplot(dim, dim, i + 1)
                m.test_plot(plot_sd=True)
                plt.title(f"Background {i+1}")

            plt.legend()
            plt.tight_layout()
        elif target == "detrend":
            # square grid of cells
            dim = ceil(sqrt(sum([1 for i in self.model_detrend if i is not None])))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i in range(self.N):
                # check properly fit
                if not isinstance(self.model_detrend[i], GaussianProcessBase):
                    continue

                m = cast(NoiseModel, self.model_detrend[i])
                y_detrended = cast(Tensor, self.y_detrend[i])

                # plot
                plt.subplot(dim, dim, i + 1)
                m.test_plot()
                with no_grad():
                    plt.plot(
                        self.time[: self.y_length[i]], y_detrended, label="Detrended"
                    )

                plt.title(f"Cell {i+1}")
            plt.legend()
