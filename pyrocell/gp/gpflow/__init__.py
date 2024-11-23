# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
from gpflow import Parameter
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import array, ceil, sqrt, std
from numpy.random import uniform

# Internal Project Imports
from pyrocell.gp.gpflow.models import OU, OUosc
from pyrocell.gp.gpflow.utils import (
    background_noise,
    detrend,
    load_data,
    fit_models,
    fit_models_replicates,
    pad_zeros,
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
                self.X,
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
            self.X,
            self.bckgd,
            self.bckgd_length,
            self.M,
            self.y_all,
            self.y_length,
            self.N,
        ) = load_data(path)

    def run(self, *args, **kwargs):
        """
        Fit background noise and trend models, adjust data and fit OU and OU+Oscillator models

        Parameters
        ----------
        **kwargs
            Named arguments for the fitting process and administrative options
        """
        # default arguments
        default_kwargs = {"verbose": False, "plots": []}
        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        # unpack arguments
        verbose = kwargs["verbose"]
        plots = set(kwargs["plots"])

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

        # --- background noise --- #
        self.bckgd_std, self.bckgd_models = background_noise(
            self.X, self.bckgd, self.bckgd_length, verbose=verbose
        )
        self.noise_list = [
            self.bckgd_std / std(self.y_all[: self.y_length[i], i, None])
            for i in range(self.N)
        ]

        # plot
        if "background" in plots:
            self.plot("background")

        # --- detrend --- #
        if verbose:
            print("\nDetrending and denoising cell data...")

        self.y_detrend, self.model_detrend = detrend(
            self.X, self.y_all, self.y_length, 7.0, verbose=verbose
        )

        if "detrend" in plots:
            self.plot("detrend")

        # --- OU and OUosc --- #

        # define priors
        OU_priors = []
        OUosc_priors = []
        for i in range(self.N):
            OU_priors.append(
                lambda: {
                    "lengthscale": Parameter(uniform(0.1, 2.0)),
                    "variance": Parameter(uniform(0.1, 2.0)),
                    "likelihood_variance": Parameter(self.noise_list[i] ** 2),
                    "train_likelihood": False,
                }
            )
            OUosc_priors.append(
                lambda: {
                    "lengthscale": uniform(0.1, 2.0),
                    "variance": uniform(0.1, 2.0),
                    "lengthscale_cos": uniform(0.1, 4.0),
                    "train_likelihood": False,
                    "train_osc_variance": False,
                }
            )

        # fit models
        K = 10
        self.ou_models = fit_models_replicates(
            K,
            self.X,
            pad_zeros(self.y_detrend),
            self.y_length,
            OU,
            OU_priors,
            2,
            verbose,
        )

    def plot(self, target: str):
        """
        Plot the data

        Parameters
        ----------
        target : str
            Type of plot to generate
        """
        plot_size = int(15 / 5)
        if target == "background":
            # generate grid for background models
            dim = int(ceil(sqrt(self.M)))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, m in enumerate(self.bckgd_models):
                plt.subplot(dim, dim, i + 1)
                m.test_plot(plot_sd=True)
                plt.title(f"Background {i+1}")

            plt.legend()
            plt.tight_layout()
        elif target == "detrend":
            # square grid of cells
            dim = int(ceil(sqrt(sum([1 for i in self.model_detrend if i is not None]))))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i in range(self.N):
                m = self.model_detrend[i]
                y_detrended = self.y_detrend[i]

                # check properly fit
                if m is None or y_detrended is None:
                    continue

                # plot
                plt.subplot(dim, dim, i + 1)
                m.test_plot()  # detrended data
                plt.plot(
                    self.X[: self.y_length[i]],
                    y_detrended,
                    label="Detrended",
                    color="orange",
                    linestyle="dashed",
                )

                plt.title(f"Cell {i+1}")

            plt.legend()
            plt.tight_layout()
