# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import ceil, log, sqrt, std, max
from numpy.random import uniform

# Internal Project Imports
from pyrocell.gp.gpflow.models import OU, OUosc
from pyrocell.gp.gpflow.utils import (
    background_noise,
    detrend,
    load_data,
    fit_models_replicates,
)


class OscillatorDetector:
    def __init__(
        self,
        path: str,
        X_name: str,
        background_name: str,
        Y_name: str,
    ):
        """
        Initialize the Oscillator Detector
        If a path is provided, load data from the csv file

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
        self.X_bckgd, self.bckgd = load_data(path, X_name, background_name)
        self.X, self.Y = load_data(path, X_name, Y_name)
        self.N, self.M = len(self.Y), len(self.bckgd)

        self.allowed = set(["background", "detrend"])

    def __str__(self):
        # create a summary of the models and data
        out = f"Oscillator Detector with {self.N} cells and {self.M} background noise models\n"
        return out

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
            self.X_bckgd, self.bckgd, 7.0, verbose=verbose
        )
        self.noise_list = [self.bckgd_std / std(self.Y[i]) for i in range(self.N)]

        # plot
        if "background" in plots:
            self.plot("background")

        # --- detrend --- #
        if verbose:
            print("\nDetrending and denoising cell data...")

        self.y_detrend, self.model_detrend = detrend(
            self.X, self.Y, 7.0, verbose=verbose
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
                    "lengthscale": uniform(0.1, 2.0),
                    "variance": uniform(0.1, 2.0),
                    "likelihood_variance": self.noise_list[i] ** 2,
                    "train_likelihood": False,
                }
            )
            OUosc_priors.append(
                lambda: {
                    "lengthscale": uniform(0.1, 2.0),
                    "variance": uniform(0.1, 2.0),
                    "lengthscale_cos": uniform(0.1, 4.0),
                    "likelihood_variance": self.noise_list[i] ** 2,
                    "train_likelihood": False,
                    "train_osc_variance": False,
                }
            )

        # fit models
        K = 10
        self.ou_models = fit_models_replicates(
            K,
            self.X,
            self.y_detrend,
            OU,
            OU_priors,
            2,
            verbose,
        )
        self.ouosc_models = fit_models_replicates(
            K,
            self.X,
            self.y_detrend,
            OUosc,
            OUosc_priors,
            2,
            verbose,
        )

        # extract posterior log likelihoods
        self.ou_likelihoods = [
            [m.log_likelihood() for m in models] for models in self.ou_models
        ]
        self.ouosc_likelihoods = [
            [m.log_likelihood() for m in models] for models in self.ouosc_models
        ]
        self.LLRs = [
            200 * (max(ouosc) - max(ou))
            for ouosc, ou in zip(self.ouosc_likelihoods, self.ou_likelihoods)
        ]

        self.ou_BICs = [
            -2 * max(ou) + 2 * log(len(self.y_detrend[i]))
            for i, ou in enumerate(self.ou_likelihoods)
        ]
        self.ouosc_BICs = [
            -2 * max(ouosc) + 3 * log(len(self.y_detrend[i]))
            for i, ouosc in enumerate(self.ouosc_likelihoods)
        ]
        self.BIC_diffs = [
            ou - ouosc for ouosc, ou in zip(self.ouosc_BICs, self.ou_BICs)
        ]

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
                    self.X[i],
                    y_detrended,
                    label="Detrended",
                    color="orange",
                    linestyle="dashed",
                )

                plt.title(f"Cell {i+1}")

            plt.legend()
            plt.tight_layout()
