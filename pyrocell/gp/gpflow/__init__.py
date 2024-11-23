# Standard Library Imports
from typing import List, Optional

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import ceil, float64, sqrt
from numpy.typing import NDArray

# Internal Project Imports
from pyrocell.gp.gpflow.utils import (
    background_noise,
    detrend,
    load_data,
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
            self.time, self.bckgd, self.bckgd_length, verbose=verbose
        )

        # plot
        if "background" in plots:
            self.plot("background")

        # --- detrend --- #
        if verbose:
            print("\nDetrending and denoising cell data...")

        self.y_detrend, self.model_detrend = detrend(
            self.time, self.y_all, self.y_length, 3, verbose=verbose
        )

        if "detrend" in plots:
            self.plot("detrend")

        self.noise_detrend: List[Optional[float64]] = [None] * self.N
        self.LLR_list: List[Optional[NDArray[float64]]] = [None] * self.N
        self.OU_LL: List[Optional[NDArray[float64]]] = [None] * self.N
        self.OUosc_LL: List[Optional[NDArray[float64]]] = [None] * self.N

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
                    self.time[: self.y_length[i]],
                    y_detrended,
                    label="Detrended",
                    color="orange",
                    linestyle="dashed",
                )

                plt.title(f"Cell {i+1}")

            plt.legend()
            plt.tight_layout()
