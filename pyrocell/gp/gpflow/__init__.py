# Standard Library Imports
from typing import List, Optional, cast

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import ceil, float64, mean, std, sqrt
from numpy.typing import NDArray

# Internal Project Imports
from pyrocell.gp.gpflow.backend import (
    OU,
    GaussianProcess,
    NoiseModel,
    OUosc,
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
            self.time, self.bckgd, self.bckgd_length, self.M, verbose=verbose
        )

        # plot
        if "background" in plots:
            self.plot("background")

        # --- detrend --- #
        if verbose:
            print("\nDetrending and denoising cell data...")

        self.model_detrend: List[Optional[GaussianProcess]] = [None] * self.N
        self.y_detrend: List[Optional[NDArray[float64]]] = [None] * self.N
        self.noise_detrend: List[Optional[float64]] = [None] * self.N
        self.LLR_list: List[Optional[NDArray[float64]]] = [None] * self.N
        self.OU_LL: List[Optional[NDArray[float64]]] = [None] * self.N
        self.OUosc_LL: List[Optional[NDArray[float64]]] = [None] * self.N

        for i in range(self.N):
            X_curr = self.time[: self.y_length[i]]
            y_curr = self.y_all[: self.y_length[i], i, None]

            # centre and standardize, adjust noise, detrend
            y_curr = (y_curr - mean(y_curr)) / std(y_curr)
            noise = self.bckgd_std / std(y_curr)
            y_detrended, m = detrend(X_curr, y_curr, 7.0, verbose=verbose)

            self.model_detrend[i], self.y_detrend[i], self.noise_detrend[i] = (
                m,
                y_detrended,
                noise,
            )

        if "detrend" in plots:
            self.plot("detrend")

    def plot(self, target: str):
        """
        Plot the data

        :param str target: String or List of strings describing plot types
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
                # check properly fit
                if not isinstance(self.model_detrend[i], NoiseModel):
                    continue

                m = cast(NoiseModel, self.model_detrend[i])
                y_detrended = cast(NDArray[float64], self.y_detrend[i])

                # plot
                plt.subplot(dim, dim, i + 1)
                m.test_plot()
                plt.plot(self.time[: self.y_length[i]], y_detrended, label="Detrended")

                plt.title(f"Cell {i+1}")