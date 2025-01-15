# Standard Library Imports
from typing import Iterable, List, Tuple, cast

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import (
    argmax,
    ceil,
    isnan,
    log,
    concatenate,
    mean,
    sqrt,
    std,
    array,
    linspace,
    zeros,
    max,
    argsort,
    zeros_like,
)
from numpy.random import uniform, multivariate_normal
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

from gpflow.kernels import White, Matern12, Cosine, Kernel

# Internal Project Imports
from gpcell.backend import GaussianProcess
from .utils import load_data, fit_processes, background_noise, detrend


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

        if verbose:
            print(
                f"Loaded data with {self.N} cells and {self.M} background noise models"
            )
            print(f"Plots: {'on' if plots else 'off'}")
            print("\n")
            print("Fitting background noise...")

        # --- background noise --- #
        self.mean_noise, self.bckgd_GPs = background_noise(
            self.X_bckgd, self.bckgd, 7.0, verbose=verbose
        )
        self.noise_list = [self.mean_noise / std(y) for y in self.Y]

        self.generate_plot("background")

        # --- detrend data --- #
        self.Y_detrended, self.detrend_GPs = detrend(
            self.X, self.Y, 7.0, verbose=verbose
        )

        self.generate_plot("detrend")

        # --- fit OU and OU*Oscillator processes --- #

        # define priors
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

        # set trainables
        ou_trainables = {"likelihood.variance": False}
        ouosc_trainables = {
            "likelihood.variance": False,
            (1, "variance"): False,
        }

        def fit_ou_ouosc(
            X, Y, ou_priors, ou_trainables, ouosc_priors, ouosc_trainables, K
        ) -> Tuple[List[float], List[float], List[Kernel], List[Kernel], List[float]]:
            ou_kernel = Matern12
            ouosc_kernel = [Matern12, Cosine]

            # fit processes
            ou_GPs = cast(
                Iterable[List[GaussianProcess]],
                fit_processes(
                    X,
                    Y,
                    ou_kernel,
                    ou_priors,
                    replicates=K,
                    trainable=ou_trainables,
                ),
            )
            ouosc_GPs = cast(
                Iterable[List[GaussianProcess]],
                fit_processes(
                    X,
                    Y,
                    ouosc_kernel,
                    ouosc_priors,
                    replicates=K,
                    trainable=ouosc_trainables,
                ),
            )

            # calculate LLR and BIC
            LLRs = []
            BIC_diffs = []
            k_max_ou_list = []
            k_max_ouosc_list = []
            periods = []

            for i, (x, y, ou, ouosc) in enumerate(
                zip(self.X, self.Y_detrended, ou_GPs, ouosc_GPs)
            ):
                ou_LL = [gp.log_posterior() for gp in ou]
                ouosc_LL = [gp.log_posterior() for gp in ouosc]

                # take process with highest posterior density
                max_ou_ll = max(ou_LL)
                max_ouosc_ll = max(ouosc_LL)
                k_max_ou = ou[argmax(ou_LL)].fit_gp.kernel
                k_max_ouosc = ouosc[argmax(ouosc_LL)].fit_gp.kernel
                k_ouosc = ouosc[0].fit_gp.kernel

                # calculate LLR and BIC
                LLR = 100 * 2 * (max_ouosc_ll - max_ou_ll) / len(y)
                BIC_OUosc = -2 * max_ouosc_ll + 3 * log(len(y))
                BIC_OU = -2 * max_ou_ll + 2 * log(len(y))
                BIC_diff = BIC_OU - BIC_OUosc

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

        self.LLRs, self.BIC_diffs, self.k_ou_list, self.k_ouosc_list, self.periods = (
            fit_ou_ouosc(
                self.X,
                self.Y_detrended,
                ou_priors,
                ou_trainables,
                ouosc_priors,
                ouosc_trainables,
                10,
            )
        )

        self.generate_plot("BIC")

        return
        # --- classification using synthetic cells --- #
        K = 10
        self.synth_LLRs = []
        detrend_kernels = [GP_d.fit_gp.kernel for GP_d in self.detrend_GPs]

        # for each cell, make K synthetic cells
        for i in range(self.N):
            X = self.X[i]
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
            LLRs, _, _, _, _ = fit_ou_ouosc(
                [X for _ in range(K)],
                synths_detrended,
                ou_priors[i],
                ou_trainables,
                ouosc_priors[i],
                ouosc_trainables,
                K,
            )

            self.synth_LLRs.extend(LLRs)

        if plots and "LLR" in plots:
            self.plot("LLR")

        # --- classification --- #

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

        if verbose:
            print(
                "Number of cells counted as oscillatory (full method): {0}/{1}".format(
                    sum(self.osc_filt), len(self.osc_filt)
                )
            )

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
        }

        if target not in plot_attributes:
            raise ValueError(f"Unknown target: {target}")

        plot_size = int(15 / 5)
        fig = None  # Initialize the figure

        if target == "background":
            dim = int(ceil(sqrt(self.M)))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, (x_bckgd, y_bckgd, m) in enumerate(
                zip(self.X_bckgd, self.bckgd, self.bckgd_GPs)
            ):
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
            dim = int(ceil(sqrt(self.N)))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, (x, y, y_detrended, m) in enumerate(
                zip(self.X, self.Y, self.Y_detrended, self.detrend_GPs)
            ):
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
            print(
                "Number of cells counted as oscillatory (BIC method): {0}/{1}".format(
                    sum(array(self.BIC_diffs) > cutoff), len(self.BIC_diffs)
                )
            )

            plt.hist(self.BIC_diffs, bins=linspace(-20, 20, 40), label="BIC")
            plt.plot([cutoff, cutoff], [0, 2], "r--", label="Cutoff")
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of experimental cells")

        elif target == "LLR":
            fig = plt.figure(figsize=(20 / 2.54, 10 / 2.54))

            plt.subplot(1, 2, 1)
            plt.hist(self.LLRs, bins=linspace(0, 40, 40))
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of experimental cells")

            plt.subplot(1, 2, 2)
            plt.hist(self.synth_LLRs, bins=linspace(0, 40, 40))
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of synthetic non-oscillatory OU cells")

        elif target == "periods":
            fig = plt.figure(figsize=(12 / 2.54, 6 / 2.54))

            plt.hist(self.periods[self.osc_filt], bins=linspace(0, 10, 20))
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
