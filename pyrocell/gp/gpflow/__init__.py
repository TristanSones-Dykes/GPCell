# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import argmax, ceil, log, mean, sqrt, std, max, array, linspace, zeros
from numpy.random import uniform, multivariate_normal
from gpflow.kernels import White

# Internal Project Imports
from pyrocell.gp.gpflow.utils import (
    background_noise,
    detrend,
    load_data,
    fit_processes,
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

        self.allowed = set(["background", "detrend", "BIC"])

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
        self.mean_noise, self.bckgd_GPs = background_noise(
            self.X_bckgd, self.bckgd, 7.0
        )
        self.noise_list = [self.mean_noise / std(y) for y in self.Y]

        # --- detrend data --- #
        self.Y_detrended, self.detrend_GPs = detrend(self.X, self.Y, 7.0)

        self.plot("background")
        self.plot("detrend")

        """# --- fit OU and OU+Oscillator models --- #
        def fit_ou_ouosc(X, Y, noise, K):
            OU_LL_list, OUosc_LL_list = [[] for _ in range(2)]

            # define priors
            ou_prior = [
                {
                    "lengthscale": uniform(0.1, 2.0),
                    "variance": uniform(0.1, 2.0),
                    "likelihood_variance": noise**2,
                    "train_likelihood": False,
                }
                for _ in range(K)
            ]
            ouosc_prior = [
                {
                    "lengthscale": uniform(0.1, 2.0),
                    "variance": uniform(0.1, 2.0),
                    "lengthscale_cos": uniform(0.1, 4.0),
                    "likelihood_variance": noise**2,
                    "train_likelihood": False,
                    "train_osc_variance": False,
                }
                for _ in range(K)
            ]

            # fit models and extract posteriors
            ou_GPs = fit_single_process(
                [X for _ in range(K)], [Y for _ in range(K)], OU, ou_prior
            )
            OU_LL_list = [gp.log_posterior_density for gp in ou_GPs]
            GP_ou = ou_GPs[argmax(OU_LL_list)]

            ouosc_GPs = fit_single_process(
                [X for _ in range(K)], [Y for _ in range(K)], OUosc, ouosc_prior
            )
            OUosc_LL_list = [gp.log_posterior_density for gp in ouosc_GPs]
            GP_ouosc = ouosc_GPs[argmax(OUosc_LL_list)]

            LLR = 100 * 2 * (max(OUosc_LL_list) - max(OU_LL_list)) / len(Y)
            BIC_OUosc = -2 * max(OUosc_LL_list) + 3 * log(len(Y))
            BIC_OU = -2 * max(OU_LL_list) + 2 * log(len(Y))
            BIC_diff = BIC_OU - BIC_OUosc

            return LLR, BIC_diff, GP_ou, GP_ouosc

        self.LLRs, self.BIC_diffs, self.ou_GPs, self.ouosc_GPs = [[] for _ in range(4)]
        for i in range(self.N):
            x_curr = self.X[i]
            y_detrended = self.Y_detrended[i]
            noise = self.noise_list[i]

            LLR, BIC_diff, GP_ou, GP_ouosc = fit_ou_ouosc(
                x_curr, y_detrended, noise, 10
            )
            self.LLRs.append(LLR)
            self.BIC_diffs.append(BIC_diff)
            self.ou_GPs.append(GP_ou)
            self.ouosc_GPs.append(GP_ouosc)

        # --- plots --- #
        for plot in plots:
            self.plot(plot)
        return

        # --- classification using synthetic cells --- #
        K = 10
        self.synth_LLRs = []
        detrend_kernels = [GP_d.fit_gp.kernel for GP_d in self.detrend_GPs]
        OU_kernels = [GP_ou.fit_gp.kernel for GP_ou in self.ou_GPs]

        # for each cell, make K synthetic cells
        for i in range(self.N):
            X = self.X[i]
            noise = self.noise_list[i]

            # configure synthetic cell kernel
            k_se = detrend_kernels[i]
            k_ou = OU_kernels[i]
            k_white = White(variance=noise**2)
            k_synth = k_se + k_ou + k_white

            # generate and detrend synthetic cell
            synths = [
                multivariate_normal(zeros(len(X)), k_synth(X)).reshape(-1, 1)
                for _ in range(K)
            ]
            synths_detrended, synth_GPs = detrend([X for _ in range(K)], synths, 7.0)

            # fit OU and OU+Oscillator models
            for j in range(K):
                LLR, BIC_diff, GP_ou, GP_ouosc = fit_ou_ouosc(
                    X, synths_detrended[j], noise, 10
                )
                self.synth_LLRs.append(LLR)

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

        plt.tight_layout()"""

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
            # square grid of cells
            dim = int(ceil(sqrt(self.M)))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, (x_bckgd, y_bckgd, m) in enumerate(
                zip(self.X_bckgd, self.bckgd, self.bckgd_GPs)
            ):
                # evaluate model
                y_mean, y_var = m(self.X_bckgd[i])
                deviation = sqrt(m.Y_var) * 2
                y_bckgd_centred = y_bckgd - mean(y_bckgd)

                # plot
                plt.subplot(dim, dim, i + 1)
                plt.plot(x_bckgd, y_mean, zorder=1, c="k", label="Fit GP")
                plt.plot(x_bckgd, y_bckgd_centred, zorder=0, c="b", label="True Data")

                # plot uncertainty
                plt.plot(x_bckgd, y_mean + deviation, zorder=1, c="r")
                plt.plot(x_bckgd, y_mean - deviation, zorder=1, c="r")
                plt.title(f"Background {i+1}")

        elif target == "detrend":
            # square grid of cells
            dim = int(ceil(sqrt(self.N)))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, (x, y, y_detrended, m) in enumerate(
                zip(self.X, self.Y, self.Y_detrended, self.detrend_GPs)
            ):
                # standardise input, evaluate model
                y_standardised = (y - mean(y)) / std(y)
                y_trend = m(x)[0]

                # plot
                plt.subplot(dim, dim, i + 1)
                plt.plot(x, y_standardised, label="True Data", color="b")
                plt.plot(x, y_trend, label="Trend", color="k", linestyle="dashed")
                plt.plot(
                    x,
                    y_detrended,
                    label="Detrended",
                    color="orange",
                )

                plt.title(f"Cell {i+1}")

        elif target == "BIC":
            fig = plt.figure(figsize=(12 / 2.54, 6 / 2.54))

            cutoff = 3
            print(
                "Number of cells counted as oscillatory (BIC method): {0}/{1}".format(
                    sum(array(self.BIC_diffs) > cutoff), len(self.BIC_diffs)
                )
            )

            plt.hist(self.BIC_diffs, bins=linspace(-20, 20, 40))  # type: ignore
            plt.plot([cutoff, cutoff], [0, 2], "r--")
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of experimental cells")

        plt.legend()
        plt.tight_layout()
