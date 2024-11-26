# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import argmax, ceil, log, mean, sqrt, std, array, linspace, zeros, max
from numpy.random import uniform, multivariate_normal
from gpflow.kernels import White, Matern12, Cosine

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
            self.X_bckgd, self.bckgd, 7.0, verbose=verbose
        )
        self.noise_list = [self.mean_noise / std(y) for y in self.Y]

        # --- detrend data --- #
        self.Y_detrended, self.detrend_GPs = detrend(
            self.X, self.Y, 7.0, verbose=verbose
        )

        self.plot("background")
        self.plot("detrend")

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
        ):
            ou_kernel = Matern12
            ouosc_kernel = [Matern12, Cosine]

            # fit processes
            ou_GPs = fit_processes(
                X,
                Y,
                ou_kernel,
                ou_priors,
                replicates=K,
                trainable=ou_trainables,
            )
            ouosc_GPs = fit_processes(
                X,
                Y,
                ouosc_kernel,
                ouosc_priors,
                replicates=K,
                trainable=ouosc_trainables,
            )

            # calculate LLR and BIC
            LLRs, BIC_diffs, max_ou_GPs, max_ouosc_GPs = [[] for _ in range(4)]
            for i, (y, ou, ouosc) in enumerate(
                zip(self.Y_detrended, ou_GPs, ouosc_GPs)
            ):
                ou_LL = [gp.log_posterior_density for gp in ou]  # type: ignore
                ouosc_LL = [gp.log_posterior_density for gp in ouosc]  # type: ignore

                max_ou_ll = max(ou_LL)
                max_ouosc_ll = max(ouosc_LL)
                max_ou = ou[argmax(ou_LL)]  # type: ignore
                max_ouosc = ouosc[argmax(ouosc_LL)]  # type: ignore

                LLR = 100 * 2 * (max_ouosc_ll - max_ou_ll) / len(y)
                BIC_OUosc = -2 * max_ouosc_ll + 3 * log(len(y))
                BIC_OU = -2 * max_ou_ll + 2 * log(len(y))
                BIC_diff = BIC_OU - BIC_OUosc

                LLRs.append(LLR)
                BIC_diffs.append(BIC_diff)
                max_ou_GPs.append(max_ou)
                max_ouosc_GPs.append(max_ouosc)

            return LLRs, BIC_diffs, max_ou_GPs, max_ouosc_GPs

        self.LLRs, self.BIC_diffs, self.ou_GPs, self.ouosc_GPs = fit_ou_ouosc(
            self.X,
            self.Y_detrended,
            ou_priors,
            ou_trainables,
            ouosc_priors,
            ouosc_trainables,
            10,
        )

        # --- plots --- #
        self.plot("BIC")

        # --- classification using synthetic cells --- #
        K = 10
        self.synth_LLRs = []
        detrend_kernels = [GP_d.fit_gp.kernel for GP_d in self.detrend_GPs]
        OU_kernels = [GP_ou.fit_gp.kernel for GP_ou in self.ou_GPs]

        # for each cell, make K synthetic cells
        for i in range(self.N):
            print(i)
            X = self.X[i]
            noise = self.noise_list[i]

            # configure synthetic cell kernel
            k_se = detrend_kernels[i]
            k_ou = OU_kernels[i]
            k_white = White(variance=noise**2)  # type: ignore
            k_synth = k_se + k_ou + k_white

            # generate and detrend synthetic cell
            synths = [
                multivariate_normal(zeros(len(X)), k_synth(X)).reshape(-1, 1)
                for _ in range(K)
            ]
            synths_detrended, synth_GPs = detrend([X for _ in range(K)], synths, 7.0)

            # fit OU and OU+Oscillator models
            LLRs, _, _, _ = fit_ou_ouosc(
                [X for _ in range(K)],
                synths_detrended,
                [ou_priors[i] for _ in range(K)],
                ou_trainables,
                [ouosc_priors[i] for _ in range(K)],
                ouosc_trainables,
                K,
            )

            self.synth_LLRs.extend(LLRs)

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

        plt.tight_layout()

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
                plt.plot(
                    x,
                    y_detrended,
                    label="Detrended",
                )
                plt.plot(x, y_standardised, label="True Data")
                plt.plot(
                    x, y_trend, label="Trend", color="k", alpha=0.5, linestyle="dashed"
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

            plt.hist(self.BIC_diffs, bins=linspace(-20, 20, 40), label="BIC")  # type: ignore
            plt.plot([cutoff, cutoff], [0, 2], "r--", label="Cutoff")
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of experimental cells")

        plt.legend()
        plt.tight_layout()
