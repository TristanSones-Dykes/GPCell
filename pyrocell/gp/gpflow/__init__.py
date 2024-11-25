# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt

# Direct Namespace Imports
from numpy import ceil, log, sqrt, std, max, array, linspace
from numpy.random import uniform

# Internal Project Imports
from pyrocell.gp.gpflow.models import OU, OUosc
from pyrocell.gp.gpflow.utils import background_noise, detrend, load_data, fit_models


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
        self.mean_std, self.bckgd_models = background_noise(
            self.X_bckgd, self.bckgd, 7.0
        )
        self.noise_list = [self.mean_std / std(y) for y in self.Y]

        # --- detrend data --- #
        self.Y_detrended, self.detrend_models = detrend(self.X, self.Y, 7.0)

        # --- fit OU and OU+Oscillator models --- #
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
            ou_gp = fit_models(
                [X for _ in range(K)], [Y for _ in range(K)], OU, ou_prior
            )
            OU_LL_list = [gp.log_posterior_density for gp in ou_gp]

            ouosc_gp = fit_models(
                [X for _ in range(K)], [Y for _ in range(K)], OUosc, ouosc_prior
            )
            OUosc_LL_list = [gp.log_posterior_density for gp in ouosc_gp]

            BIC_OUosc = -2 * max(OUosc_LL_list) + 3 * log(len(Y))
            BIC_OU = -2 * max(OU_LL_list) + 2 * log(len(Y))
            BICdiff = BIC_OU - BIC_OUosc

            return BICdiff

        self.BICdiff_list = []
        for cell in range(self.N):
            x_curr = self.X[cell]
            y_detrended = self.Y_detrended[cell]
            noise = self.noise_list[cell]

            BICdiff = fit_ou_ouosc(x_curr, y_detrended, noise, 10)
            self.BICdiff_list.append(BICdiff)

        # --- plots --- #
        for plot in plots:
            self.plot(plot)

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
            dim = int(
                ceil(sqrt(sum([1 for i in self.detrend_models if i is not None])))
            )
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i in range(self.N):
                m = self.detrend_models[i]
                y_detrended = self.Y_detrended[i]

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
        elif target == "BIC":
            fig = plt.figure(figsize=(12 / 2.54, 6 / 2.54))

            cutoff = 3
            print(
                "Number of cells counted as oscillatory (BIC method): {0}/{1}".format(
                    sum(array(self.BICdiff_list) > cutoff), len(self.BICdiff_list)
                )
            )

            plt.hist(self.BICdiff_list, bins=linspace(-20, 20, 40))  # type: ignore
            plt.plot([cutoff, cutoff], [0, 2], "r--")
            plt.xlabel("LLR")
            plt.ylabel("Frequency")
            plt.title("LLRs of experimental cells")
