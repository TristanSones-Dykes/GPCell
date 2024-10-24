# Standard library imports
from math import ceil, sqrt
from typing import Union, List

# External library imports
from matplotlib.pylab import f
import matplotlib.pyplot as plt
from sympy import Trace, plot
import torch

# External type imports
from pyro.infer import Trace_ELBO
from pyro.nn.module import PyroSample
from pyro.distributions import Uniform
from torch import tensor

# Internal imports
from .. import utils, gp

class OscillatorDetector:
    def __init__(self, path: str = None):
        """
        Initialize the Oscillator Detector
        
        :param str path: Path to the csv file
        """
        if path is not None:
            self.time, self.bckgd, self.bckgd_length, self.M, self.y_all, self.y_length, self.N = utils.load_data(path)
            self.allowed = set(["background", "detrend"])

    def load_data(self, path: str):
        """
        Load data from a csv file
        
        :param str path: Path to the csv file
        """
        self.time, self.bckgd, self.bckgd_length, self.M, self.y_all, self.y_length, self.N = utils.load_data(path)

    def fit_models(self, *args, **kwargs):
        """
        Fit background noise and trend models, adjust data and fit OU and OU+Oscillator models

        :param bool verbose: Print fitting progress
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
            print(f"Loaded data with {self.N} cells and {self.M} background noise models")
            print(f"Plots: {"on" if plots else "off"}")
            print("\n")
            print("Fitting background noise...")

        # --- data preprocessing --- #
        
        # centre all data
        for i in range(self.N):
            y_curr = self.y_all[:self.y_length[i],i]
            y_curr -= torch.mean(y_curr)
        for i in range(self.M):
            y_curr = self.bckgd[:self.bckgd_length[i],i]
            y_curr -= torch.mean(y_curr)

        # --- background noise --- #
        std, models = gp.background_noise(self.time, self.bckgd, self.bckgd_length, self.M, verbose=verbose)
        self.bckgd_std = std
        self.bckgd_models = models

        # plot and start next step
        if "background" in plots:
            self.plot("background")
        if verbose:
            print("\nDetrending and denoising cell data...")


        # --- detrend and denoise cell data --- #
        self.model_detrend = [gp.GaussianProcess] * self.N
        self.y_detrend, self.noise_detrend, self.LLR_list, self.BIC_list, self.OU_params, self.OUosc_params = [[torch.Tensor] * self.N for _ in range(6)]

        ou_priors = {
            "lengthscale": PyroSample(Uniform(tensor(0.1), tensor(2.0))),
            "variance": PyroSample(Uniform(tensor(0.1), tensor(2.0)))
        }
        osc_priors = {
            "lengthscale": PyroSample(Uniform(tensor(0.1), tensor(4.0)))
        }

        # loop through and fit models
        for i in range(self.N):
            # reference input data
            X_curr = self.time[:self.y_length[i]]
            y_curr = self.y_all[:self.y_length[i],i]

            # normalise and reshape inplace
            noise = self.bckgd_std / torch.std(y_curr)
            y_curr = y_curr / torch.std(y_curr)

            # detrend
            res = gp.detrend(X_curr, y_curr, 7.1, verbose=verbose)

            # skip if failed
            if res is None:
                continue
            y_detrended, noise_model = res
            
            # fit OU, OU+Oscillator
            """ou = gp.OU(ou_priors)
            ouosc = gp.OUosc(ou_priors, osc_priors)

            ou.fit(X_curr, y_detrend, Trace_ELBO().differentiable_loss)
            ouosc.fit(X_curr, y_detrend, Trace_ELBO().differentiable_loss)"""

            self.y_detrend[i] = y_detrended
            self.model_detrend[i] = noise_model
            self.noise_detrend[i] = noise

        if "detrend" in plots:
            self.plot("detrend")

    def plot(self, target: str): 
        """
        Plot the data
        
        :param str target: String or List of strings describing plot types
        """
        plot_size = 15 / 5
        if target == "background":
            # generate grid for background models
            dim = ceil(sqrt(self.M))
            fig = plt.figure(figsize=(plot_size * dim, plot_size * dim))

            for i, m in enumerate(self.bckgd_models):
                plt.subplot(dim, dim, i+1)
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
                if self.model_detrend[i] is None:
                    continue

                m, y_detrended = self.model_detrend[i], self.y_detrend[i]

                # plot
                plt.subplot(dim, dim, i+1)
                m.test_plot()
                with torch.no_grad():
                    plt.plot(self.time[:self.y_length[i]], y_detrended, label="Detrended")

                plt.title(f"Cell {i+1}")
            plt.legend()