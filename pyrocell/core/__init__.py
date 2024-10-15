# Standard library imports
from typing import Union, List

# External library imports
import matplotlib.pyplot as plt
from sympy import Trace
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

    def load_data(self, path: str):
        """
        Load data from a csv file
        
        :param str path: Path to the csv file
        """
        self.time, self.bckgd, self.bckgd_length, self.M, self.y_all, self.y_length, self.N = utils.load_data(path)

    def fit_models(self, verbose: bool = False):
        """
        Fit background noise and trend models, adjust data and fit OU and OU+Oscillator models

        :param bool verbose: Print fitting progress
        """
        if verbose:
            print("Fitting background noise...")

        # background noise
        std, models = gp.background_noise(self.time, self.bckgd, self.bckgd_length, self.M, verbose=verbose)
        self.bckgd_std = std
        self.bckgd_models = models

        if verbose:
            print("\nDetrending and denoising cell data...")

        # detrend and denoise cell data
        self.mean_detrend, self.var_detrend, self.noise_detrend, self.LLR_list, self.BIC_list, self.OU_params, self.OUosc_params = [[torch.Tensor] * self.N for _ in range(7)]

        ou_priors = {
            "lengthscale": PyroSample(Uniform(tensor(0.1), tensor(2.0))),
            "variance": PyroSample(Uniform(tensor(0.1), tensor(2.0)))
        }
        osc_priors = {
            "lengthscale": PyroSample(Uniform(tensor(0.1), tensor(4.0)))
        }

        for i in range(self.N):
            # extract and normalize
            X_curr = self.time[:self.y_length[i]]
            y_curr = self.y_all[:self.y_length[i],i,None]
            noise = self.bckgd_std / torch.std(y_curr)
            y_curr = (y_curr - torch.mean(y_curr)) / torch.std(y_curr)

            # remove y-dim
            y_curr = y_curr.reshape(-1)

            # detrend and plot
            mean, var, y_detrend = gp.detrend(X_curr, y_curr, 7.1)
            if verbose and i == 1:
                plt.plot(X_curr, y_curr, label="Original")
                plt.plot(X_curr, y_detrend, label="Detrended")
                plt.plot(X_curr, mean, label="Mean")
                plt.legend()

            # fit OU, OU+Oscillator
            ou = gp.OU(ou_priors)
            ouosc = gp.OUosc(ou_priors, osc_priors)

            ou.fit(X_curr, y_detrend, Trace_ELBO().differentiable_loss)
            ouosc.fit(X_curr, y_detrend, Trace_ELBO().differentiable_loss)

            self.mean_detrend[i] = mean
            self.var_detrend[i] = var
            self.noise_detrend[i] = noise



    def plot(self, target: Union[str, List[str]]):
        """
        Plot the data
        
        :param str target: String or List of strings describing plot types
        """
        if isinstance(target, str):
            target = set([target])
        else:
            target = set(target)

        allowed = set(["background"])
        if not target.issubset(allowed):
            raise ValueError(f"{target - allowed} are not valid plot types")
        
        for t in target:
            if t == "background":
                fig = plt.figure(figsize=(15/2.54,15/2.54))

                for i, m in enumerate(self.bckgd_models):
                    plt.subplot(2, 2, i + 1)
                    m.test_plot(plot_sd=True)

                plt.tight_layout()