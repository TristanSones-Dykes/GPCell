from typing import Union, List
from .. import utils, gp
import matplotlib.pyplot as plt

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
        """

        # background noise
        std, models = gp.background_noise(self.time, self.bckgd, self.bckgd_length, self.M, verbose=verbose)
        self.bckgd_std = std
        self.bckgd_models = models

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