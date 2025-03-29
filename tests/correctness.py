# Standard Library Imports
from typing import (
    List,
    Tuple,
)
import unittest
import os

# Third-Party Library Imports
import numpy as np
import pandas as pd
import tensorflow_probability as tfp

# Direct Namespace Imports
from gpflow.utilities import to_default_float
import gpflow

# Internal Project Imports
import gpcell as gc
from gpcell import backend, utils
from gpcell.backend._types import GPPrior

# --------------------------------------------------------- #
# --- script to check correctness of the implementation --- #
# --------------------------------------------------------- #


class TestCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.path = "data/hes/Hes1_example.csv"

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # read in data
        cls.X, cls.Y = utils.load_data(cls.path, "Time (h)", "Cell")
        cls.X_bckgd, cls.bckgd = utils.load_data(cls.path, "Time (h)", "Background")
        cls.N, cls.M = len(cls.Y), len(cls.bckgd)
        cls.K = 10

        # calculate background noise
        print("Calculating background noise\n")
        cls.mean_noise, cls.bckgd_GPs = utils.background_noise(
            cls.X_bckgd, cls.bckgd, 7.0
        )
        cls.noise_list = [cls.mean_noise / np.std(y) for y in cls.Y]
        print(f"GPCell: {cls.mean_noise}, {[float(gp.noise) for gp in cls.bckgd_GPs]}")

        # generate priors
        cls.OU_prior_gens, cls.OUosc_prior_gens = generate_prior_data(
            cls.N, cls.noise_list, 5
        )

        # run the original analysis
        (
            cls.notebook_noise_list,
            cls.detrend_param_list,
            cls.LLR_list,
            cls.BICdiff_list,
            cls.OU_param_list,
            cls.OUosc_param_list,
            cls.time,
            cls.y_length,
        ) = original_analysis(cls.path, cls.OU_prior_gens, cls.OUosc_prior_gens, cls.K)

        # run the GPCell analysis
        params = {
            "verbose": True,
            "joblib": True,
            "set_noise": cls.mean_noise,
            "ou_prior_gen": cls.OU_prior_gens,
            "ouosc_prior_gen": cls.OUosc_prior_gens,
        }
        cls.detector = gc.OscillatorDetector(cls.X, cls.Y, cls.N, **params)
        cls.detector.fit(methods=["BIC"])

    def test_noise(self):
        # check if the results are the same
        self.assertTrue(np.allclose(self.notebook_noise_list, self.noise_list))

    def test_detrend_params(self):
        # check if the results are the same
        for i in range(self.N):
            self.assertTrue(
                np.allclose(
                    self.detrend_param_list[i].lengthscales.numpy(),
                    self.detector.detrend_GPs[i].fit_gp.kernel.lengthscales.numpy(),  # type: ignore
                )
            )

        for i in range(self.N):
            self.assertTrue(
                np.allclose(
                    self.detrend_param_list[i].variance.numpy(),
                    self.detector.detrend_GPs[i].fit_gp.kernel.variance.numpy(),  # type: ignore
                )
            )

    def test_OU_params(self):
        gpcell_OU_kernels = self.detector.k_ou_list
        for i in range(self.N):
            self.assertAlmostEqual(
                self.OU_param_list[i].lengthscales.numpy(),
                gpcell_OU_kernels[i].lengthscales.numpy(),  # type: ignore
            )
            self.assertAlmostEqual(
                self.OU_param_list[i].variance.numpy(),
                gpcell_OU_kernels[i].variance.numpy(),  # type: ignore
            )

    def test_OUosc_params(self):
        gpcell_OUosc_kernels = self.detector.k_ouosc_list
        for i in range(self.N):
            self.assertAlmostEqual(
                self.OUosc_param_list[i].kernels[0].lengthscales.numpy(),
                gpcell_OUosc_kernels[i].kernels[0].lengthscales.numpy(),  # type: ignore
            )
            self.assertAlmostEqual(
                self.OUosc_param_list[i].kernels[0].variance.numpy(),
                gpcell_OUosc_kernels[i].kernels[0].variance.numpy(),  # type: ignore
            )
            self.assertAlmostEqual(
                self.OUosc_param_list[i].kernels[1].lengthscales.numpy(),
                gpcell_OUosc_kernels[i].kernels[1].lengthscales.numpy(),  # type: ignore
            )

    def test_BICdiff(self):
        self.assertTrue(
            np.allclose(
                self.BICdiff_list,
                self.detector.BIC_diffs,
            )
        )

    def test_LLR(self):
        self.assertTrue(
            np.allclose(
                self.LLR_list,
                self.detector.LLRs,
            )
        )

    # def test_bootstrap(self):
    #     # run the bootstrap analysis
    #     osc_filt, LLR_synth_array = bootstrap_analysis(
    #         self.time,
    #         self.y_length,
    #         self.N,
    #         self.noise_list,
    #         self.LLR_list,
    #         self.OU_param_list,
    #         self.detrend_param_list,
    #         self.OU_prior_gens,
    #         self.OUosc_prior_gens,
    #     )

    #     self.detector.fit(methods=["bootstrap"])

    #     # check if the results are the same
    #     self.assertTrue(np.allclose(osc_filt, self.detector.osc_filt))


# prior generation function
def generate_prior_data(
    n: int, noise_list: List[np.float64], K: int
) -> Tuple[List[backend.GPPriorFactory], List[backend.GPPriorFactory]]:
    """Function to generate priors that can be used to verify models

    Args:
        n (int): number of cells
        K (int): number of replicates
    """
    np.random.seed(42)
    OU_priors = []
    OUosc_priors = []

    for i in range(n):
        OU = []
        OUosc = []
        noise = noise_list[i]
        for _ in range(K):
            l_1 = np.random.uniform(0.1, 2.0)
            var = np.random.uniform(0.1, 2.0)
            l_2 = np.random.uniform(0.1, 4.0)
            OU.append(
                {
                    "kernel.lengthscales": l_1,
                    "kernel.variance": var,
                    "likelihood.variance": noise**2,
                }
            )
            OUosc.append(
                {
                    "kernel.kernels[0].lengthscales": l_1,
                    "kernel.kernels[0].variance": var,
                    "kernel.kernels[1].lengthscales": l_2,
                    "likelihood.variance": noise**2,
                }
            )
        OU_priors.append(OU)
        OUosc_priors.append(OUosc)

    # create fixed prior classes
    class fixed_prior_generator(backend.FixedPriorGen):
        def __init__(self, priors: List[backend.GPPrior]):
            self.priors = priors
            self.i = 0

        def __call__(self, noise=None) -> GPPrior:
            prior = self.priors[self.i]
            self.i = (self.i + 1) % K
            return prior

    OU_prior_gens = [fixed_prior_generator(priors) for priors in OU_priors]
    OUosc_prior_gens = [fixed_prior_generator(priors) for priors in OUosc_priors]

    return OU_prior_gens, OUosc_prior_gens


# their notebook
def original_analysis(
    path: str,
    OU_prior_gens: List[backend.GPPriorFactory],
    OUosc_prior_gens: List[backend.GPPriorFactory],
    K: int,
):
    # ----------------- #
    # --- LOAD DATA --- #
    # ----------------- #
    def load_data(file_name):
        df = pd.read_csv(file_name).fillna(0)
        data_cols = [col for col in df if col.startswith("Cell")]  # type: ignore
        bckgd_cols = [col for col in df if col.startswith("Background")]  # type: ignore
        time = df["Time (h)"].values[:, None]

        bckgd = df[bckgd_cols].values
        M = np.shape(bckgd)[1]

        bckgd_length = np.zeros(M, dtype=np.int32)

        for i in range(M):
            bckgd_curr = bckgd[:, i]
            bckgd_length[i] = np.max(np.nonzero(bckgd_curr)) + 1

        y_all = df[data_cols].values

        N = np.shape(y_all)[1]

        y_all = df[data_cols].values
        np.max(np.nonzero(y_all))

        y_length = np.zeros(N, dtype=np.int32)

        for i in range(N):
            y_curr = y_all[:, i]
            y_length[i] = np.max(np.nonzero(y_curr)) + 1

        return time, bckgd, bckgd_length, M, y_all, y_length, N

    time, bckgd, bckgd_length, M, y_all, y_length, N = load_data(path)

    # ------------------------ #
    # --- BACKGROUND MODEL --- #
    # ------------------------ #

    def optimised_background_model(X, Y):
        k = gpflow.kernels.SquaredExponential()
        m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
        m.kernel.lengthscales = gpflow.Parameter(
            to_default_float(7.1),
            transform=tfp.bijectors.Softplus(low=to_default_float(7.0)),
        )
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            m.training_loss,
            m.trainable_variables,  # type: ignore
            options=dict(maxiter=100),
        )

        return m

    std_vec = np.zeros(M)

    for i in range(M):
        # was not the same as above
        X = time[: bckgd_length[i]]
        Y = bckgd[: bckgd_length[i], i, None]
        Y = Y - np.mean(Y)

        m = optimised_background_model(X, Y)

        mean, var = m.predict_y(X)

        std_vec[i] = m.likelihood.variance**0.5  # type: ignore
    std = np.mean(
        std_vec
    )  # the estimated standard deviation of the experimental noise, averaged over all background traces
    print(f"Notebook: {std}, {std_vec}\n\n")

    (
        noise_list,
        detrend_param_list,
        LLR_list,
        BICdiff_list,
        OU_param_list,
        OUosc_param_list,
    ) = [[] for _ in range(6)]

    for cell in range(N):
        x_curr = time[: y_length[cell]]
        y_curr = y_all[: y_length[cell], cell, None]

        noise = std / np.std(y_curr)
        y_curr = (y_curr - np.mean(y_curr)) / np.std(y_curr)

        k_trend, mean_trend, var_trend, Y_detrended = detrend_cell(x_curr, y_curr, 7.0)

        LLR, BICdiff, k_ou, k_ou_osc = fit_models(
            x_curr, Y_detrended, noise, K, OU_prior_gens[cell], OUosc_prior_gens[cell]
        )

        noise_list.append(noise)
        detrend_param_list.append(k_trend)
        LLR_list.append(LLR)
        BICdiff_list.append(BICdiff)
        OU_param_list.append(k_ou)
        OUosc_param_list.append(k_ou_osc)

    cutoff = 3
    print(
        "Number of cells counted as oscillatory (Notebook): {0}/{1}".format(
            sum(np.array(BICdiff_list) > cutoff), len(BICdiff_list)
        )
    )

    return (
        noise_list,
        detrend_param_list,
        LLR_list,
        BICdiff_list,
        OU_param_list,
        OUosc_param_list,
        time,
        y_length,
    )


# ------------------ #
# --- DETRENDING --- #
# ------------------ #


def detrend_cell(X, Y, detrend_lengthscale):
    k_trend = gpflow.kernels.SquaredExponential()
    m = gpflow.models.GPR(data=(X, Y), kernel=k_trend, mean_function=None)

    m.kernel.lengthscales = gpflow.Parameter(
        to_default_float(detrend_lengthscale + 0.1),
        transform=tfp.bijectors.Softplus(low=to_default_float(detrend_lengthscale)),
    )

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))  # type: ignore

    mean, var = m.predict_f(X)

    Y_detrended = Y - mean
    Y_detrended = Y_detrended - np.mean(Y_detrended)

    return k_trend, mean, var, Y_detrended


# ------------------------------- #
# --- FIT OU AND OUOSC MODELS --- #
# ------------------------------- #


def fit_models(X, Y, noise, K, OU_prior_gen, OUosc_prior_gen):
    OU_LL_list, OU_param_list, OUosc_LL_list, OUosc_param_list = [[] for _ in range(4)]

    for k in range(K):
        # setup OU model
        k_ou = gpflow.kernels.Matern12()
        m = gpflow.models.GPR(data=(X, Y), kernel=k_ou, mean_function=None)

        # get priors
        OU_priors = OU_prior_gen()

        # assign priors and run
        m.kernel.variance.assign(OU_priors["kernel.variance"])  # type: ignore
        m.kernel.lengthscales.assign(OU_priors["kernel.lengthscales"])  # type: ignore
        m.likelihood.variance.assign(OU_priors["likelihood.variance"])  # type: ignore
        gpflow.set_trainable(m.likelihood.variance, False)  # type: ignore

        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            m.training_loss,
            m.trainable_variables,  # type: ignore
            options=dict(maxiter=100),
        )

        # extract
        nlmlOU = m.log_posterior_density()
        OU_LL = nlmlOU
        OU_LL_list.append(OU_LL)
        OU_param_list.append(k_ou)

        # setup OUosc model
        k_ou_osc = gpflow.kernels.Matern12() * gpflow.kernels.Cosine()
        m = gpflow.models.GPR(data=(X, Y), kernel=k_ou_osc, mean_function=None)

        # get priors
        OUosc_priors = OUosc_prior_gen()

        # assign priors and run
        m.kernel.kernels[0].variance.assign(  # type: ignore
            OUosc_priors["kernel.kernels[0].variance"]
        )
        m.kernel.kernels[0].lengthscales.assign(  # type: ignore
            OUosc_priors["kernel.kernels[0].lengthscales"]
        )
        m.kernel.kernels[1].lengthscales.assign(  # type: ignore
            OUosc_priors["kernel.kernels[1].lengthscales"]
        )
        m.likelihood.variance.assign(OUosc_priors["likelihood.variance"])  # type: ignore
        gpflow.set_trainable(m.likelihood.variance, False)  # type: ignore
        gpflow.set_trainable(m.kernel.kernels[1].variance, False)  # type: ignore
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            m.training_loss,
            m.trainable_variables,  # type: ignore
            options=dict(maxiter=100),
        )

        # extract
        nlmlOSC = m.log_posterior_density()  # opt_logs.fun
        OU_osc_LL = nlmlOSC
        OUosc_LL_list.append(OU_osc_LL)
        OUosc_param_list.append(k_ou_osc)

    LLR = 100 * 2 * (np.max(OUosc_LL_list) - np.max(OU_LL_list)) / len(Y)
    BIC_OUosc = -2 * np.max(OUosc_LL_list) + 3 * np.log(len(Y))
    BIC_OU = -2 * np.max(OU_LL_list) + 2 * np.log(len(Y))
    BICdiff = BIC_OU - BIC_OUosc
    k_ou = OU_param_list[np.argmax(OU_LL_list)]
    k_ou_osc = OUosc_param_list[np.argmax(OUosc_LL_list)]

    return LLR, BICdiff, k_ou, k_ou_osc


# ---------------------------#
# --- BOOTSTRAP ANALYSIS --- #
# ---------------------------#


def bootstrap_analysis(
    time: np.ndarray,
    y_length: np.ndarray,
    N: int,
    noise_list: List[np.float64],
    LLR_list: List[np.float64],
    OU_param_list: List[gpflow.kernels.Kernel],
    detrend_param_list: List[gpflow.kernels.Kernel],
    OU_prior_gens: List[backend.GPPriorFactory],
    OUosc_prior_gens: List[backend.GPPriorFactory],
):
    repeats = 10  # this controls the number of synthetic OU cells simulated for each observed cell

    LLR_list_synth = []

    for cell in range(N):
        print(cell)

        X = time[: y_length[cell]]
        noise = noise_list[cell]

        k_se = detrend_param_list[cell]
        k_ou = OU_param_list[cell]
        k_white = gpflow.kernels.White(variance=noise**2)  # type: ignore

        k_synth = k_se + k_ou + k_white

        for repeat in range(repeats):
            y_synth = np.random.multivariate_normal(
                np.zeros(len(X)), k_synth(X)
            ).reshape(-1, 1)
            k_trend, mean_trend, var_trend, Y_detrended = detrend_cell(X, y_synth, 7.0)
            LLR, BICdiff, k_ou, k_ou_osc = fit_models(
                X, Y_detrended, noise, 10, OU_prior_gens[cell], OUosc_prior_gens[cell]
            )
            LLR_list_synth.append(LLR)

    LLR_array = np.array(LLR_list)
    LLR_synth_array = np.array(LLR_list_synth)

    # LLRs can be tiny and just negative - this just sets them to zero
    LLR_array[LLR_array < 0] = 0
    LLR_synth_array[LLR_synth_array < 0] = 0

    LLR_combined = np.concatenate((LLR_array, LLR_synth_array), 0)

    upper = np.max(LLR_combined)
    lower1 = np.min(LLR_combined)
    lower = upper - 0.9 * (upper - lower1)
    grid = np.linspace(lower, upper, 20)

    piest = np.zeros_like(grid)

    for i, cutoff in enumerate(grid):
        num = sum(LLR_array < cutoff) / len(LLR_array)
        denom = sum(LLR_synth_array < cutoff) / len(LLR_synth_array)
        piest[i] = num / denom

    xx = np.linspace(lower, upper, 100)
    from scipy.interpolate import CubicSpline

    cs = CubicSpline(grid, piest)
    yy = cs(xx)

    piGUESS1 = yy[0]

    I = np.argsort(LLR_array)

    LLR_array_sorted = LLR_array[I]

    q1 = np.zeros_like(LLR_array_sorted)

    for i, thresh in enumerate(LLR_array_sorted):
        q1[i] = (
            piGUESS1
            * (sum(LLR_synth_array >= thresh) / len(LLR_synth_array))
            / (sum(LLR_array_sorted >= thresh) / len(LLR_array_sorted))
        )

    q_vals = q1[np.argsort(I)]
    osc_filt = q_vals < 0.05

    print(
        "Number of cells counted as oscillatory (full method): {0}/{1}".format(
            sum(osc_filt), len(osc_filt)
        )
    )

    return osc_filt, LLR_synth_array
