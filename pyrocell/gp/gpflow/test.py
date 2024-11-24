from re import A
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import print_summary, set_trainable, to_default_float
from scipy.signal import find_peaks

from pyrocell.gp.gpflow.models import NoiseModel
from pyrocell.gp.gpflow.utils import fit_models, load_data, background_noise, detrend

from numpy.testing import assert_equal


class TestOsc:
    def __init__(self):
        import os

        os.chdir("../data/hes")

        file_name = "Hes1_example.csv"  # add your file name here
        time, bckgd = load_data(file_name, "Time (h)", "Background")
        M = len(bckgd)

        y_time, y_all = load_data(file_name, "Time (h)", "Cell")
        N = len(y_all)

        y_test, bckgd_test = y_all.copy(), bckgd.copy()

        assert_equal(y_all[0], y_test[0])
        assert_equal(bckgd[0], bckgd_test[0])

        new_priors = [
            {
                "lengthscale": gpflow.Parameter(
                    to_default_float(7.1),
                    transform=tfp.bijectors.Softplus(low=to_default_float(7.0)),
                )
            }
            for _ in range(M)
        ]
        GPs = fit_models(time, bckgd, NoiseModel, new_priors, 1)
        new_std_vec = [gp.fit_gp.likelihood.variance**0.5 for gp in GPs]
        new_std = np.mean(new_std_vec)

        newer_std, newer_GPs = background_noise(time, bckgd, 7.0)

        print(new_std)
        print(newer_std, "\n")

        assert_equal(y_all[0], y_test[0])
        assert_equal(bckgd[0], bckgd_test[0])

        def detrend_cell(X, Y, detrend_lengthscale):
            k_trend = gpflow.kernels.SquaredExponential()
            m = gpflow.models.GPR(data=(X, Y), kernel=k_trend, mean_function=None)

            m.kernel.lengthscales = gpflow.Parameter(
                to_default_float(detrend_lengthscale + 0.1),
                transform=tfp.bijectors.Softplus(
                    low=to_default_float(detrend_lengthscale)
                ),
            )

            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(
                m.training_loss, m.trainable_variables, options=dict(maxiter=100)
            )

            mean, var = m.predict_f(X)

            Y_detrended = Y - mean
            Y_detrended = Y_detrended - np.mean(Y_detrended)

            return k_trend, mean, var, Y_detrended

        noise_list, detrended_list, mean_list = [[] for _ in range(3)]
        detrend_param_list = []
        for cell in range(N):
            x_curr = y_time[cell]
            y = y_all[cell]
            noise = new_std / np.std(y)
            y_curr = (y - np.mean(y)) / np.std(y)

            k_trend, mean_trend, var_trend, Y_detrended = detrend_cell(
                x_curr, y_curr, 7.0
            )
            detrended_list.append(Y_detrended)
            mean_list.append(mean_trend)
            noise_list.append(noise)
            detrend_param_list.append(k_trend)

        new_priors = [
            {
                "lengthscale": gpflow.Parameter(
                    to_default_float(7.1),
                    transform=tfp.bijectors.Softplus(low=to_default_float(7.0)),
                )
            }
            for _ in range(N)
        ]
        GPs = fit_models(y_time, y_all, NoiseModel, new_priors, 2)
        new_detrend_param_list = [gp.fit_gp.kernel.parameters for gp in GPs]

        newer_detrend, newer_GPs = detrend(y_time, y_all, 7.0)

        print(noise_list, "\n")

        for old, new, newer in zip(
            detrend_param_list, new_detrend_param_list, newer_GPs
        ):
            print(old.parameters)
            print(newer.fit_gp.kernel.parameters)
            print(new, "\n")

        assert_equal(y_all[0], y_test[0])
        assert_equal(bckgd[0], bckgd_test[0])

        BICdiff_list = []

        def fit_ou_ouosc(X, Y, noise, K):
            OU_LL_list, OUosc_LL_list = [[] for _ in range(2)]

            for k in range(K):
                k_ou = gpflow.kernels.Matern12()

                m = gpflow.models.GPR(data=(X, Y), kernel=k_ou, mean_function=None)
                m.kernel.variance.assign(np.random.uniform(0.1, 2.0))
                m.kernel.lengthscales.assign(np.random.uniform(0.1, 2.0))
                m.likelihood.variance.assign(noise**2)
                gpflow.set_trainable(m.likelihood.variance, False)
                opt = gpflow.optimizers.Scipy()
                opt_logs = opt.minimize(
                    m.training_loss,
                    m.trainable_variables,
                    options=dict(maxiter=100),
                )

                nlmlOU = m.log_posterior_density()

                OU_LL = nlmlOU
                OU_LL_list.append(OU_LL)

                k_ou_osc = gpflow.kernels.Matern12() * gpflow.kernels.Cosine()

                m = gpflow.models.GPR(data=(X, Y), kernel=k_ou_osc, mean_function=None)
                m.likelihood.variance.assign(noise**2)
                gpflow.set_trainable(m.likelihood.variance, False)
                gpflow.set_trainable(m.kernel.kernels[1].variance, False)
                m.kernel.kernels[0].variance.assign(np.random.uniform(0.1, 2.0))
                m.kernel.kernels[0].lengthscales.assign(np.random.uniform(0.1, 2.0))
                m.kernel.kernels[1].lengthscales.assign(np.random.uniform(0.1, 4.0))
                opt = gpflow.optimizers.Scipy()
                opt_logs = opt.minimize(
                    m.training_loss,
                    m.trainable_variables,
                    options=dict(maxiter=100),
                )

                nlmlOSC = m.log_posterior_density()  # opt_logs.fun

                OU_osc_LL = nlmlOSC
                OUosc_LL_list.append(OU_osc_LL)

            BIC_OUosc = -2 * np.max(OUosc_LL_list) + 3 * np.log(len(Y))
            BIC_OU = -2 * np.max(OU_LL_list) + 2 * np.log(len(Y))
            BICdiff = BIC_OU - BIC_OUosc

            return BICdiff

        for cell in range(N):
            x_curr = y_time[cell]
            Y_detrended = detrended_list[cell]
            noise = noise_list[cell]

            BICdiff = fit_ou_ouosc(x_curr, Y_detrended, noise, 10)
            BICdiff_list.append(BICdiff)

        fig = plt.figure(figsize=(12 / 2.54, 6 / 2.54))

        cutoff = 3
        print(
            "Number of cells counted as oscillatory (BIC method): {0}/{1}".format(
                sum(np.array(BICdiff_list) > cutoff), len(BICdiff_list)
            )
        )

        plt.hist(BICdiff_list, bins=np.linspace(-20, 20, 40))
        plt.plot([cutoff, cutoff], [0, 2], "r--")
        plt.xlabel("LLR")
        plt.ylabel("Frequency")
        plt.title("LLRs of experimental cells")

        assert_equal(y_all[0], y_test[0])
        assert_equal(bckgd[0], bckgd_test[0])
