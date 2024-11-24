import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import print_summary, set_trainable, to_default_float
from scipy.signal import find_peaks


class TestOsc:
    def __init__(self):
        def load_data(file_name):
            df = pd.read_csv(file_name).fillna(0)
            data_cols = [col for col in df if col.startswith("Cell")]
            bckgd_cols = [col for col in df if col.startswith("Background")]
            time = df["Time (h)"].values[:, None]

            bckgd = df[bckgd_cols].values
            M = np.shape(bckgd)[1]

            bckgd_length = np.zeros(M, dtype=np.int32)

            for i in range(M):
                bckgd_curr = bckgd[:, i]
                bckgd_length[i] = np.max(np.nonzero(bckgd_curr))

            y_all = df[data_cols].values

            N = np.shape(y_all)[1]

            y_all = df[data_cols].values
            np.max(np.nonzero(y_all))

            y_length = np.zeros(N, dtype=np.int32)

            for i in range(N):
                y_curr = y_all[:, i]
                y_length[i] = np.max(np.nonzero(y_curr))

            return time, bckgd, bckgd_length, M, y_all, y_length, N

        import os

        os.chdir("../data/hes")

        file_name = "Hes1_example.csv"  # add your file name here
        time, bckgd, bckgd_length, M, y_all, y_length, N = load_data(file_name)

        def optimised_background_model(X, Y):
            k = gpflow.kernels.SquaredExponential()
            m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
            m.kernel.lengthscales = gpflow.Parameter(
                to_default_float(7.1),
                transform=tfp.bijectors.Softplus(low=to_default_float(7.0)),
            )
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(
                m.training_loss, m.trainable_variables, options=dict(maxiter=100)
            )

            print(m.kernel.parameters)

            return m

        std_vec = np.zeros(M)

        fig = plt.figure(figsize=(15 / 2.54, 15 / 2.54))

        for i in range(M):
            X = time[: bckgd_length[i]]
            Y = bckgd[: bckgd_length[i], i, None]
            Y = Y - np.mean(Y)

            m = optimised_background_model(X, Y)

            mean, var = m.predict_y(X)

            plt.subplot(2, 2, i + 1)
            plt.plot(X, Y)
            plt.plot(X, mean, "k")
            plt.plot(X, mean + 2 * var**0.5, "r")
            plt.plot(X, mean - 2 * var**0.5, "r")

            if i % 2 == 0:
                plt.ylabel("Luminescence (AU)")
            if i >= 2:
                plt.xlabel("Time (h)")

            std_vec[i] = m.likelihood.variance**0.5

        plt.tight_layout()
        std = np.mean(
            std_vec
        )  # the estimated standard deviation of the experimental noise, averaged over all background traces

        print(std)

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

        def fit_models(X, Y, noise, K):
            OU_LL_list, OU_param_list, OUosc_LL_list, OUosc_param_list = [
                [] for _ in range(4)
            ]

            for k in range(K):
                k_ou = gpflow.kernels.Matern12()

                m = gpflow.models.GPR(data=(X, Y), kernel=k_ou, mean_function=None)
                m.kernel.variance.assign(np.random.uniform(0.1, 2.0))
                m.kernel.lengthscales.assign(np.random.uniform(0.1, 2.0))
                m.likelihood.variance.assign(noise**2)
                gpflow.set_trainable(m.likelihood.variance, False)

                # gpflow.utilities.print_summary(m)
                opt = gpflow.optimizers.Scipy()
                opt_logs = opt.minimize(
                    m.training_loss, m.trainable_variables, options=dict(maxiter=100)
                )

                nlmlOU = m.log_posterior_density()

                OU_LL = nlmlOU
                OU_LL_list.append(OU_LL)
                OU_param_list.append(k_ou)

                k_ou_osc = gpflow.kernels.Matern12() * gpflow.kernels.Cosine()

                m = gpflow.models.GPR(data=(X, Y), kernel=k_ou_osc, mean_function=None)
                m.likelihood.variance.assign(noise**2)
                gpflow.set_trainable(m.likelihood.variance, False)
                gpflow.set_trainable(m.kernel.kernels[1].variance, False)
                m.kernel.kernels[0].variance.assign(np.random.uniform(0.1, 2.0))
                m.kernel.kernels[0].lengthscales.assign(np.random.uniform(0.1, 2.0))
                m.kernel.kernels[1].lengthscales.assign(np.random.uniform(0.1, 4.0))

                # print_summary(m)
                opt = gpflow.optimizers.Scipy()
                opt_logs = opt.minimize(
                    m.training_loss, m.trainable_variables, options=dict(maxiter=100)
                )

                # print_summary(m)
                # print("---")

                nlmlOSC = m.log_posterior_density()  # opt_logs.fun

                OU_osc_LL = nlmlOSC
                OUosc_LL_list.append(OU_osc_LL)
                OUosc_param_list.append(k_ou_osc)

            print(OU_LL_list)
            print(len(OU_LL_list))
            print(OUosc_LL_list)
            print(len(OUosc_LL_list))

            LLR = 100 * 2 * (np.max(OUosc_LL_list) - np.max(OU_LL_list)) / len(Y)
            BIC_OUosc = -2 * np.max(OUosc_LL_list) + 3 * np.log(len(Y))
            BIC_OU = -2 * np.max(OU_LL_list) + 2 * np.log(len(Y))
            BICdiff = BIC_OU - BIC_OUosc

            print(np.max(OUosc_LL_list), np.max(OU_LL_list))
            print(BIC_OUosc, BIC_OU)
            print(BICdiff)

            k_ou = OU_param_list[np.argmax(OU_LL_list)]
            k_ou_osc = OUosc_param_list[np.argmax(OUosc_LL_list)]

            cov_ou_osc = OUosc_param_list[0](X).numpy()[0, :]
            peaks, _ = find_peaks(cov_ou_osc, height=0)

            if len(peaks) != 0:
                period = X[peaks[0]]
            else:
                period = 0

            return LLR, BICdiff, k_ou, k_ou_osc, period

        def plot_model_fits(
            cell,
            x_curr,
            y_curr,
            mean_trend,
            noise,
            LLR,
            k_trend,
            k_ou,
            k_ou_osc,
            period,
        ):
            fig = plt.figure(figsize=(12 / 2.54, 8 / 2.54))
            plt.plot(x_curr, y_curr)
            plt.plot(x_curr, mean_trend, "k--", alpha=0.5)
            plt.xlabel("Time (hours)")
            plt.ylabel("Luminescence (normalised) (AU)")
            plt.title("Cell " + str(cell) + " , LLR = " + f"{LLR:.1f}")

        (
            noise_list,
            detrend_param_list,
            LLR_list,
            BICdiff_list,
            OU_param_list,
            OUosc_param_list,
            period_list,
        ) = [[] for _ in range(7)]

        detrended_list = []
        mean_list = []

        for cell in range(N):
            print(cell)

            x_curr = time[: y_length[cell]]
            y_curr = y_all[: y_length[cell], cell, None]
            noise = std / np.std(y_curr)
            y_curr = (y_curr - np.mean(y_curr)) / np.std(y_curr)

            print("detrending")
            k_trend, mean_trend, var_trend, Y_detrended = detrend_cell(
                x_curr, y_curr, 7.0
            )
            print(mean_trend.shape)
            detrended_list.append(Y_detrended)
            mean_list.append(mean_trend)

            print("OU/OUosc")
            LLR, BICdiff, k_ou, k_ou_osc, period = fit_models(
                x_curr, Y_detrended, noise, 10
            )

            if cell == 0:
                plot_model_fits(
                    cell,
                    x_curr,
                    y_curr,
                    mean_trend,
                    noise,
                    LLR,
                    k_trend,
                    k_ou,
                    k_ou_osc,
                    period,
                )

            noise_list.append(noise)
            detrend_param_list.append(k_trend)
            LLR_list.append(LLR)
            BICdiff_list.append(BICdiff)
            OU_param_list.append(k_ou)
            OUosc_param_list.append(k_ou_osc)
            period_list.append(period)

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
