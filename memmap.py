# Standard Library Imports
from functools import partial
import operator

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

# Direct Namespace Imports
from gpflow.kernels import Matern12, Cosine

# Internal Project Imports
from gpcell.utils import benchmark_memmap_performance
from gpcell.backend.priors import hes_ouosc_prior, ouosc_trainables


# Benchmark Parameters
max_lengths = [int(x) for x in np.linspace(10, 30, 3, dtype=int)]
num_traces = [5, 10, 20, 50]  # , 100, 200]
N, K = max(num_traces), 10
noise = np.float64(0.1)

# Model Parameters
kernels = [Matern12, Cosine]
ouosc_prior_gens = [partial(hes_ouosc_prior, noise) for _ in range(N)]
trainables = ouosc_trainables

# Flags
Y_var = False
verbose = False
mcmc = False
op = operator.mul


def generate_data() -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    # Generate Data
    X_list = []
    Y_list = []

    for length in max_lengths:
        for num in num_traces:
            # simulate signals with varying lengths
            X_lengths = np.random.randint(int(length // 1.5), length + 1, num)
            X = [
                np.linspace(0, X_length / 10, X_length).reshape(
                    -1, 1
                )  # Scale time axis
                for X_length in X_lengths
            ]
            Y = []

            for x in X:
                # Create signal with:
                # 1. Main oscillatory component
                # 2. Secondary higher-frequency component
                # 3. Small nonlinear trend
                # 4. Noise with appropriate scale
                t = x.flatten()

                # Primary oscillation (main signal)
                primary = np.sin(2 * np.pi * 0.5 * t)

                # Secondary faster oscillation with smaller amplitude
                secondary = 0.3 * np.sin(2 * np.pi * 1.5 * t + 0.3)

                # Add slight trend (quadratic)
                trend = 0.1 * t**2 / max(t) ** 2

                # Combine components
                signal = primary + secondary + trend

                # Add noise with appropriate scale (less noise for clearer pattern)
                noise_scale = 0.15
                noisy_signal = signal + noise_scale * np.random.randn(len(t))

                Y.append(noisy_signal.reshape(-1, 1))

            X_list.append(X)
            Y_list.append(Y)

    return X_list, Y_list


def plot_data(X_list: list[list[np.ndarray]], Y_list: list[list[np.ndarray]]):
    # Plot a subset of the data, with separate subplots for each max_length group
    fig, axes = plt.subplots(
        1, len(max_lengths), figsize=(20, 6), sharex=True, sharey=True
    )

    # Use a different color for each example in the subplot
    colors = ["blue", "green", "red", "purple", "orange"]

    # Number of examples to show per length group
    examples_per_length = 5

    # Plot processes grouped by length
    for length_idx, length in enumerate(max_lengths):
        ax = axes[length_idx]
        count = 0  # Count examples for this length

        # Set subplot title
        ax.set_title(f"Length: {length}")

        # Find all data sets with this length
        for i, (X, Y) in enumerate(zip(X_list, Y_list)):
            curr_length_idx = i // len(num_traces)
            if curr_length_idx != length_idx:
                continue

            # Plot some examples from this group
            for j, (x, y) in enumerate(zip(X, Y)):
                if count >= examples_per_length:
                    break

                # Plot with different colors for each example
                ax.plot(
                    x.flatten(),
                    y.flatten(),
                    alpha=0.8,
                    linewidth=1.5,
                    color=colors[count % len(colors)],
                    label=f"Example {count + 1}" if length_idx == 0 else "_nolegend_",
                )

                count += 1

            if count >= examples_per_length:
                break

    # Add common labels
    fig.suptitle("Stochastic Processes by Maximum Length", fontsize=14)

    # Only add legend to the first subplot to avoid repetition
    if len(max_lengths) > 0:
        axes[0].legend(loc="best")

    # Add grid to all subplots
    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate Data
    X_list, Y_list = generate_data()

    # Plot Data
    # plot_data(X_list, Y_list)

    # Benchmark Memory-Mapped Performance
    benchmark_memmap_performance(
        X_list,
        Y_list,
        max_lengths,
        num_traces,
        kernels,
        ouosc_prior_gens,
        trainables,
        K,
        Y_var,
        mcmc,
        op,
        verbose,
    )
