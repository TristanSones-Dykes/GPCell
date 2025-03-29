# Standard Library Imports
from typing import (
    Tuple,
)

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

# Direct Namespace Imports
from astropy.timeseries import LombScargle

# Internal Project Imports
from gpcell import OscillatorDetector
from gpcell.utils import load_sim
from gpcell.backend.priors import sim_ou_prior, sim_ouosc_prior


def plot_rocs_and_timeseries(
    x1: np.ndarray,
    dataNORMED1: np.ndarray,
    FP11: np.ndarray,
    TP11: np.ndarray,
    FP21: np.ndarray,
    TP21: np.ndarray,
    x2: np.ndarray,
    dataNORMED2: np.ndarray,
    FP12: np.ndarray,
    TP12: np.ndarray,
    FP22: np.ndarray,
    TP22: np.ndarray,
    x3: np.ndarray,
    dataNORMED3: np.ndarray,
    FP13: np.ndarray,
    TP13: np.ndarray,
    FP23: np.ndarray,
    TP23: np.ndarray,
    n_cells: int,
) -> None:
    """
    Generates a 3x3 grid of subplots replicating the MATLAB figure.

    Subplots:
      1. (A) Experiment 1, first time series
      2. (B) Experiment 1, second time series
      3. (C) ROC curves for experiment 1.
      4. (D) Experiment 2, first time series
      5. (E) Experiment 2, second time series
      6. (F) ROC curves for experiment 2.
      7. (G) Experiment 3, first time series
      8. (H) Experiment 3, second time series
      9. (I) ROC curves for experiment 3.
    """

    # --- Debug / Diagnostic prints ---
    print("=== Debug Info: Checking shapes and partial data ===")
    print("dataNORMED1 shape:", dataNORMED1.shape)
    print("dataNORMED2 shape:", dataNORMED2.shape)
    print("dataNORMED3 shape:", dataNORMED3.shape)

    print("FP11[:5]:", FP11[:5], "TP11[:5]:", TP11[:5])
    print("FP21[:5]:", FP21[:5], "TP21[:5]:", TP21[:5])
    print("FP12[:5]:", FP12[:5], "TP12[:5]:", TP12[:5])
    print("FP22[:5]:", FP22[:5], "TP22[:5]:", TP22[:5])
    print("FP13[:5]:", FP13[:5], "TP13[:5]:", TP13[:5])
    print("FP23[:5]:", FP23[:5], "TP23[:5]:", TP23[:5])
    print("===================================================")

    # Before plotting, you can do some checks:
    #  - For example, dataNORMEDX.shape[1] should be >= 2 to plot columns 0 and 1
    #  - If shapes are invalid, you can raise an error or skip those subplots.
    if dataNORMED1.shape[1] < 2:
        print(
            "Warning: dataNORMED1 has fewer than 2 columns, skipping second trace plot."
        )
    if dataNORMED2.shape[1] < 2:
        print(
            "Warning: dataNORMED2 has fewer than 2 columns, skipping second trace plot."
        )
    if dataNORMED3.shape[1] < 2:
        print(
            "Warning: dataNORMED3 has fewer than 2 columns, skipping second trace plot."
        )

    plt.figure(figsize=(15, 15))

    # Subplot A: Experiment 1, first cell timeseries.
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(x1, dataNORMED1[:, 0], label="Trace 1")
    ax1.set_xlim(0, np.max(x1))
    ax1.set_ylim([-4, 4])  # type: ignore
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Normalised expression")
    ax1.text(-15, 1, r"$\sigma_N^{2} = 0.1$", fontsize=10)
    ax1.text(-15, -1, "25 hours", fontsize=10)
    xlims = ax1.get_xlim()
    ylims = ax1.get_ylim()
    ax1.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "A",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    # Subplot B: Experiment 1, second cell timeseries.
    # Only plot if we have at least 2 columns:
    ax2 = plt.subplot(3, 3, 2)
    if dataNORMED1.shape[1] >= 2:
        ax2.plot(x1, dataNORMED1[:, 1], label="Trace 2")
    else:
        ax2.text(0.5, 0.5, "Not enough columns", ha="center", va="center")
    ax2.set_xlim(0, np.max(x1))
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Normalised expression")
    xlims = ax2.get_xlim()
    ylims = ax2.get_ylim()
    ax2.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "B",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    # Subplot C: Experiment 1 ROC curves.
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(FP11 / n_cells, TP11 / n_cells, color="r", label="GP")
    ax3.plot(FP21, TP21, color="b", label="L-S")
    ax3.set_xlabel("1 - Specificity (false positive rate)")
    ax3.set_ylabel("Sensitivity (true positive rate)")
    ax3.legend(loc="lower right")
    xlims = ax3.get_xlim()
    ylims = ax3.get_ylim()
    ax3.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "C",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    # Subplot D: Experiment 2, first timeseries.
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(x2, dataNORMED2[:, 0], label="Trace 1")
    ax4.set_xlim(0, np.max(x2))
    ax4.set_ylim([-4, 4])  # type: ignore
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Normalised expression")
    ax4.text(-15, 1, r"$\sigma_N^{2} = 0.5$", fontsize=10)
    ax4.text(-15, -1, "25 hours", fontsize=10)
    xlims = ax4.get_xlim()
    ylims = ax4.get_ylim()
    ax4.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "D",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    # Subplot E: Experiment 2, second timeseries.
    ax5 = plt.subplot(3, 3, 5)
    if dataNORMED2.shape[1] >= 2:
        ax5.plot(x2, dataNORMED2[:, 1], label="Trace 2")
    else:
        ax5.text(0.5, 0.5, "Not enough columns", ha="center", va="center")
    ax5.set_xlim(0, np.max(x2))
    ax5.set_xlabel("Time (hours)")
    ax5.set_ylabel("Normalised expression")
    xlims = ax5.get_xlim()
    ylims = ax5.get_ylim()
    ax5.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "E",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    # Subplot F: Experiment 2 ROC curves.
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(FP12 / n_cells, TP12 / n_cells, color="r", label="GP")
    ax6.plot(FP22, TP22, color="b", label="L-S")
    ax6.set_xlabel("1 - Specificity (false positive rate)")
    ax6.set_ylabel("Sensitivity (true positive rate)")
    ax6.legend(loc="lower right")
    xlims = ax6.get_xlim()
    ylims = ax6.get_ylim()
    ax6.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "F",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    # Subplot G: Experiment 3, first timeseries.
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(x3, dataNORMED3[:, 0], label="Trace 1")
    ax7.set_xlim(0, np.max(x3))
    ax7.set_ylim([-4, 4])  # type: ignore
    ax7.set_xlabel("Time (hours)")
    ax7.set_ylabel("Normalised expression")
    ax7.text(-15 / 2.5, 1, r"$\sigma_N^{2} = 0.1$", fontsize=10)
    ax7.text(-15 / 2.5, -1, "10 hours", fontsize=10)
    xlims = ax7.get_xlim()
    ylims = ax7.get_ylim()
    ax7.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "G",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    # Subplot H: Experiment 3, second timeseries.
    ax8 = plt.subplot(3, 3, 8)
    if dataNORMED3.shape[1] >= 2:
        ax8.plot(x3, dataNORMED3[:, 1], label="Trace 2")
    else:
        ax8.text(0.5, 0.5, "Not enough columns", ha="center", va="center")
    ax8.set_xlim(0, np.max(x3))
    ax8.set_xlabel("Time (hours)")
    ax8.set_ylabel("Normalised expression")
    xlims = ax8.get_xlim()
    ylims = ax8.get_ylim()
    ax8.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "H",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    # Subplot I: Experiment 3 ROC curves.
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(FP13 / n_cells, TP13 / n_cells, color="r", label="GP")
    ax9.plot(FP23, TP23, color="b", label="L-S")
    ax9.set_xlabel("1 - Specificity (false positive rate)")
    ax9.set_ylabel("Sensitivity (true positive rate)")
    ax9.legend(loc="lower right")
    xlims = ax9.get_xlim()
    ylims = ax9.get_ylim()
    ax9.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "I",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )

    plt.tight_layout()
    plt.show()


def compute_rocs_from_file(
    filename: str, noise: float, n_cells: int, joblib: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads simulated data from a CSV file and computes ROC curves based on
    (1) BIC differences computed by get_bic_diff, and
    (2) a Lomb Scargle beat-detection method.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the simulated data.
        The CSV is assumed to have a "Time" column and 2*n_cells "Cell" columns.
    noise : float
        Noise level used in simulation (e.g. np.sqrt(0.1)).
    n_cells : int
        Number of cells per condition (so total columns = 2*n_cells).
    joblib : bool, optional
        Whether to use joblib for parallel processing, by default False.

    Returns
    -------
    FP1 : ndarray
        Array of false-positive counts (BIC method).
    TP1 : ndarray
        Array of true-positive counts (BIC method).
    FP2 : ndarray
        Array of false-positive counts (Lomb Scargle method).
    TP2 : ndarray
        Array of true-positive counts (Lomb Scargle method).
    """
    # --- Read the generated data  --- #
    x, y_list = load_sim(filename)
    total_columns = len(y_list)

    if total_columns != 2 * n_cells:
        raise ValueError(
            f"Expected 2*n_cells = {2 * n_cells} columns, but CSV has {total_columns} columns."
        )

    # This detector will perform the model fitting and return BIC differences.
    params = {
        "verbose": True,
        "joblib": joblib,
        "plots": ["BIC"],
        "set_noise": noise,
        "detrend": False,
        "ou_prior_gen": sim_ou_prior,
        "ouosc_prior_gen": sim_ouosc_prior,
    }
    od = OscillatorDetector.from_file(filename, "Time", "", "Cell", params=params)
    od.fit("BIC")
    BICdiffM = od.BIC_diffs  # list or array with length equal to total_columns
    BICdiffTOT = np.array(BICdiffM)

    # --- ROC analysis using BIC differences --- #

    # Split BIC differences into two groups: A (first n_cells) and B (next n_cells)
    A = BICdiffTOT[:n_cells]
    B = BICdiffTOT[n_cells : 2 * n_cells]

    # Define a threshold vector spanning a little below the min to a little above the max.
    thresh = np.linspace(BICdiffTOT.min() - 1, BICdiffTOT.max() + 1, 200)
    FP1 = np.zeros_like(thresh)
    TP1 = np.zeros_like(thresh)
    for i, th in enumerate(thresh):
        FP1[i] = np.sum(A > th)
        TP1[i] = np.sum(B > th)

    # --- ROC analysis using Lomb–Scargle beat detection ---#

    thrvec = np.exp(np.linspace(-15, 0, 200))
    n_thr = len(thrvec)
    n_series = len(y_list)

    # Preallocate a matrix to store detection results:
    # rows: time series; columns: thresholds
    beatmat = np.empty((n_series, n_thr), dtype=int)

    # Loop once over all time series to compute their periodograms
    for i, y in enumerate(y_list):
        ls = LombScargle(x, y)
        frequency, power = ls.autopower(normalization="standard")
        # For each threshold, compute false alarm level and determine detection
        for j, thr in enumerate(thrvec):
            pth = ls.false_alarm_level(thr)
            beatmat[i, j] = int(np.any(power > pth))

    # Compute False Positives (FP) and True Positives (TP) vectorized over thresholds.
    # Note: following the original code's convention:
    #   - First half of y_list are considered oscillatory signals.
    #   - Second half are non-oscillatory.
    FP2 = (
        np.sum(beatmat[:n_cells, :], axis=0) / n_cells
    )  # fraction detected among first half
    TP2 = (
        np.sum(beatmat[n_cells:, :], axis=0) / n_cells
    )  # fraction detected among second half

    return FP1, TP1, FP2, TP2
