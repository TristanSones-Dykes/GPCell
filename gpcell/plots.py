# Standard Library Imports
from typing import (
    Tuple,
)

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Direct Namespace Imports
from astropy.timeseries import LombScargle

# Internal Project Imports
from gpcell import OscillatorDetector
from gpcell.backend import Numeric


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
    CellNum: int,
) -> None:
    """
    Generates a 3x3 grid of subplots replicating the MATLAB figure.

    Subplots:
      1. (A) Experiment 1, first time series (from dataNORMED1[:,0]).
      2. (B) Experiment 1, second time series (from dataNORMED1[:,1]).
      3. (C) ROC curves for experiment 1: red = GP (BIC-based), blue = L-S.
      4. (D) Experiment 2, first time series (from dataNORMED2[:,0]).
      5. (E) Experiment 2, second time series (from dataNORMED2[:,1]).
      6. (F) ROC curves for experiment 2.
      7. (G) Experiment 3, first time series (from dataNORMED3[:,0]).
      8. (H) Experiment 3, second time series (from dataNORMED3[:,1]).
      9. (I) ROC curves for experiment 3.
    """
    plt.figure(figsize=(15, 15))

    # Subplot A: Experiment 1, first cell timeseries.
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(x1, dataNORMED1[:, 0])
    ax1.set_xlim(0, np.max(x1))
    ax1.set_ylim([-4, 4])
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
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(x1, dataNORMED1[:, 1])
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
    # Plot GP ROC (BIC-based) in red, L-S ROC in blue.
    ax3.plot(FP11 / CellNum, TP11 / CellNum, color="r", label="GP")
    ax3.plot(FP21 / CellNum, TP21 / CellNum, color="b", label="L-S")
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
    ax4.plot(x2, dataNORMED2[:, 0])
    ax4.set_xlim(0, np.max(x2))
    ax4.set_ylim([-4, 4])
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
    ax5.plot(x2, dataNORMED2[:, 1])
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
    ax6.plot(FP12 / CellNum, TP12 / CellNum, color="r", label="GP")
    ax6.plot(FP22 / CellNum, TP22 / CellNum, color="b", label="L-S")
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
    ax7.plot(x3, dataNORMED3[:, 0])
    ax7.set_xlim(0, np.max(x3))
    ax7.set_ylim([-4, 4])
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
    ax8.plot(x3, dataNORMED3[:, 1])
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
    ax9.plot(FP13 / CellNum, TP13 / CellNum, color="r", label="GP")
    ax9.plot(FP23 / CellNum, TP23 / CellNum, color="b", label="L-S")
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
    filename: str, Noise: Numeric, CellNum: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads simulated data from a CSV file and computes ROC curves based on
    (1) BIC differences computed by get_bic_diff, and
    (2) a Lomb Scargle beat-detection method.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the simulated data.
        The CSV is assumed to have a "Time" column and columns named "Cell 1", "Cell 2", etc.
    Noise : float
        Noise level used in simulation (e.g. np.sqrt(0.1)).
    CellNum : int
        Number of replicates per condition (the file should have 2*CellNum cell columns).

    Returns
    -------
    FP1 : ndarray
        Array (length 200) of false-positive counts (BIC method).
    TP1 : ndarray
        Array (length 200) of true-positive counts (BIC method).
    FP2 : ndarray
        Array (length 200) of false-positive counts (Lomb Scargle beat detection).
    TP2 : ndarray
        Array (length 200) of true-positive counts (Lomb Scargle beat detection).
    """
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(filename)

    # Extract the time vector and convert it to a numpy array.
    # Assume the "Time" column is in the CSV.
    x = df["Time"].values  # x is assumed to be already converted (e.g. in hours)

    # Get the simulation data (all columns except "Time")
    dataNORMED = df.drop(columns="Time").values  # shape: (num_time_points, total_cells)
    n_cells = dataNORMED.shape[1]

    # Compute BIC differences using OscillatorDetector.
    od = OscillatorDetector.from_data(filename, "Time", "", "Cell", set_noise=Noise)
    od.fit("BIC")
    BICdiffM = od.BIC_diffs
    BICdiffTOT = np.array(BICdiffM)

    # Split the BIC differences into two groups:
    # First CellNum are one condition (group A, e.g., non-oscillatory)
    # Next CellNum are the other condition (group B, e.g., oscillatory)
    A = BICdiffTOT[:CellNum]
    B = BICdiffTOT[CellNum : 2 * CellNum]

    # --- ROC using BIC differences ---
    # Sweep through a range of thresholds.
    thresh = np.linspace(np.min(BICdiffTOT) - 1, np.max(BICdiffTOT) + 1, 200)
    FP1 = np.zeros_like(thresh)
    TP1 = np.zeros_like(thresh)
    for i, th in enumerate(thresh):
        FP1[i] = np.sum(A > th)  # false positives from group A
        TP1[i] = np.sum(B > th)  # true positives from group B

    # --- ROC using Lomb–Scargle beat detection ---
    # Define an exponential range of thresholds.
    thrvec = np.exp(np.linspace(-15, -0.01, 200))
    FP2 = np.zeros(len(thrvec))
    TP2 = np.zeros(len(thrvec))
    for i, thr in enumerate(thrvec):
        beatvec = np.zeros(n_cells)
        # For each cell, compute the periodogram and check if it exceeds the false-alarm level.
        for j in range(n_cells):
            y1 = dataNORMED[:, j]  # using the original simulated data
            Pd = 1 - thr
            # Define a frequency grid; adjust limits as needed.
            f = np.linspace(0.01, 10, 1000)
            ls = LombScargle(x, y1)
            pxx = ls.power(f, normalization="psd")
            pth = ls.false_alarm_level(Pd)
            beatvec[j] = 1 if np.any(pxx > pth) else 0  # type: ignore
        # Split beat detection results into the two groups.
        A_beat = beatvec[:CellNum]
        B_beat = beatvec[CellNum : 2 * CellNum]
        FP2[i] = np.sum(A_beat)
        TP2[i] = np.sum(B_beat)

    return FP1, TP1, FP2, TP2
