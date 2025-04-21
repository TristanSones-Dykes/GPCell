# Standard Library Imports
from typing import List, Optional, Sequence, Tuple

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

# Direct Namespace Imports
from astropy.timeseries import LombScargle
from matplotlib import axes
from sklearn.metrics import roc_curve, auc

# Internal Project Imports
from gpcell import OscillatorDetector
from gpcell.utils import load_sim
from gpcell.backend import sim_ou_prior, sim_ouosc_prior


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
    colour_order: List[str] = ["r", "b"],
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

    # Plot ROC curves for experiment 1
    plot_roc(
        [FP11, FP21],
        [TP11, TP21],
        labels=["GP", "L-S"],
        axes=plt.subplot(3, 3, 3),
        colour_order=colour_order,
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

    # Plot ROC curves for experiment 2
    plot_roc(
        [FP12, FP22],
        [TP12, TP22],
        labels=["GP", "L-S"],
        axes=plt.subplot(3, 3, 6),
        colour_order=colour_order,
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

    # Plot ROC curves for experiment 3
    plot_roc(
        [FP13, FP23],
        [TP13, TP23],
        labels=["GP", "L-S"],
        axes=plt.subplot(3, 3, 9),
        colour_order=colour_order,
    )

    plt.tight_layout()
    plt.show()


def plot_roc(
    FP: Sequence[np.ndarray] | np.ndarray,
    TP: Sequence[np.ndarray] | np.ndarray,
    labels: Optional[Sequence[str]] = None,
    axes: Optional[axes.Axes] = None,
    colour_order: List[str] = ["r", "b"],
):
    """
    Plots ROC curves for multiple models.

    Parameters
    ----------
    FP : Sequence[np.ndarray] | np.ndarray
        False positive rates for each model.
    TP : Sequence[np.ndarray] | np.ndarray
        True positive rates for each model.
    labels : Optional[Sequence[str]], optional
        List of labels for each model, by default None.
    axes : Optional[axes.Axes], optional
        Matplotlib axes to plot on, by default None.
    colour_order : List[str], optional
        List of colors for each model, by default ["r", "b"].
    """
    # axes IO
    if axes is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = axes

    # Check if FP and TP are lists or arrays
    if isinstance(FP, np.ndarray):
        FP = [FP]
    if isinstance(TP, np.ndarray):
        TP = [TP]
    if len(FP) != len(TP):
        raise ValueError(f"FP and TP must have the same length: {FP} vs {TP}")

    # Plot ROC curves
    for i, (FP, TP) in enumerate(zip(FP, TP)):
        # calculate AUC
        auc_value = auc(FP, TP)

        # set legend label
        if labels is not None:
            label = labels[i] + f" (AUC = {auc_value:.2f})"
        else:
            label = f"Model {i + 1} (AUC = {auc_value:.2f})"

        # set colours
        if i < len(colour_order):
            colour = colour_order[i]
        else:
            colour = None

        ax.plot(FP, TP, label=label, color=colour)

    ax.set_xlabel("1 - Specificity (false positive rate)")
    ax.set_ylabel("Sensitivity (true positive rate)")
    ax.legend(loc="lower right")
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.text(
        xlims[0] - 0.1 * (xlims[1] - xlims[0]),
        ylims[1] + 0.03 * (ylims[1] - ylims[0]),
        "C",
        fontsize=9,
        ha="right",
        va="bottom",
        color="k",
    )


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

    # Normalize FP and TP for the first method
    FP1 /= n_cells
    TP1 /= n_cells

    # --- ROC analysis using Lomb–Scargle beat detection ---#
    # Compute ROC curves using Lomb-Scargle method
    FP2, TP2 = compute_LS_roc(x, y_list, n_cells)

    return FP1, TP1, FP2, TP2


def compute_LS_roc(
    x: np.ndarray, y_list: Sequence[np.ndarray], n_cells
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Lomb-Scargle ROC curve.

    Parameters
    ----------
    x : np.ndarray
        Time values.
    y_list : list of np.ndarray
        List of time series data for each cell.
    n_cells : int
        Number of cells.

    Returns
    -------
    FP : list
        List of false positive rates.
    TP : list
        List of true positive rates.
    """
    # Initialize lists to store false positives and true positives
    FP = []
    TP = []

    true_labels = np.array([0] * n_cells + [1] * n_cells)
    max_scores = np.zeros((2 * n_cells,))

    # Loop through each cell's data
    for i in range(2 * n_cells):
        # Extract the time series data for the current cell
        y = y_list[i]

        # Calculate the Lomb-Scargle periodogram
        frequency, power = LombScargle(x, y).autopower()
        max_score = np.max(power)

        # Add the maximum score to the list
        max_scores[i] = max_score

    # Calculate the ROC curve
    FP, TP, _ = roc_curve(true_labels, max_scores)

    return FP, TP
