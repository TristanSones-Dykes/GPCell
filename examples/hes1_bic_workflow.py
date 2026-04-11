"""Run the canonical Hes1 oscillation-detection workflow with GPCell.

This example uses the maintained functional API:

    load_detector_data(...) -> run_pipeline(...)

The default run fits all 12 bundled Hes1 cell traces with the BIC workflow. Use
``--max-cells`` and ``--bic-replicates`` for a faster smoke check while setting up an
environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "hes" / "Hes1_example.csv"
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="gpflow.versions",
)

from gpcell import DetectorConfig, DetectorData, load_detector_data, run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify bundled Hes1 single-cell traces with GPCell's functional API."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the Hes-style CSV file.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional number of cell traces to analyze for a quick smoke run.",
    )
    parser.add_argument(
        "--bic-replicates",
        type=int,
        default=10,
        help="Replicate OU/OUosc fits per cell for the BIC workflow.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Also run the bootstrap thresholding workflow.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10,
        help="Synthetic traces per cell when --bootstrap is enabled.",
    )
    parser.add_argument(
        "--joblib",
        action="store_true",
        help="Use Joblib parallel fitting.",
    )
    parser.add_argument(
        "--bic-cutoff",
        type=float,
        default=3.0,
        help="BIC-difference cutoff for the simple oscillatory count.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path for a BIC-difference bar plot.",
    )
    return parser.parse_args()


def subset_cells(data: DetectorData, max_cells: int | None) -> DetectorData:
    if max_cells is None:
        return data
    if max_cells < 1:
        raise ValueError("--max-cells must be positive when provided.")
    return DetectorData(
        X=data.X[:max_cells],
        Y=data.Y[:max_cells],
        X_bckgd=data.X_bckgd,
        bckgd=data.bckgd,
    )


def validate_bootstrap_settings(args: argparse.Namespace, data: DetectorData) -> None:
    if not args.bootstrap:
        return
    if data.n_cells < 3:
        raise ValueError(
            "Bootstrap thresholding needs more than a tiny one-cell smoke run. "
            "Use --max-cells 3 or omit --max-cells."
        )
    if args.bootstrap_replicates < 10:
        raise ValueError(
            "Use --bootstrap-replicates 10 or higher, or omit --bootstrap for a fast smoke run."
        )


def print_bic_summary(
    bic_diffs: Sequence[float],
    llrs: Sequence[float],
    periods: Sequence[float],
    cutoff: float,
) -> None:
    print("\nBIC summary")
    print("cell  bic_diff    llr        period_h   oscillatory")
    print("----  ----------  ---------  ---------  -----------")
    for i, (bic_diff, llr, period) in enumerate(
        zip(bic_diffs, llrs, periods, strict=True),
        start=1,
    ):
        print(
            f"{i:>4}  {bic_diff:>10.3f}  {llr:>9.3f}  {period:>9.3f}  {str(bic_diff > cutoff):>11}"
        )

    oscillatory = int(np.sum(np.array(bic_diffs) > cutoff))
    print(f"\nCells above BIC cutoff {cutoff:g}: {oscillatory}/{len(bic_diffs)}")


def save_bic_plot(path: Path, bic_diffs: Sequence[float], cutoff: float) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    cell_ids = np.arange(1, len(bic_diffs) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(cell_ids, bic_diffs, color="#3b7a57")
    ax.axhline(
        cutoff,
        color="#b23a48",
        linestyle="--",
        linewidth=1.5,
        label=f"BIC cutoff {cutoff:g}",
    )
    ax.set_xlabel("Cell")
    ax.set_ylabel("BIC difference")
    ax.set_title("Hes1 BIC oscillation screen")
    ax.set_xticks(cell_ids)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data = load_detector_data(
        path=str(args.data),
        x_name="Time (h)",
        background_name="Background",
        y_name="Cell",
    )
    data = subset_cells(data, args.max_cells)
    validate_bootstrap_settings(args, data)

    config = DetectorConfig(
        joblib=args.joblib,
        bic_replicates=args.bic_replicates,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    methods = ["BIC", "BOOTSTRAP"] if args.bootstrap else ["BIC"]
    result = run_pipeline(data, config, methods=methods)
    if result.bic is None:
        raise RuntimeError("BIC workflow did not produce a result.")

    print(f"Data: {args.data}")
    print(f"Cell traces analyzed: {data.n_cells}")
    if result.prepared.mean_noise is not None:
        print(f"Estimated mean background noise: {result.prepared.mean_noise:.3f}")
    print(f"BIC replicates per model: {args.bic_replicates}")
    print(f"Joblib parallel fitting: {args.joblib}")

    print_bic_summary(
        result.bic.BIC_diffs,
        result.bic.LLRs,
        result.bic.periods,
        args.bic_cutoff,
    )

    if result.bootstrap is not None:
        print(
            "\nBootstrap q-value classification count: "
            f"{int(np.sum(result.bootstrap.osc_filt))}/{len(result.bootstrap.osc_filt)}"
        )

    if args.plot is not None:
        save_bic_plot(args.plot, result.bic.BIC_diffs, args.bic_cutoff)
        print(f"\nSaved BIC summary plot to {args.plot}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
