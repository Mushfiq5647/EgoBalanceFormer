#!/usr/bin/env python3
"""
Plot predicted vs ground-truth CoP trajectories for a selected sequence.

Usage:
  python analysis/plot_sequence_forecast.py \
    --results_json log/kfold_subjectwise/fold_01/test_results.json \
    --sequence 10-amanda-trial-1-cl \
    --output_png log/kfold_subjectwise/fold_01/forecast_plot.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    # These plots are placed three-across in the paper, so the source
    # typography must be substantially larger to remain legible after
    # LaTeX scales each panel down.
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_json", required=True, help="Path to test_results.json from balance_prediction/test.py")
    p.add_argument(
        "--sequence",
        default=None,
        help="Sequence selector: full path or basename substring (e.g., '5-karen-trial-1-cm')",
    )
    p.add_argument("--output_png", required=True, help="Output plot path")
    p.add_argument("--title", default=None, help="Optional figure title prefix")
    p.add_argument("--title_exact", default=None,
                   help="If set, use this exact figure title (no auto-added suffix)")
    return p.parse_args()


def _pick_keys(row: Dict) -> Tuple[str, str, str, str, str]:
    # Prefer absolute if present; otherwise fallback to direct or deviation keys.
    if {"pred_abs_CoPX", "pred_abs_CoPY", "gt_abs_CoPX", "gt_abs_CoPY"}.issubset(row.keys()):
        return "pred_abs_CoPX", "pred_abs_CoPY", "gt_abs_CoPX", "gt_abs_CoPY", "Absolute"
    if {"pred_CoPX", "pred_CoPY", "gt_CoPX", "gt_CoPY"}.issubset(row.keys()):
        return "pred_CoPX", "pred_CoPY", "gt_CoPX", "gt_CoPY", "Absolute"
    if {"pred_dev_CoPX", "pred_dev_CoPY", "gt_dev_CoPX", "gt_dev_CoPY"}.issubset(row.keys()):
        return "pred_dev_CoPX", "pred_dev_CoPY", "gt_dev_CoPX", "gt_dev_CoPY", "Deviation"
    raise ValueError("Could not find compatible CoP keys in per_sample rows.")


def main() -> None:
    args = parse_args()
    with open(args.results_json, "r") as f:
        data = json.load(f)

    per_sample: List[Dict] = data.get("per_sample", [])
    if not per_sample:
        raise ValueError(f"No per_sample entries in {args.results_json}")

    # Filter rows for requested sequence.
    seq_sel = args.sequence
    if seq_sel is None:
        rows = per_sample
    else:
        rows = [
            r for r in per_sample
            if seq_sel in str(r.get("sequence_dir", "")) or os.path.basename(str(r.get("sequence_dir", ""))) == seq_sel
        ]
        if not rows:
            raise ValueError(f"No samples matched sequence selector '{seq_sel}'")

    pred_x_key, pred_y_key, gt_x_key, gt_y_key, value_mode = _pick_keys(rows[0])

    pred_x = np.array([float(r[pred_x_key]) for r in rows], dtype=np.float32)
    pred_y = np.array([float(r[pred_y_key]) for r in rows], dtype=np.float32)
    gt_x = np.array([float(r[gt_x_key]) for r in rows], dtype=np.float32)
    gt_y = np.array([float(r[gt_y_key]) for r in rows], dtype=np.float32)

    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(rows))

    actual_line, = axes[0].plot(x, gt_x, label="Actual", linewidth=1.2)
    predicted_line, = axes[0].plot(x, pred_x, label="Predicted", linewidth=1.2, alpha=0.9)
    axes[0].set_ylabel("CoPX (cm)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, gt_y, label="Actual", linewidth=1.2)
    axes[1].plot(x, pred_y, label="Predicted", linewidth=1.2, alpha=0.9)
    axes[1].set_ylabel("CoPY (cm)")
    axes[1].set_xlabel("Sequential prediction index")
    axes[1].grid(True, alpha=0.3)

    seq_name = "ALL_TEST_SEQUENCES" if seq_sel is None else os.path.basename(str(rows[0].get("sequence_dir", seq_sel)))
    if args.title_exact is not None:
        fig.suptitle(args.title_exact, fontsize=30)
    else:
        title_prefix = args.title if args.title else "Actual vs Predicted CoP"
        fig.suptitle(f"{title_prefix} | {seq_name} | mode={value_mode}", fontsize=30)
    fig.legend(
        [actual_line, predicted_line],
        ["Actual", "Predicted"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        fontsize=22,
    )
    # Keep the two trajectory panels large while leaving clear space for the
    # title and shared legend above them.
    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.14,
        top=0.76,
        hspace=0.38,
    )
    fig.savefig(args.output_png, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {args.output_png}")
    print(f"Matched samples: {len(rows)}")
    print(f"Sequence: {seq_name}")


if __name__ == "__main__":
    main()
