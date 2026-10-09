#!/usr/bin/env python3
"""Plot ground-truth and predicted CoP trajectories from test results.

Example:
  python -u analysis/plot_cop_trajectories.py \
    --results_json log/cop_est15_k10_hmdonly_nojg_e100/test_results_epoch100.json \
    --output_png log/cop_est15_k10_hmdonly_nojg_e100/copxy_actual_vs_pred_epoch100.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    # The figure is scaled down in the paper, so use larger source text.
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_json",
        required=True,
        help="Path to a test_results JSON containing per_sample predictions.",
    )
    parser.add_argument("--output_png", required=True, help="Output figure path.")
    return parser.parse_args()


def select_keys(row: Dict) -> Tuple[str, str, str, str]:
    key_sets = (
        ("gt_abs_CoPX", "pred_abs_CoPX", "gt_abs_CoPY", "pred_abs_CoPY"),
        ("gt_CoPX", "pred_CoPX", "gt_CoPY", "pred_CoPY"),
        ("gt_dev_CoPX", "pred_dev_CoPX", "gt_dev_CoPY", "pred_dev_CoPY"),
    )
    for keys in key_sets:
        if all(key in row for key in keys):
            return keys
    raise ValueError("No compatible CoP prediction and ground-truth keys were found.")


def main() -> None:
    args = parse_args()
    with open(args.results_json, "r") as file:
        data = json.load(file)

    rows: List[Dict] = data.get("per_sample", [])
    if not rows:
        raise ValueError(f"No per_sample entries found in {args.results_json}")

    gt_x_key, pred_x_key, gt_y_key, pred_y_key = select_keys(rows[0])
    gt_x = [float(row[gt_x_key]) for row in rows]
    pred_x = [float(row[pred_x_key]) for row in rows]
    gt_y = [float(row[gt_y_key]) for row in rows]
    pred_y = [float(row[pred_y_key]) for row in rows]
    indices = range(len(rows))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    actual_line, = axes[0].plot(indices, gt_x, label="Actual", linewidth=0.9)
    predicted_line, = axes[0].plot(indices, pred_x, label="Predicted", linewidth=0.9, alpha=0.85)
    axes[0].set_ylabel("CoPX (cm)")
    axes[0].set_title("Actual vs Predicted CoPX")
    axes[0].grid(alpha=0.2)

    axes[1].plot(indices, gt_y, label="Actual", linewidth=0.9)
    axes[1].plot(indices, pred_y, label="Predicted", linewidth=0.9, alpha=0.85)
    axes[1].set_ylabel("CoPY (cm)")
    axes[1].set_xlabel("Sequential prediction index")
    axes[1].set_title("Actual vs Predicted CoPY")
    axes[1].grid(alpha=0.2)

    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    fig.legend(
        [actual_line, predicted_line],
        ["Actual", "Predicted"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(args.output_png, dpi=220)
    plt.close(fig)
    print(f"Saved: {args.output_png}")
    print(f"Plotted samples: {len(rows)}")


if __name__ == "__main__":
    main()
