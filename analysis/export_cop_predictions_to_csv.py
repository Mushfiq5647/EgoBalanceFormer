"""
Export CoP predictions to per-subject CSV files.

Reads the JSON output produced by balance_prediction/test.py (with --output_json)
and writes one CSV per subject (sequence directory basename), containing:

  index, pred_CoPX, pred_CoPY, gt_CoPX, gt_CoPY

Usage:
  cd <repo root>
  python analysis/export_cop_predictions_to_csv.py \
      --results_json log/cop_st_transformer/test_results_epoch50.json

The script only uses the "per_sample" list from the JSON. It assumes that
all entries correspond to sequences listed in utils/test.txt (i.e., test set).
"""

import argparse
import json
import os
import csv
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_json",
        required=True,
        help="Path to test_results.json produced by balance_prediction/test.py",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write CSVs to (default: same dir as results_json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.results_json
    with open(results_path) as f:
        data = json.load(f)

    per_sample = data.get("per_sample")
    if per_sample is None:
        raise ValueError(f"No 'per_sample' key found in {results_path}")

    # Group by subject (basename of sequence_dir)
    groups = defaultdict(list)
    for i, row in enumerate(per_sample):
        seq_dir = row.get("sequence_dir", "")
        subj = os.path.basename(seq_dir.rstrip("/"))
        groups[subj].append(
            {
                "index": i,
                "pred_CoPX": float(row["pred_CoPX"]),
                "pred_CoPY": float(row["pred_CoPY"]),
                "gt_CoPX": float(row["gt_CoPX"]),
                "gt_CoPY": float(row["gt_CoPY"]),
            }
        )

    out_dir = args.output_dir or os.path.dirname(results_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    for subj, rows in groups.items():
        csv_path = os.path.join(out_dir, f"{subj}_cop_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "pred_CoPX", "pred_CoPY", "gt_CoPX", "gt_CoPY"],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()

