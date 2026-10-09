#!/usr/bin/env python3
"""
Report synchronization/data-filter statistics for balance prediction folds.

Computes, per fold and globally:
  - number of synchronized windows
  - MAD removal fraction on CoP
  - max-offset acceptance rate for CoP->pose alignment

This mirrors the dataset alignment logic (timestamp nearest-neighbor + max_offset_ms)
without running pose inference.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _parse_timestamp(ts_str: str) -> float:
    parts = ts_str.replace("-", " ").replace(":", " ").split()
    if len(parts) >= 6:
        h, m, s = int(parts[3]), int(parts[4]), int(parts[5])
        ms = int(parts[6]) if len(parts) > 6 else 0
        return h * 3600.0 + m * 60.0 + s + ms / 1000.0
    return 0.0


def _filter_outliers_mad(data: np.ndarray, threshold: float) -> np.ndarray:
    if len(data) == 0:
        return np.ones(len(data), dtype=bool)
    median = np.median(data, axis=0)
    abs_dev = np.abs(data - median)
    mad = np.median(abs_dev, axis=0)
    z = abs_dev / (1.4826 * mad + 1e-8)
    return np.all(z < threshold, axis=1)


@dataclass
class SeqStats:
    sequence: str
    pose_frames_total: int
    cop_total_raw: int
    cop_after_mad: int
    mad_removed: int
    cop_after_horizon: int
    accepted_offset: int
    rejected_offset: int
    rejected_window_start: int
    synchronized_windows: int


def _load_sequence_stats(
    seq_dir: str,
    heatmap_dir: str,
    window_size: int,
    max_offset_ms: float,
    mad_threshold: float,
    forecast_horizon: int,
) -> SeqStats:
    gt_path = os.path.join(seq_dir, "ground_truth.json")
    bal_path = os.path.join(seq_dir, "balance.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing ground_truth.json: {seq_dir}")
    if not os.path.exists(bal_path):
        raise FileNotFoundError(f"Missing balance.json: {seq_dir}")

    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    if not isinstance(gt_data, list):
        raise ValueError(f"Expected list in {gt_path}")

    # heatmaps are required for estimated15 pose alignment
    pattern_1 = os.path.join(seq_dir, heatmap_dir, "*_hm15x64x64.npy")
    hm_files = sorted(glob.glob(pattern_1))
    if not hm_files:
        seq_base = os.path.basename(seq_dir.rstrip("/"))
        pattern_2 = os.path.join(heatmap_dir, f"{seq_base}/*_hm15x64x64.npy")
        hm_files = sorted(glob.glob(pattern_2))
    if not hm_files:
        raise FileNotFoundError(f"No heatmaps found for {seq_dir}")

    gt_by_name = {item.get("image_name"): item for item in gt_data if "image_name" in item}
    pose_ts_list: List[float] = []
    for hm in hm_files:
        stem = os.path.basename(hm).replace("_hm15x64x64.npy", "")
        gt_item = gt_by_name.get(stem)
        if gt_item is None:
            continue
        ts_str = gt_item.get("timestamp", "")
        if ts_str:
            pose_ts_list.append(_parse_timestamp(ts_str))
    pose_ts = np.array(pose_ts_list, dtype=np.float64)
    if len(pose_ts) < window_size:
        raise ValueError(f"Not enough pose frames for {seq_dir}: {len(pose_ts)}")

    with open(bal_path, "r") as f:
        bal_data = json.load(f)
    if not isinstance(bal_data, list):
        raise ValueError(f"Expected list in {bal_path}")

    cop_vals: List[List[float]] = []
    cop_ts_list: List[float] = []
    for row in bal_data:
        ts_str = row.get("timestamp", "")
        if not ts_str:
            continue
        cop_vals.append([row.get("copX", 0.0), row.get("copY", 0.0)])
        cop_ts_list.append(_parse_timestamp(ts_str))

    cop = np.array(cop_vals, dtype=np.float32)
    cop_ts = np.array(cop_ts_list, dtype=np.float64)
    cop_total_raw = len(cop)

    if mad_threshold > 0 and len(cop) > 0:
        mask = _filter_outliers_mad(cop, mad_threshold)
        cop = cop[mask]
        cop_ts = cop_ts[mask]
        mad_removed = int((~mask).sum())
    else:
        mad_removed = 0

    cop_after_mad = len(cop)
    max_offset_s = max_offset_ms / 1000.0

    accepted_offset = 0
    rejected_offset = 0
    rejected_window_start = 0
    synchronized = 0

    # horizon means target index is i + H; if unavailable skip sample
    cop_after_horizon = max(0, cop_after_mad - forecast_horizon)

    for i in range(cop_after_horizon):
        t_cop = cop_ts[i]
        insert = int(np.searchsorted(pose_ts, t_cop))
        cands: List[int] = []
        if insert < len(pose_ts):
            cands.append(insert)
        if insert > 0:
            cands.append(insert - 1)
        if not cands:
            rejected_offset += 1
            continue

        end_idx = min(cands, key=lambda j: abs(pose_ts[j] - t_cop))
        offset_s = abs(float(pose_ts[end_idx] - t_cop))
        if offset_s > max_offset_s:
            rejected_offset += 1
            continue

        accepted_offset += 1
        start_idx = end_idx - window_size + 1
        if start_idx < 0:
            rejected_window_start += 1
            continue

        synchronized += 1

    return SeqStats(
        sequence=os.path.basename(seq_dir.rstrip("/")),
        pose_frames_total=int(len(pose_ts)),
        cop_total_raw=int(cop_total_raw),
        cop_after_mad=int(cop_after_mad),
        mad_removed=int(mad_removed),
        cop_after_horizon=int(cop_after_horizon),
        accepted_offset=int(accepted_offset),
        rejected_offset=int(rejected_offset),
        rejected_window_start=int(rejected_window_start),
        synchronized_windows=int(synchronized),
    )


def _read_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _aggregate(seq_stats: List[SeqStats]) -> Dict[str, float]:
    cop_raw = sum(s.cop_total_raw for s in seq_stats)
    cop_after_mad = sum(s.cop_after_mad for s in seq_stats)
    mad_removed = sum(s.mad_removed for s in seq_stats)
    cop_after_h = sum(s.cop_after_horizon for s in seq_stats)
    accepted = sum(s.accepted_offset for s in seq_stats)
    rejected_off = sum(s.rejected_offset for s in seq_stats)
    rej_win = sum(s.rejected_window_start for s in seq_stats)
    sync = sum(s.synchronized_windows for s in seq_stats)

    return {
        "sequences": len(seq_stats),
        "cop_total_raw": cop_raw,
        "cop_after_mad": cop_after_mad,
        "mad_removed": mad_removed,
        "mad_removed_fraction_pct": 100.0 * mad_removed / max(cop_raw, 1),
        "cop_after_horizon": cop_after_h,
        "offset_acceptance_rate_pct": 100.0 * accepted / max(cop_after_h, 1),
        "offset_rejected": rejected_off,
        "window_start_rejected": rej_win,
        "synchronized_windows": sync,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fold_lists_glob", required=True,
                   help="Glob for fold list files, e.g. tmp/kfold_subjectwise/fold_*_test.txt")
    p.add_argument("--heatmap_dir", default="custom_pred_heatmaps")
    p.add_argument("--window_size", type=int, default=11)
    p.add_argument("--max_offset_ms", type=float, default=50.0)
    p.add_argument("--mad_threshold", type=float, default=4.0)
    p.add_argument("--forecast_horizon", type=int, default=0)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fold_files = sorted(glob.glob(args.fold_lists_glob))
    if not fold_files:
        raise ValueError(f"No fold list files found: {args.fold_lists_glob}")

    out: Dict[str, object] = {
        "args": vars(args),
        "folds": {},
    }
    all_seq_stats: List[SeqStats] = []

    for ff in fold_files:
        fold_name = os.path.splitext(os.path.basename(ff))[0]
        seqs = _read_list(ff)
        seq_stats: List[SeqStats] = []
        errors: List[str] = []
        for s in seqs:
            try:
                st = _load_sequence_stats(
                    seq_dir=s,
                    heatmap_dir=args.heatmap_dir,
                    window_size=args.window_size,
                    max_offset_ms=args.max_offset_ms,
                    mad_threshold=args.mad_threshold,
                    forecast_horizon=args.forecast_horizon,
                )
                seq_stats.append(st)
                all_seq_stats.append(st)
            except Exception as exc:
                errors.append(f"{os.path.basename(s)}: {exc}")

        fold_agg = _aggregate(seq_stats) if seq_stats else {}
        out["folds"][fold_name] = {
            "aggregate": fold_agg,
            "n_errors": len(errors),
            "errors": errors[:20],  # cap
        }

    out["global"] = _aggregate(all_seq_stats)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {args.output_json}")
    print(json.dumps(out["global"], indent=2))


if __name__ == "__main__":
    main()

