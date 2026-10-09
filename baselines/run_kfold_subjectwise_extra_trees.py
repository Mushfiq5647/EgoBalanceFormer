#!/usr/bin/env python3
"""
Subject-wise K-fold Extra Trees runner for balance prediction.

Uses BalanceDatasetEstimated15 and flattens window features for tree regression.
Trains two independent ExtraTreesRegressor models (CoPX, CoPY).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesRegressor
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is not available. Install it first:\n"
        "  pip install scikit-learn\n"
        f"Import error: {exc}"
    )

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.dataset_estimated15 import BalanceDatasetEstimated15  # noqa: E402
from utils.metrics import compute_cop_metrics  # noqa: E402


def _subject_key_from_seq(seq: str) -> str:
    base = os.path.basename(seq.rstrip("/"))
    parts = base.split("-")
    if len(parts) < 2:
        return base
    return f"{parts[0]}-{parts[1]}"


def load_subject_groups(list_file: str) -> Dict[str, List[str]]:
    with open(list_file) as f:
        seqs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    groups: Dict[str, List[str]] = {}
    for seq in seqs:
        key = _subject_key_from_seq(seq)
        groups.setdefault(key, []).append(seq)
    return groups


def make_subject_folds(subjects: List[str], n_folds: int, seed: int, shuffle: bool) -> List[List[str]]:
    subs = subjects[:]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(subs)
    return [list(x) for x in np.array_split(np.array(subs, dtype=object), n_folds)]


def write_list(path: str, seqs: List[str]) -> None:
    with open(path, "w") as f:
        for s in seqs:
            f.write(s + "\n")


def _active_keys(args: argparse.Namespace, sample: dict) -> List[str]:
    keys: List[str] = []
    if not args.no_pose:
        keys.append("pose_pos")
        if not args.no_velocity and "pose_vel" in sample:
            keys.append("pose_vel")
    if not args.no_joint_groups and "joint_groups" in sample:
        keys.append("joint_groups")
    if not args.no_joint_rotations and "joint_rotations" in sample:
        keys.append("joint_rotations")
        if not args.no_velocity and "joint_rot_vel" in sample:
            keys.append("joint_rot_vel")
    if not args.no_com and "com" in sample:
        keys.append("com")
        if not args.no_velocity and "com_vel" in sample:
            keys.append("com_vel")
    if not args.no_vr and "vr_pos" in sample:
        keys.append("vr_pos")
        if not args.no_velocity and "vr_vel" in sample:
            keys.append("vr_vel")
    return keys


def _flatten_sample(sample: dict, keys: List[str]) -> np.ndarray:
    parts = []
    for k in keys:
        arr = sample[k].numpy().astype(np.float32).reshape(-1)
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def _dataset_to_xy(ds: BalanceDatasetEstimated15, keys: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, means = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        xs.append(_flatten_sample(s, keys))
        ys.append(s["cop"].numpy().astype(np.float32))
        means.append(s["cop_mean"].numpy().astype(np.float32))
    return np.stack(xs), np.stack(ys), np.stack(means)


def _build_ds(list_file: str, args: argparse.Namespace) -> BalanceDatasetEstimated15:
    return BalanceDatasetEstimated15(
        list_file=list_file,
        window_size=args.window_size,
        fps=args.fps,
        pose_smooth_window=args.pose_smooth_window,
        vel_smooth_window=args.vel_smooth_window,
        max_offset_ms=args.max_offset_ms,
        use_velocity=not args.no_velocity,
        use_vr=not args.no_vr,
        ae_checkpoint=args.ae_checkpoint,
        use_body_part=args.use_body_part,
        heatmap_dir=args.heatmap_dir,
        ae_batch_size=args.ae_batch_size,
        mad_threshold=args.mad_threshold,
        vr_mode=args.vr_mode,
        cop_target_mode=args.cop_target_mode,
        cop_k=args.cop_k,
        butterworth=args.butterworth,
        bw_cutoff_hz=args.bw_cutoff_hz,
        bw_order=args.bw_order,
    )


def run_fold(
    fold_idx: int,
    test_subjects: List[str],
    subject_to_seqs: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict:
    fold_name = f"fold_{fold_idx:02d}"
    fold_log_dir = os.path.join(args.base_log_dir, fold_name)
    os.makedirs(fold_log_dir, exist_ok=True)

    train_subjects = [s for s in subject_to_seqs.keys() if s not in set(test_subjects)]
    train_seqs = [seq for s in train_subjects for seq in subject_to_seqs[s]]
    test_seqs = [seq for s in test_subjects for seq in subject_to_seqs[s]]

    train_list = os.path.join(args.tmp_dir, f"{fold_name}_train.txt")
    test_list = os.path.join(args.tmp_dir, f"{fold_name}_test.txt")
    write_list(train_list, train_seqs)
    write_list(test_list, test_seqs)

    print(
        f"\nFold {fold_idx}/{args.n_folds} | test_subjects={','.join(test_subjects)} "
        f"| train_seqs={len(train_seqs)} test_seqs={len(test_seqs)}"
    )

    train_ds = _build_ds(train_list, args)
    test_ds = _build_ds(test_list, args)
    keys = _active_keys(args, train_ds[0])
    print("  active keys:", keys)

    x_train, y_train, _ = _dataset_to_xy(train_ds, keys)
    x_test, y_test, cop_mean_test = _dataset_to_xy(test_ds, keys)

    et_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model_x = ExtraTreesRegressor(**et_params)
    model_y = ExtraTreesRegressor(**et_params)
    model_x.fit(x_train, y_train[:, 0])
    model_y.fit(x_train, y_train[:, 1])

    pred_dev = np.stack([model_x.predict(x_test), model_y.predict(x_test)], axis=1).astype(np.float32)
    gt_dev = y_test.astype(np.float32)
    metrics_dev = compute_cop_metrics(pred_dev, gt_dev)

    is_absolute = args.cop_target_mode == "absolute"
    if is_absolute:
        metrics_abs = metrics_dev
    else:
        pred_abs = pred_dev + cop_mean_test
        gt_abs = gt_dev + cop_mean_test
        metrics_abs = compute_cop_metrics(pred_abs, gt_abs)

    result = {
        "fold": fold_idx,
        "test_subjects": test_subjects,
        "active_keys": keys,
        "metrics_deviation": metrics_dev,
        "metrics_absolute": metrics_abs,
    }
    with open(os.path.join(fold_log_dir, "test_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def aggregate(results: List[Dict]) -> Dict:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"valid_folds": 0}
    primary_key = "metrics_absolute"
    metric_names = [
        "rmse_x", "rmse_y", "rmse_total",
        "mae_x", "mae_y", "mae_total",
        "r2_x", "r2_y",
        "corr_x", "corr_y", "corr_total",
        "within_10mm", "within_20mm", "within_30mm",
    ]
    agg = {"valid_folds": len(valid), "primary_key": primary_key, "mean_std": {}}
    for m in metric_names:
        vals = [r.get(primary_key, {}).get(m) for r in valid if r.get(primary_key, {}).get(m) is not None]
        if vals:
            agg["mean_std"][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subject-wise K-fold Extra Trees runner")
    p.add_argument("--list_file", default="splits/train_subjectwise_all.txt")
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle_subjects", action="store_true")

    # Dataset settings
    p.add_argument("--window_size", type=int, default=11)
    p.add_argument("--fps", type=float, default=22.0)
    p.add_argument("--pose_smooth_window", type=int, default=5)
    p.add_argument("--vel_smooth_window", type=int, default=3)
    p.add_argument("--max_offset_ms", type=float, default=50.0)
    p.add_argument("--mad_threshold", type=float, default=4.0)
    p.add_argument("--cop_target_mode", type=str, default="k4", choices=["k4", "seqmean", "absolute"])
    p.add_argument("--cop_k", type=int, default=10)
    p.add_argument("--no_vr", action="store_true")
    p.add_argument("--no_pose", action="store_true")
    p.add_argument("--no_velocity", action="store_true")
    p.add_argument("--no_joint_groups", action="store_true")
    p.add_argument("--no_joint_rotations", action="store_true")
    p.add_argument("--no_com", action="store_true")
    p.add_argument("--vr_mode", type=str, default="hmd_only", choices=["rel_lr", "hmd_only", "hmd_lr"])
    p.add_argument("--butterworth", action="store_true")
    p.add_argument("--bw_cutoff_hz", type=float, default=3.0)
    p.add_argument("--bw_order", type=int, default=2)
    p.add_argument("--ae_checkpoint", required=True)
    p.add_argument("--use_body_part", action="store_true")
    p.add_argument("--heatmap_dir", type=str, default="custom_pred_heatmaps")
    p.add_argument("--ae_batch_size", type=int, default=128)

    # ExtraTrees settings
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--max_depth", type=int, default=0, help="0 means unlimited")
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--n_jobs", type=int, default=8)

    p.add_argument("--base_log_dir", default="log/ablation_studies/extra_trees_kfold")
    p.add_argument("--tmp_dir", default="tmp/kfold_subjectwise_extra_trees")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.base_log_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    groups = load_subject_groups(args.list_file)
    subjects = sorted(groups.keys())
    if args.n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if args.n_folds > len(subjects):
        raise ValueError(f"n_folds={args.n_folds} > number of subjects={len(subjects)}")

    folds = make_subject_folds(subjects, args.n_folds, args.seed, args.shuffle_subjects)
    print(f"Loaded {len(subjects)} subjects from {args.list_file}")
    print("Fold subject counts:", [len(f) for f in folds])

    all_results: List[Dict] = []
    for i, test_subs in enumerate(folds, start=1):
        try:
            result = run_fold(i, test_subs, groups, args)
        except Exception as exc:
            result = {"error": str(exc), "fold": i, "test_subjects": test_subs}
            print(f"[fold {i}] ERROR: {exc}")
        all_results.append(result)

    summary = aggregate(all_results)
    print("\nK-fold subject-wise summary:")
    print(json.dumps(summary, indent=2))

    out = {
        "args": vars(args),
        "subjects": subjects,
        "folds": folds,
        "results": all_results,
        "summary": summary,
    }
    out_path = os.path.join(args.base_log_dir, "kfold_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
