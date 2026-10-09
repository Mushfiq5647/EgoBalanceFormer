#!/usr/bin/env python3
"""
Subject-wise K-fold cross-validation runner for balance prediction.

Reads a sequence list file (e.g. splits/train_subjectwise_all.txt),
groups sequences by subject, creates K folds at subject level, and for each fold:
  - trains on K-1 subject folds
  - tests on the held-out subject fold
Then aggregates metrics across folds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _subject_key_from_seq(seq: str) -> str:
    """
    Extract subject key from sequence basename.
    Example:
      10-amanda-trial-5-sh -> 10-amanda
      3-akter-trial-1-ch   -> 3-akter
    """
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


def _latest_epoch_checkpoint(log_dir: str) -> str | None:
    """
    Return latest epoch_XXXX.pth checkpoint by numeric epoch, or None.
    """
    p = Path(log_dir)
    cands = []
    for ck in p.glob("epoch_*.pth"):
        stem = ck.stem  # epoch_XXXX
        try:
            ep = int(stem.split("_")[1])
        except Exception:
            continue
        cands.append((ep, str(ck)))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]


def _checkpoint_path(log_dir: str, test_checkpoint: str, epochs: int) -> str:
    best_path = os.path.join(log_dir, "best_model.pth")
    final_path = os.path.join(log_dir, f"epoch_{epochs:04d}.pth")
    if test_checkpoint == "best":
        return best_path
    if test_checkpoint == "final":
        return final_path
    # auto:
    # - prefer best model if available
    # - else use latest saved epoch_XXXX.pth
    # - else fallback to final epoch path
    if os.path.exists(best_path):
        return best_path
    latest = _latest_epoch_checkpoint(log_dir)
    if latest is not None:
        return latest
    return final_path


def _run(cmd: List[str]) -> int:
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd).returncode


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

    train_cmd = [
        "python", "balance_prediction/train.py",
        "--train_list", train_list,
        "--val_list", test_list,
        "--dataset_type", args.dataset_type,
        "--model_type", args.model_type,
        "--window_size", str(args.window_size),
        "--last_k_frames", str(args.last_k_frames),
        "--fps", str(args.fps),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--log_dir", fold_log_dir,
        "--save_from_epoch", str(args.save_from_epoch),
        "--loss_type", args.loss_type,
        "--huber_delta", str(args.huber_delta),
        "--corr_alpha", str(args.corr_alpha),
        "--max_offset_ms", str(args.max_offset_ms),
        "--mad_threshold", str(args.mad_threshold),
        "--print_every", str(args.print_every),
        "--early_stop_patience", str(args.early_stop_patience),
    ]

    # Common model/data args
    train_cmd += [
        "--d_joint", str(args.d_joint),
        "--d_model", str(args.d_model),
        "--gru_hidden_size", str(args.gru_hidden_size),
        "--gru_layers", str(args.gru_layers),
        "--n_spatial_layers", str(args.n_spatial_layers),
        "--n_temporal_layers", str(args.n_temporal_layers),
        "--n_fusion_layers", str(args.n_fusion_layers),
        "--n_decoder_layers", str(args.n_decoder_layers),
        "--pose_smooth_window", str(args.pose_smooth_window),
        "--vel_smooth_window", str(args.vel_smooth_window),
        "--dropout", str(args.dropout),
        "--cop_target_mode", args.cop_target_mode,
        "--cop_k", str(args.cop_k),
        "--forecast_horizon", str(args.forecast_horizon),
        "--rotation_source", args.rotation_source,
    ]

    if args.no_vr:
        train_cmd.append("--no_vr")
    if args.no_velocity:
        train_cmd.append("--no_velocity")
    if args.no_joint_groups:
        train_cmd.append("--no_joint_groups")
    if args.no_joint_rotations:
        train_cmd.append("--no_joint_rotations")
    if args.no_com:
        train_cmd.append("--no_com")
    if args.no_pose:
        train_cmd.append("--no_pose")
    if args.include_hmd:
        train_cmd.append("--include_hmd")
    if args.vr_mode is not None:
        train_cmd += ["--vr_mode", args.vr_mode]
    if args.butterworth:
        train_cmd.append("--butterworth")
        train_cmd += ["--bw_cutoff_hz", str(args.bw_cutoff_hz), "--bw_order", str(args.bw_order)]

    # estimated15 args
    if args.dataset_type == "estimated15":
        train_cmd += ["--ae_checkpoint", args.ae_checkpoint]
        train_cmd += ["--heatmap_dir", args.heatmap_dir]
        train_cmd += ["--ae_batch_size", str(args.ae_batch_size)]
        if args.align_to_gt:
            train_cmd.append("--align_to_gt")

    if _run(train_cmd) != 0:
        return {"error": "train_failed", "fold": fold_idx, "test_subjects": test_subjects}

    test_cmd = [
        "python", "balance_prediction/test.py",
        "--test_list", test_list,
        "--dataset_type", args.dataset_type,
        "--model_type", args.model_type,
        "--window_size", str(args.window_size),
        "--last_k_frames", str(args.last_k_frames),
        "--fps", str(args.fps),
        "--batch_size", str(args.batch_size),
        "--checkpoint", _checkpoint_path(
            fold_log_dir,
            args.test_checkpoint,
            args.epochs,
        ),
        "--normalizer", os.path.join(fold_log_dir, "normalizer.npz"),
        "--output_json", os.path.join(fold_log_dir, "test_results.json"),
        "--d_joint", str(args.d_joint),
        "--d_model", str(args.d_model),
        "--gru_hidden_size", str(args.gru_hidden_size),
        "--gru_layers", str(args.gru_layers),
        "--n_spatial_layers", str(args.n_spatial_layers),
        "--n_temporal_layers", str(args.n_temporal_layers),
        "--n_fusion_layers", str(args.n_fusion_layers),
        "--n_decoder_layers", str(args.n_decoder_layers),
        "--pose_smooth_window", str(args.pose_smooth_window),
        "--vel_smooth_window", str(args.vel_smooth_window),
        "--dropout", str(args.dropout),
        "--max_offset_ms", str(args.max_offset_ms),
        "--mad_threshold", str(args.mad_threshold),
        "--cop_target_mode", args.cop_target_mode,
        "--cop_k", str(args.cop_k),
        "--forecast_horizon", str(args.forecast_horizon),
        "--rotation_source", args.rotation_source,
    ]

    if args.no_vr:
        test_cmd.append("--no_vr")
    if args.no_velocity:
        test_cmd.append("--no_velocity")
    if args.no_joint_groups:
        test_cmd.append("--no_joint_groups")
    if args.no_joint_rotations:
        test_cmd.append("--no_joint_rotations")
    if args.no_com:
        test_cmd.append("--no_com")
    if args.no_pose:
        test_cmd.append("--no_pose")
    if args.include_hmd:
        test_cmd.append("--include_hmd")
    if args.vr_mode is not None:
        test_cmd += ["--vr_mode", args.vr_mode]
    if args.butterworth:
        test_cmd.append("--butterworth")
        test_cmd += ["--bw_cutoff_hz", str(args.bw_cutoff_hz), "--bw_order", str(args.bw_order)]

    if args.dataset_type == "estimated15":
        test_cmd += ["--ae_checkpoint", args.ae_checkpoint]
        test_cmd += ["--heatmap_dir", args.heatmap_dir]
        test_cmd += ["--ae_batch_size", str(args.ae_batch_size)]
        if args.align_to_gt:
            test_cmd.append("--align_to_gt")

    if _run(test_cmd) != 0:
        return {"error": "test_failed", "fold": fold_idx, "test_subjects": test_subjects}

    out_path = os.path.join(fold_log_dir, "test_results.json")
    if not os.path.exists(out_path):
        return {"error": "missing_results", "fold": fold_idx, "test_subjects": test_subjects}

    with open(out_path) as f:
        data = json.load(f)
    data["fold"] = fold_idx
    data["test_subjects"] = test_subjects
    return data


def aggregate(results: List[Dict]) -> Dict:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"valid_folds": 0}

    primary_key = "metrics_absolute" if all("metrics_absolute" in r for r in valid) else "metrics_deviation"
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
    p = argparse.ArgumentParser(description="Subject-wise K-fold runner for balance_prediction/train.py + test.py")
    p.add_argument("--list_file", default="splits/train_subjectwise_all.txt")
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle_subjects", action="store_true")

    p.add_argument("--dataset_type", default="estimated15", choices=["default", "gt19_root", "gt19_root_aligned", "gt19_enhanced", "estimated15"])
    p.add_argument("--model_type", default="st_transformer_enhanced", choices=["transformer", "gru", "gru_enhanced", "lstm", "lstm_enhanced", "linear", "deeptcn", "cnn_lstm", "cnn_lstm_enhanced", "stgcn", "st_transformer", "st_transformer_enhanced"])

    p.add_argument("--window_size", type=int, default=11)
    p.add_argument("--last_k_frames", type=int, default=0,
                   help="Use only last K frames from each window during train/test (0=use all)")
    p.add_argument("--fps", type=float, default=22.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--d_joint", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--gru_hidden_size", type=int, default=64)
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--n_spatial_layers", type=int, default=2)
    p.add_argument("--n_temporal_layers", type=int, default=2)
    p.add_argument("--n_fusion_layers", type=int, default=1)
    p.add_argument("--n_decoder_layers", type=int, default=1)

    p.add_argument("--pose_smooth_window", type=int, default=5)
    p.add_argument("--vel_smooth_window", type=int, default=3)
    p.add_argument("--max_offset_ms", type=float, default=50.0)
    p.add_argument("--mad_threshold", type=float, default=4.0)
    p.add_argument("--cop_target_mode", type=str, default="k4", choices=["k4", "seqmean", "absolute"])
    p.add_argument("--cop_k", type=int, default=10)
    p.add_argument("--forecast_horizon", type=int, default=0,
                   help="Forecast horizon in CoP steps (2 Hz): target at t+H")
    p.add_argument("--rotation_source", type=str, default="gt", choices=["gt", "pose"])

    p.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber", "correlation"])
    p.add_argument("--huber_delta", type=float, default=0.5)
    p.add_argument("--corr_alpha", type=float, default=0.5)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--save_from_epoch", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--test_checkpoint", type=str, default="auto", choices=["auto", "best", "final"])

    p.add_argument("--no_vr", action="store_true")
    p.add_argument("--no_velocity", action="store_true")
    p.add_argument("--no_joint_groups", action="store_true")
    p.add_argument("--no_joint_rotations", action="store_true")
    p.add_argument("--no_com", action="store_true")
    p.add_argument("--no_pose", action="store_true")

    p.add_argument("--include_hmd", action="store_true")
    p.add_argument("--vr_mode", type=str, default="hmd_only", choices=["rel_lr", "hmd_only", "hmd_lr"])

    p.add_argument("--butterworth", action="store_true")
    p.add_argument("--bw_cutoff_hz", type=float, default=3.0)
    p.add_argument("--bw_order", type=int, default=2)

    p.add_argument("--ae_checkpoint", type=str, default="log/egocentric_ae_15j_monocular/epoch_50_net_AutoEncoder.pth")
    p.add_argument("--heatmap_dir", type=str, default="custom_pred_heatmaps")
    p.add_argument("--ae_batch_size", type=int, default=128)
    p.add_argument("--align_to_gt", action="store_true",
                   help="Apply Procrustes alignment to GT for estimated15 (default: enabled)")
    p.set_defaults(align_to_gt=True)

    p.add_argument("--base_log_dir", default="log/kfold_subjectwise")
    p.add_argument("--tmp_dir", default="tmp/kfold_subjectwise")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.last_k_frames < 0:
        raise ValueError("--last_k_frames must be >= 0")
    if args.last_k_frames > args.window_size:
        raise ValueError("--last_k_frames cannot exceed --window_size")
    if args.forecast_horizon < 0:
        raise ValueError("--forecast_horizon must be >= 0")
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
        result = run_fold(i, test_subs, groups, args)
        all_results.append(result)
        if "error" in result:
            raise RuntimeError(
                f"K-fold aborted at fold {i}: {result['error']} "
                f"(test_subjects={','.join(test_subs)})"
            )

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
