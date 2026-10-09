#!/usr/bin/env python3
"""
Leave-One-Person-Out (LOPO) Cross-Validation for Balance Prediction

This script automatically:
1. Extracts unique subjects from utils/all_sequence.txt
2. For each subject (fold):
   - Test: all trials from that subject
   - Train: all trials from remaining subjects
   - Runs train.py → test.py
3. Aggregates results across all folds

Usage:
    cd <repo root>
    python balance_prediction/run_lopo.py --dataset_type gt19_enhanced --model_type st_transformer_enhanced --no_vr
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _get_test_checkpoint_path(fold_log_dir: str, test_checkpoint: str, epochs: int) -> str:
    """Get the checkpoint path for testing."""
    if test_checkpoint == "best":
        return os.path.join(fold_log_dir, "best_model.pth")
    elif test_checkpoint == "final":
        return os.path.join(fold_log_dir, f"epoch_{epochs:04d}.pth")
    else:
        raise ValueError(f"Unknown test_checkpoint: {test_checkpoint}")


def extract_subjects_and_sequences(all_seq_file: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Extract subjects and group sequences by subject.
    
    Returns:
        subjects: sorted list of subject names
        subject_seqs: dict mapping subject -> list of sequence paths
    """
    with open(all_seq_file) as f:
        sequences = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    subject_seqs = {}
    for seq in sequences:
        basename = os.path.basename(seq)  # e.g. "3-akter-trial-1-ch"
        parts = basename.split("-")
        if len(parts) >= 2:
            subject = parts[1]  # e.g. "akter"
            if subject not in subject_seqs:
                subject_seqs[subject] = []
            subject_seqs[subject].append(seq)
    
    subjects = sorted(subject_seqs.keys())
    return subjects, subject_seqs


def create_fold_files(fold_idx: int, test_subject: str, subject_seqs: Dict[str, List[str]], 
                     tmp_dir: str) -> Tuple[str, str]:
    """Create train/test files for one fold."""
    train_file = os.path.join(tmp_dir, f"train_fold_{fold_idx:02d}_{test_subject}.txt")
    test_file = os.path.join(tmp_dir, f"test_fold_{fold_idx:02d}_{test_subject}.txt")
    
    # Test: all sequences from test_subject
    test_seqs = subject_seqs[test_subject]
    
    # Train: all sequences from other subjects
    train_seqs = []
    for subj, seqs in subject_seqs.items():
        if subj != test_subject:
            train_seqs.extend(seqs)
    
    # Write files
    with open(train_file, "w") as f:
        for seq in train_seqs:
            f.write(seq + "\n")
    
    with open(test_file, "w") as f:
        for seq in test_seqs:
            f.write(seq + "\n")
    
    print(f"Fold {fold_idx:2d} ({test_subject:10s}): {len(train_seqs):3d} train, {len(test_seqs):2d} test")
    return train_file, test_file


def run_train_test(fold_idx: int, test_subject: str, train_file: str, test_file: str, 
                  args: argparse.Namespace) -> Dict:
    """Run train.py + test.py for one fold."""
    fold_log_dir = os.path.join(args.base_log_dir, f"fold_{fold_idx:02d}_{test_subject}")
    os.makedirs(fold_log_dir, exist_ok=True)
    
    # Build train command
    train_cmd = [
        "python", "balance_prediction/train.py",
        "--train_list", train_file,
        "--val_list", test_file,
        "--dataset_type", args.dataset_type,
        "--model_type", args.model_type,
        "--window_size", str(args.window_size),
        "--fps", str(args.fps),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--d_joint", str(args.d_joint),
        "--d_model", str(args.d_model),
        "--n_spatial_layers", str(args.n_spatial_layers),
        "--n_temporal_layers", str(args.n_temporal_layers),
        "--n_fusion_layers", str(args.n_fusion_layers),
        "--n_decoder_layers", str(args.n_decoder_layers),
        "--pose_smooth_window", str(args.pose_smooth_window),
        "--vel_smooth_window", str(args.vel_smooth_window),
        "--save_from_epoch", str(args.save_from_epoch),
        "--log_dir", fold_log_dir,
    ]
    
    # Add optional flags
    if args.no_vr:
        train_cmd.append("--no_vr")
    if args.no_velocity:
        train_cmd.append("--no_velocity")
    if args.no_joint_groups:
        train_cmd.append("--no_joint_groups")
    
    # Loss function args
    train_cmd.extend(["--loss_type", args.loss_type])
    train_cmd.extend(["--huber_delta", str(args.huber_delta)])
    train_cmd.extend(["--corr_alpha", str(args.corr_alpha)])
    
    # MAD filtering
    if args.mad_threshold is not None:
        train_cmd.extend(["--mad_threshold", str(args.mad_threshold)])
    
    # Add dataset-specific args
    if args.dataset_type == "estimated15":
        if args.ae_checkpoint:
            train_cmd.extend(["--ae_checkpoint", args.ae_checkpoint])
        if args.heatmap_dir:
            train_cmd.extend(["--heatmap_dir", args.heatmap_dir])
        if args.align_to_gt:
            train_cmd.append("--align_to_gt")
    
    print(f"  Training fold {fold_idx} ({test_subject})...")
    print(f"    Command: {' '.join(train_cmd)}")
    
    # Stream training output in real-time (show epoch progress, val_rmse, etc.)
    result = subprocess.run(train_cmd, text=True)
    if result.returncode != 0:
        print(f"  ERROR in training fold {fold_idx} (exit code {result.returncode})")
        return {"error": "train_failed"}
    
    # Build test command
    test_cmd = [
        "python", "balance_prediction/test.py",
        "--test_list", test_file,
        "--dataset_type", args.dataset_type,
        "--model_type", args.model_type,
        "--window_size", str(args.window_size),
        "--fps", str(args.fps),
        "--batch_size", str(args.batch_size),
        "--checkpoint", _get_test_checkpoint_path(fold_log_dir, args.test_checkpoint, args.epochs),
        "--normalizer", os.path.join(fold_log_dir, "normalizer.npz"),
        "--d_joint", str(args.d_joint),
        "--d_model", str(args.d_model),
        "--n_spatial_layers", str(args.n_spatial_layers),
        "--n_temporal_layers", str(args.n_temporal_layers),
        "--n_fusion_layers", str(args.n_fusion_layers),
        "--n_decoder_layers", str(args.n_decoder_layers),
        "--pose_smooth_window", str(args.pose_smooth_window),
        "--vel_smooth_window", str(args.vel_smooth_window),
        "--output_json", os.path.join(fold_log_dir, "test_results.json"),
    ]
    
    # Add optional flags
    if args.no_vr:
        test_cmd.append("--no_vr")
    if args.no_velocity:
        test_cmd.append("--no_velocity")
    if args.no_joint_groups:
        test_cmd.append("--no_joint_groups")
    
    # MAD filtering (must match training)
    if args.mad_threshold is not None:
        test_cmd.extend(["--mad_threshold", str(args.mad_threshold)])
    
    # Add dataset-specific args
    if args.dataset_type == "estimated15":
        if args.ae_checkpoint:
            test_cmd.extend(["--ae_checkpoint", args.ae_checkpoint])
        if args.heatmap_dir:
            test_cmd.extend(["--heatmap_dir", args.heatmap_dir])
        if args.align_to_gt:
            test_cmd.append("--align_to_gt")
    
    print(f"  Testing fold {fold_idx} ({test_subject})...")
    
    # Stream test output in real-time (show metrics)
    result = subprocess.run(test_cmd, text=True)
    if result.returncode != 0:
        print(f"  ERROR in testing fold {fold_idx} (exit code {result.returncode})")
        return {"error": "test_failed"}
    
    # Load test results
    results_file = os.path.join(fold_log_dir, "test_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    else:
        return {"error": "no_results_file"}


def aggregate_results(fold_results: List[Dict], subjects: List[str]) -> None:
    """Aggregate and print LOPO results."""
    print("\n" + "="*80)
    print("LEAVE-ONE-PERSON-OUT (LOPO) RESULTS")
    print("="*80)

    # test.py saves:
    #   "metrics_deviation" — raw model output (deviation from per-seq mean)
    #   "metrics_absolute"  — deviation + per-seq mean → reconstructed absolute CoP
    # Absolute is the primary metric (real-world interpretable cm error).
    # Deviation is secondary (tells you how well the model learns dynamic lean).

    has_absolute = all("metrics_absolute" in r for r in fold_results if "error" not in r)
    PRIMARY_KEY   = "metrics_absolute" if has_absolute else "metrics_deviation"
    PRIMARY_LABEL = "Absolute (deviation + per-seq mean)" if has_absolute else "Deviation"

    metrics_to_agg = [
        "rmse_x", "rmse_y", "rmse_total",
        "mae_x",  "mae_y",  "mae_total",
        "r2_x",   "r2_y",
        "corr_x", "corr_y", "corr_total",
        "within_10mm", "within_20mm", "within_30mm",
    ]

    # Per-fold results table (primary metric)
    print(f"\nPer-fold results [{PRIMARY_LABEL}]:")
    print(f"{'Fold':4s} {'Subject':12s} {'RMSE_X':8s} {'RMSE_Y':8s} {'RMSE_Tot':9s} "
          f"{'R2_X':8s} {'R2_Y':8s} {'Corr_Tot':9s}")
    print("-" * 80)

    valid_results = []
    for i, (result, subject) in enumerate(zip(fold_results, subjects)):
        if "error" in result:
            print(f"{i+1:4d} {subject:12s} ERROR: {result['error']}")
        else:
            metrics  = result.get(PRIMARY_KEY, {})
            rmse_x   = metrics.get("rmse_x",     float("nan"))
            rmse_y   = metrics.get("rmse_y",     float("nan"))
            rmse_tot = metrics.get("rmse_total", float("nan"))
            r2_x     = metrics.get("r2_x",       float("nan"))
            r2_y     = metrics.get("r2_y",       float("nan"))
            corr_tot = metrics.get("corr_total", float("nan"))

            print(f"{i+1:4d} {subject:12s} {rmse_x:8.3f} {rmse_y:8.3f} {rmse_tot:9.3f} "
                  f"{r2_x:8.3f} {r2_y:8.3f} {corr_tot:9.3f}")
            valid_results.append(result)

    if not valid_results:
        print("\nNo valid results to aggregate!")
        return

    # Aggregate primary metrics
    print(f"\nAggregated LOPO ({len(valid_results)}/{len(subjects)} folds) — {PRIMARY_LABEL}:")
    print("-" * 55)
    for metric in metrics_to_agg:
        values = [r.get(PRIMARY_KEY, {}).get(metric) for r in valid_results
                  if r.get(PRIMARY_KEY, {}).get(metric) is not None]
        if values:
            print(f"  {metric:15s}: {np.mean(values):7.3f} ± {np.std(values):6.3f}")

    # Secondary: deviation metrics (how well model learns dynamic lean)
    print(f"\nAggregated LOPO ({len(valid_results)}/{len(subjects)} folds) — Deviation (model target):")
    print("-" * 55)
    for metric in metrics_to_agg:
        values = [r.get("metrics_deviation", {}).get(metric) for r in valid_results
                  if r.get("metrics_deviation", {}).get(metric) is not None]
        if values:
            print(f"  {metric:15s}: {np.mean(values):7.3f} ± {np.std(values):6.3f}")


def main():
    parser = argparse.ArgumentParser(description="LOPO Cross-Validation for Balance Prediction")
    
    # Data
    parser.add_argument("--all_seq_file", default="pose_estimation/utils/all_sequence.txt",
                       help="File containing all sequence paths")
    parser.add_argument("--dataset_type", required=True,
                       choices=["gt19_enhanced", "estimated15"],
                       help="Dataset type")
    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--fps", type=float, default=22.0)
    
    # Model
    parser.add_argument("--model_type", required=True,
                       choices=["transformer", "gru", "gru_enhanced", "lstm", "lstm_enhanced", "cnn_lstm", "cnn_lstm_enhanced", "st_transformer", "st_transformer_enhanced"])
    parser.add_argument("--d_joint", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_spatial_layers", type=int, default=2)
    parser.add_argument("--n_temporal_layers", type=int, default=3)
    parser.add_argument("--n_fusion_layers", type=int, default=2)
    parser.add_argument("--n_decoder_layers", type=int, default=2)
    parser.add_argument("--pose_smooth_window", type=int, default=5)
    parser.add_argument("--vel_smooth_window", type=int, default=3)
    parser.add_argument("--no_velocity", action="store_true")
    parser.add_argument("--no_vr", action="store_true")
    parser.add_argument("--no_joint_groups", action="store_true",
                       help="Disable joint-group branch in st_transformer_enhanced")
    
    # Loss function
    parser.add_argument("--loss_type", type=str, default="huber",
                       choices=["mse", "huber", "correlation"],
                       help="Loss function: mse, huber, or correlation (correlation-aware)")
    parser.add_argument("--huber_delta", type=float, default=1.0,
                       help="Delta for Huber loss (default: 1.0)")
    parser.add_argument("--corr_alpha", type=float, default=0.5,
                       help="Weight for correlation term in correlation-aware loss (default: 0.5)")
    
    # Data filtering
    parser.add_argument("--mad_threshold", type=float, default=None,
                       help="MAD threshold for CoP outlier removal (e.g., 3.5). None=disabled")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_from_epoch", type=int, default=50,
                       help="Only save models from this epoch onwards (default: 50)")
    parser.add_argument("--test_checkpoint", type=str, default="final",
                       choices=["best", "final"],
                       help="Which checkpoint to use for testing: 'best' (best_model.pth) or 'final' (epoch_XXXX.pth, default)")
    
    # Estimated15 specific
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    parser.add_argument("--heatmap_dir", type=str, default="custom_pred_heatmaps")
    parser.add_argument("--align_to_gt", action="store_true")
    
    # Output
    parser.add_argument("--base_log_dir", default="log/lopo",
                       help="Base directory for LOPO results")
    parser.add_argument("--tmp_dir", default="tmp/lopo_folds",
                       help="Temporary directory for fold files")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.base_log_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    # Extract subjects and sequences
    print("Extracting subjects from", args.all_seq_file)
    subjects, subject_seqs = extract_subjects_and_sequences(args.all_seq_file)
    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
    
    # Run LOPO
    fold_results = []
    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{len(subjects)}: Testing on {test_subject}")
        print('='*60)
        
        # Create fold files
        train_file, test_file = create_fold_files(fold_idx, test_subject, subject_seqs, args.tmp_dir)
        
        # Run train + test
        result = run_train_test(fold_idx, test_subject, train_file, test_file, args)
        fold_results.append(result)
    
    # Aggregate results
    aggregate_results(fold_results, subjects)
    
    # Save aggregated results
    output_file = os.path.join(args.base_log_dir, "lopo_summary.json")
    with open(output_file, "w") as f:
        json.dump({
            "subjects": subjects,
            "fold_results": fold_results,
            "args": vars(args)
        }, f, indent=2)
    
    print(f"\nLOPO results saved to: {output_file}")


if __name__ == "__main__":
    main()
