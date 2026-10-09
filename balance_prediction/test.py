"""
Evaluate a trained Transformer CoP predictor.

Example:
    cd <repo root>

    python balance_prediction/test.py \
        --test_list  utils/test.txt \
        --checkpoint log/cop_transformer/best_model.pth \
        --normalizer log/cop_transformer/normalizer.npz \
        --output_json log/cop_transformer/test_results.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.dataset       import BalanceDataset  # noqa
from data.dataset_gt19_root import BalanceDatasetGT19Root  # noqa
from data.dataset_gt19_root_aligned import BalanceDatasetGT19RootAligned  # noqa
from data.dataset_gt19_enhanced import BalanceDatasetGT19Enhanced  # noqa
from data.dataset_estimated15 import BalanceDatasetEstimated15  # noqa
from data.normalization  import Normalizer                          # noqa
from models.transformer  import TransformerCoPPredictor             # noqa
from models.gru_regressor import GRUCoPPredictor                    # noqa
from models.gru_enhanced_regressor import GRUEnhancedCoPPredictor  # noqa
from models.lstm_regressor import LSTMCoPPredictor                  # noqa
from models.lstm_enhanced_regressor import LSTMEnhancedCoPPredictor # noqa
from models.linear_regressor import LinearCoPPredictor              # noqa
from models.deeptcn_regressor import DeepTCNCoPPredictor            # noqa
from models.cnn_lstm_regressor import CNNLSTMCoPPredictor           # noqa
from models.cnn_lstm_enhanced_regressor import CNNLSTMEnhancedCoPPredictor  # noqa
from models.stgcn_regressor import STGCNCoPPredictor                 # noqa
from models.st_transformer import SpatioTemporalCoPPredictor        # noqa
from models.st_transformer_enhanced import SpatioTemporalCoPPredictorEnhanced  # noqa
from utils.metrics       import compute_cop_metrics, print_metrics  # noqa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--test_list",    required=True)
    p.add_argument("--window_size",  type=int,   default=11)
    p.add_argument("--last_k_frames", type=int, default=0,
                   help="Use only last K frames from each window for temporal-context ablation (0=use all)")
    p.add_argument("--fps",          type=float, default=22.0)
    p.add_argument("--smooth_window",type=int,   default=3)
    p.add_argument("--pose_source",  type=str,   default="json",
                   choices=["json", "ae_tensor"])
    p.add_argument("--pose_filename",type=str,   default="estimated_poses.json")
    p.add_argument("--heatmap_subdir", type=str, default="custom_pred_heatmaps")
    p.add_argument("--ae_checkpoint", type=str, default=None)
    p.add_argument("--ae_device", type=str, default="cpu")
    p.add_argument("--ae_hidden_size", type=int, default=20)
    p.add_argument("--ae_batch_size", type=int, default=128)
    p.add_argument("--use_body_part", action="store_true")
    p.add_argument("--vr_filename",  type=str,   default="vr_data.json")
    p.add_argument("--dataset_type", type=str, default="default",
                   choices=["default", "gt19_root", "gt19_root_aligned", "gt19_enhanced", "estimated15"],
                   help="default: existing loader, gt19_root: raw 19 joints, "
                        "gt19_root_aligned: timestamp-aligned gt19, "
                        "gt19_enhanced: aligned + joint groups + CoM (pose-only or pose+VR), "
                        "estimated15: 15-joint estimated pose from AE")
    p.add_argument("--max_offset_ms", type=float, default=50.0,
                   help="Max pose↔CoP timestamp offset in ms (only for gt19_root_aligned)")
    p.add_argument("--no_velocity", action="store_true",
                   help="Disable pose/VR velocities (pose_pos, vr_pos still used) for gt19_enhanced")
    p.add_argument("--no_vr", action="store_true",
                   help="Disable VR inputs (pose-only) for gt19_enhanced")
    
    # Estimated15 dataset options (ae_checkpoint already defined above)
    p.add_argument("--heatmap_dir", type=str, default="custom_pred_heatmaps",
                   help="Directory containing heatmaps (estimated15 only)")
    p.add_argument("--align_to_gt", action="store_true",
                   help="Apply Procrustes alignment to GT before testing (estimated15 only, default: enabled)")
    p.set_defaults(align_to_gt=True)
    p.add_argument("--include_hmd", action="store_true",
                   help="Include HMD position in VR features for estimated15 (vr_pos: 9D)")
    p.add_argument("--vr_mode", type=str, default=None,
                   choices=["rel_lr", "hmd_only", "hmd_lr"],
                   help="VR feature mode for estimated15: rel_lr (default), hmd_only, or hmd_lr")
    p.add_argument("--cop_target_mode", type=str, default="k4",
                   choices=["k4", "seqmean", "absolute"],
                   help="CoP target for estimated15: k4/seqmean deviation, or absolute")
    p.add_argument("--cop_k", type=int, default=4,
                   help="Baseline K for cop_target_mode=k4 (estimated15 only)")
    p.add_argument("--forecast_horizon", type=int, default=0,
                   help="Forecast horizon in CoP steps (2 Hz): evaluate target at t+H")
    p.add_argument("--butterworth", action="store_true",
                   help="Use Butterworth low-pass filtering for estimated15 inputs")
    p.add_argument("--bw_cutoff_hz", type=float, default=3.0,
                   help="Butterworth cutoff frequency in Hz (estimated15 only)")
    p.add_argument("--bw_order", type=int, default=2,
                   help="Butterworth order (estimated15 only)")
    p.add_argument("--rotation_source", type=str, default="gt",
                   choices=["gt", "pose"],
                   help="Rotation feature source for estimated15: gt or pose-derived")
    
    # Data filtering
    p.add_argument("--mad_threshold", type=float, default=None,
                   help="MAD threshold for CoP outlier removal (e.g., 3.5). None=disabled")
    
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_workers",  type=int,   default=4)

    # Model
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--normalizer",   default=None,
                   help="Path to normalizer.npz saved during training")
    p.add_argument("--d_model",         type=int,   default=128)
    p.add_argument("--nhead",           type=int,   default=8)
    p.add_argument("--num_layers",      type=int,   default=4)
    p.add_argument("--dim_feedforward", type=int,   default=512)
    p.add_argument("--dropout",         type=float, default=0.0)   # off at test time
    p.add_argument("--model_type",      type=str, default="transformer",
                   choices=["transformer", "gru", "gru_enhanced", "lstm", "lstm_enhanced", "linear", "deeptcn", "cnn_lstm", "cnn_lstm_enhanced", "stgcn", "st_transformer", "st_transformer_enhanced"])
    p.add_argument("--gru_hidden_size", type=int, default=128)
    p.add_argument("--gru_layers",      type=int, default=2)
    p.add_argument("--tcn_levels",      type=int, default=4)
    p.add_argument("--tcn_kernel_size", type=int, default=3)
    p.add_argument("--d_joint",           type=int, default=64)
    p.add_argument("--n_spatial_layers",  type=int, default=2)
    p.add_argument("--n_temporal_layers", type=int, default=3)
    p.add_argument("--n_cross_layers",    type=int, default=2)
    p.add_argument("--n_fusion_layers",   type=int, default=2)
    p.add_argument("--n_decoder_layers",  type=int, default=2)
    p.add_argument("--pose_smooth_window", type=int, default=5)
    p.add_argument("--vel_smooth_window",  type=int, default=3)

    # Output
    p.add_argument("--output_json",  default=None,
                   help="Optional path to save per-window predictions as JSON")
    p.add_argument("--output_activation", type=str, default="linear",
                   choices=["linear", "tanh"],
                   help="Final output activation for compatible models")
    p.add_argument("--no_joint_groups", action="store_true",
                   help="Disable joint-group branch in st_transformer_enhanced")
    p.add_argument("--no_joint_rotations", action="store_true",
                   help="Disable joint-rotation branch in st_transformer_enhanced")
    p.add_argument("--no_com", action="store_true",
                   help="Disable CoM branch in st_transformer_enhanced")
    p.add_argument("--no_pose", action="store_true",
                   help="Zero-out pose_pos/pose_vel inputs (branch ablation)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.last_k_frames < 0:
        raise ValueError("--last_k_frames must be >= 0")
    if args.last_k_frames > args.window_size:
        raise ValueError("--last_k_frames cannot exceed --window_size")
    if args.forecast_horizon < 0:
        raise ValueError("--forecast_horizon must be >= 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dataset_type == "gt19_root_aligned":
        ds = BalanceDatasetGT19RootAligned(
            list_file=args.test_list,
            window_size=args.window_size,
            fps=args.fps,
            max_offset_ms=args.max_offset_ms,
        )
    elif args.dataset_type == "gt19_enhanced":
        ds = BalanceDatasetGT19Enhanced(
            list_file=args.test_list,
            window_size=args.window_size,
            fps=args.fps,
            pose_smooth_window=args.pose_smooth_window,
            vel_smooth_window=args.vel_smooth_window,
            max_offset_ms=args.max_offset_ms,
            use_velocity=not args.no_velocity,
            use_vr=not args.no_vr,
            mad_threshold=args.mad_threshold,
        )
    elif args.dataset_type == "estimated15":
        if args.ae_checkpoint is None:
            raise ValueError("--ae_checkpoint is required for dataset_type=estimated15")
        ds = BalanceDatasetEstimated15(
            list_file=args.test_list,
            window_size=args.window_size,
            fps=args.fps,
            pose_smooth_window=args.pose_smooth_window,
            vel_smooth_window=args.vel_smooth_window,
            max_offset_ms=args.max_offset_ms,
            use_velocity=not args.no_velocity,
            use_vr=not args.no_vr,
            ae_checkpoint=args.ae_checkpoint,
            heatmap_dir=args.heatmap_dir,
            use_body_part=args.use_body_part,
            ae_hidden_size=args.ae_hidden_size,
            ae_batch_size=args.ae_batch_size,
            align_to_gt=args.align_to_gt,
            mad_threshold=args.mad_threshold,
            include_hmd=args.include_hmd,
            vr_mode=args.vr_mode,
            cop_target_mode=args.cop_target_mode,
            cop_k=args.cop_k,
            forecast_horizon=args.forecast_horizon,
            butterworth=args.butterworth,
            bw_cutoff_hz=args.bw_cutoff_hz,
            bw_order=args.bw_order,
            rotation_source=args.rotation_source,
        )
    else:
        ds_cls = BalanceDataset if args.dataset_type == "default" else BalanceDatasetGT19Root
        if args.dataset_type == "gt19_root" and args.pose_source != "json":
            raise ValueError("dataset_type='gt19_root' requires --pose_source json")
        ds = ds_cls(
            list_file=args.test_list,
            window_size=args.window_size,
            fps=args.fps,
            smooth_window=args.smooth_window,
            pose_filename=args.pose_filename,
            pose_source=args.pose_source,
            heatmap_subdir=args.heatmap_subdir,
            ae_checkpoint=args.ae_checkpoint,
            ae_device=args.ae_device,
            ae_hidden_size=args.ae_hidden_size,
            ae_batch_size=args.ae_batch_size,
            use_body_part=args.use_body_part,
            vr_filename=args.vr_filename,
        )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Normalizer ────────────────────────────────────────────────────────────
    normalizer = None
    if args.normalizer and os.path.exists(args.normalizer):
        normalizer = Normalizer.load(args.normalizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    sample = ds[0]
    eff_window = args.last_k_frames if args.last_k_frames > 0 else args.window_size
    n_pose_joints = int(sample["pose_pos"].shape[1])
    n_vel_joints  = int(sample["pose_vel"].shape[1]) if "pose_vel" in sample else 0
    n_vr_coords   = 0 if args.no_vr else int(sample["vr_pos"].shape[-1])
    n_groups      = int(sample["joint_groups"].shape[1]) if "joint_groups" in sample else 5
    use_vr_vel    = ("vr_vel" in sample) and (not args.no_vr)
    if args.model_type == "st_transformer":
        model = SpatioTemporalCoPPredictor(
            n_joints=n_pose_joints,
            n_vr_coords=n_vr_coords,
            d_joint=args.d_joint,
            d_model=args.d_model,
            nhead=args.nhead,
            n_spatial_layers=args.n_spatial_layers,
            n_temporal_layers=args.n_temporal_layers,
            n_cross_layers=args.n_cross_layers,
            n_decoder_layers=args.n_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            window_size=eff_window,
            output_activation=args.output_activation,
            use_joint_groups=not args.no_joint_groups,
            use_joint_rotations=not args.no_joint_rotations,
            use_pose_vel=not args.no_velocity,
            use_com=not args.no_com,
        ).to(device)
    elif args.model_type == "st_transformer_enhanced":
        model = SpatioTemporalCoPPredictorEnhanced(
            n_joints=n_pose_joints,
            n_groups=n_groups,
            n_vr_coords=n_vr_coords,
            d_joint=args.d_joint,
            d_model=args.d_model,
            nhead=args.nhead,
            n_spatial_layers=args.n_spatial_layers,
            n_temporal_layers=args.n_temporal_layers,
            n_fusion_layers=args.n_fusion_layers,
            n_decoder_layers=args.n_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            window_size=eff_window,
            output_activation=args.output_activation,
            use_joint_groups=not args.no_joint_groups,
            use_joint_rotations=not args.no_joint_rotations,
            use_rotation_vel=not args.no_velocity,
            use_pose_vel=not args.no_velocity,
            use_com=not args.no_com,
        ).to(device)
    elif args.model_type == "lstm_enhanced":
        model = LSTMEnhancedCoPPredictor(
            n_joints=n_pose_joints,
            n_groups=n_groups,
            n_vr_coords=n_vr_coords,
            hidden_size=args.gru_hidden_size,
            num_layers=args.gru_layers,
            dropout=args.dropout,
            output_activation=args.output_activation,
            use_joint_groups=not args.no_joint_groups,
            use_joint_rotations=not args.no_joint_rotations,
            use_rotation_vel=not args.no_velocity,
            use_pose_vel=not args.no_velocity,
            use_com=not args.no_com,
        ).to(device)
    elif args.model_type == "gru_enhanced":
        model = GRUEnhancedCoPPredictor(
            n_joints=n_pose_joints,
            n_groups=n_groups,
            n_vr_coords=n_vr_coords,
            hidden_size=args.gru_hidden_size,
            num_layers=args.gru_layers,
            dropout=args.dropout,
            output_activation=args.output_activation,
            use_joint_groups=not args.no_joint_groups,
            use_joint_rotations=not args.no_joint_rotations,
            use_rotation_vel=not args.no_velocity,
            use_pose_vel=not args.no_velocity,
            use_com=not args.no_com,
        ).to(device)
    elif args.model_type == "cnn_lstm_enhanced":
        model = CNNLSTMEnhancedCoPPredictor(
            n_joints=n_pose_joints,
            n_groups=n_groups,
            n_vr_coords=n_vr_coords,
            conv_channels=max(args.gru_hidden_size // 2, 32),
            hidden_size=args.gru_hidden_size,
            num_layers=args.gru_layers,
            dropout=args.dropout,
            output_activation=args.output_activation,
            use_joint_groups=not args.no_joint_groups,
            use_joint_rotations=not args.no_joint_rotations,
            use_rotation_vel=not args.no_velocity,
            use_pose_vel=not args.no_velocity,
            use_com=not args.no_com,
        ).to(device)
    elif args.model_type == "gru":
        model = GRUCoPPredictor(
            n_pose_joints=n_pose_joints,
            n_vel_joints=n_vel_joints,
            n_vr_coords=n_vr_coords,
            use_vr_vel=use_vr_vel,
            hidden_size=args.gru_hidden_size,
            num_layers=args.gru_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "lstm":
        model = LSTMCoPPredictor(
            n_pose_joints=n_pose_joints,
            n_vel_joints=n_vel_joints,
            n_vr_coords=n_vr_coords,
            use_vr_vel=use_vr_vel,
            hidden_size=args.gru_hidden_size,
            num_layers=args.gru_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "linear":
        model = LinearCoPPredictor(
            n_pose_joints=n_pose_joints,
            n_vel_joints=n_vel_joints,
            n_vr_coords=n_vr_coords,
            use_vr_vel=use_vr_vel,
            window_size=eff_window,
        ).to(device)
    elif args.model_type == "deeptcn":
        model = DeepTCNCoPPredictor(
            n_pose_joints=n_pose_joints,
            n_vel_joints=n_vel_joints,
            n_vr_coords=n_vr_coords,
            use_vr_vel=use_vr_vel,
            hidden_channels=args.gru_hidden_size,
            num_levels=args.tcn_levels,
            kernel_size=args.tcn_kernel_size,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "cnn_lstm":
        model = CNNLSTMCoPPredictor(
            n_pose_joints=n_pose_joints,
            n_vel_joints=n_vel_joints,
            n_vr_coords=n_vr_coords,
            use_vr_vel=use_vr_vel,
            conv_channels=max(args.gru_hidden_size // 2, 32),
            hidden_size=args.gru_hidden_size,
            num_layers=args.gru_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "stgcn":
        model = STGCNCoPPredictor(
            n_pose_joints=n_pose_joints,
            n_vel_joints=n_vel_joints,
            n_vr_coords=n_vr_coords,
            use_vr_vel=use_vr_vel,
            hidden_size=args.gru_hidden_size,
            num_layers=args.gru_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        model = TransformerCoPPredictor(
            n_pose_joints=n_pose_joints,
            n_vel_joints=n_vel_joints,
            n_vr_coords=n_vr_coords,
            use_vr_vel=use_vr_vel,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            window_size=eff_window,
        ).to(device)
    print(f"Model: {args.model_type}, n_pose_joints={n_pose_joints}, n_vr_coords={n_vr_coords}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)   # handle both raw state_dict and wrapped ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_targets, all_cop_means, all_seq_dirs = [], [], [], []

    with torch.no_grad():
        for batch in dl:
            # Normalize full batch (cop_mean is skipped by the normalizer)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            cop_mean_np = batch["cop_mean"].cpu().numpy() if "cop_mean" in batch else None
            if normalizer and normalizer.is_fitted():
                batch = normalizer.normalize_batch(batch)

            def _slice_last_k(x: torch.Tensor | None) -> torch.Tensor | None:
                if x is None or args.last_k_frames <= 0:
                    return x
                if x.dim() >= 2 and x.shape[1] >= args.last_k_frames:
                    return x[:, -args.last_k_frames:, ...]
                return x

            pose_pos = _slice_last_k(batch["pose_pos"])
            pose_vel = _slice_last_k(batch.get("pose_vel"))
            vr_pos = _slice_last_k(batch.get("vr_pos"))
            vr_vel = _slice_last_k(batch.get("vr_vel"))
            joint_groups = _slice_last_k(batch.get("joint_groups"))
            joint_rotations = _slice_last_k(batch.get("joint_rotations"))
            joint_rot_vel = _slice_last_k(batch.get("joint_rot_vel"))
            com = _slice_last_k(batch.get("com"))
            com_vel = _slice_last_k(batch.get("com_vel"))
            if args.no_pose:
                pose_pos = torch.zeros_like(pose_pos)
                if pose_vel is not None:
                    pose_vel = torch.zeros_like(pose_vel)

            if isinstance(model, (SpatioTemporalCoPPredictor, SpatioTemporalCoPPredictorEnhanced, GRUEnhancedCoPPredictor, LSTMEnhancedCoPPredictor, CNNLSTMEnhancedCoPPredictor)):
                pred = model(
                    pose_pos=pose_pos,
                    pose_vel=pose_vel,
                    vr_pos=vr_pos,
                    vr_vel=vr_vel,
                    joint_groups=None if getattr(model, "use_joint_groups", True) is False else joint_groups,
                    joint_rotations=None if getattr(model, "use_joint_rotations", True) is False else joint_rotations,
                    joint_rot_vel=None if getattr(model, "use_joint_rotations", True) is False else joint_rot_vel,
                    com=None if getattr(model, "use_com", True) is False else com,
                    com_vel=None if getattr(model, "use_com", True) is False else com_vel,
                )
            else:
                pred = model(
                    pose_pos=pose_pos,
                    pose_vel=pose_vel,
                    vr_pos=vr_pos,
                    vr_vel=vr_vel,
                )

            # Denormalize back to cm (deviation space)
            pred_np = pred.cpu().numpy()
            gt_np   = batch["cop"].cpu().numpy()
            if normalizer and normalizer.is_fitted():
                pred_np = normalizer.denormalize_cop(torch.from_numpy(pred_np)).numpy()
                gt_np   = normalizer.denormalize_cop(torch.from_numpy(gt_np)).numpy()

            all_preds.append(pred_np)
            all_targets.append(gt_np)
            if cop_mean_np is not None:
                all_cop_means.append(cop_mean_np)
            all_seq_dirs.extend(batch["sequence_dir"])

    all_preds   = np.concatenate(all_preds)    # (N,2) CoP deviations from neutral
    all_targets = np.concatenate(all_targets)  # (N,2) CoP deviations from neutral
    has_cop_mean = len(all_cop_means) > 0
    if has_cop_mean:
        all_cop_means = np.concatenate(all_cop_means)  # (N,2) per-sample neutral CoP

    target_mode = getattr(ds, "cop_target_mode", "k4")
    is_absolute_target = target_mode == "absolute"

    # ── Primary metrics (what the model was trained to predict) ───────────────
    metrics_dev = compute_cop_metrics(all_preds, all_targets)
    label = "Absolute metrics — model prediction target" if is_absolute_target else "Deviation metrics — model prediction target"
    print(f"\n[{label}] ({len(all_preds)} samples)")
    print_metrics(metrics_dev)
    metrics_global = metrics_dev  # alias for downstream code

    # ── Absolute metrics (add back per-sequence CoP mean) ─────────────────────
    if has_cop_mean and (not is_absolute_target):
        abs_preds   = all_preds   + all_cop_means
        abs_targets = all_targets + all_cop_means
        metrics_abs = compute_cop_metrics(abs_preds, abs_targets)
        print(f"\n[Absolute metrics — deviation + per-seq mean] ({len(abs_preds)} samples)")
        print_metrics(metrics_abs)
    else:
        abs_preds   = all_preds
        abs_targets = all_targets

    # ── Collapse check: pred_std vs gt_std ────────────────────────────────────
    pred_std = float(np.std(all_preds))
    gt_std = float(np.std(all_targets))
    pred_std_x, pred_std_y = float(np.std(all_preds[:, 0])), float(np.std(all_preds[:, 1]))
    gt_std_x, gt_std_y = float(np.std(all_targets[:, 0])), float(np.std(all_targets[:, 1]))
    print(f"\n  Std (predictions): total={pred_std:.3f} cm  X={pred_std_x:.3f}  Y={pred_std_y:.3f}")
    print(f"  Std (ground truth): total={gt_std:.3f} cm  X={gt_std_x:.3f}  Y={gt_std_y:.3f}")
    if gt_std > 1e-6:
        ratio = pred_std / gt_std
        if ratio < 0.5:
            print(f"  → Pred std is {ratio:.2f}x gt_std: model may be collapsing toward the mean.")
        else:
            print(f"  → Pred std / gt_std = {ratio:.2f} (predictions have similar spread to gt).")

    # ── Subject-wise metrics (absolute CoP, more interpretable) ─────────────
    subject_to_indices: dict[str, list[int]] = {}
    for i, seq in enumerate(all_seq_dirs):
        subj = os.path.basename(seq.rstrip("/"))
        subject_to_indices.setdefault(subj, []).append(i)

    per_subject_metrics: dict[str, dict] = {}
    print("\nPer-subject deviation metrics:")
    for subj, idxs in sorted(subject_to_indices.items()):
        idxs_arr = np.array(idxs, dtype=int)
        # Deviation metrics (primary: model was trained on this)
        m_dev = compute_cop_metrics(all_preds[idxs_arr], all_targets[idxs_arr])
        per_subject_metrics[subj] = m_dev
        print(f"  [{subj}] N={len(idxs_arr)}  "
              f"RMSE={m_dev['rmse_total']:.3f} cm  "
              f"R2(X/Y)={m_dev['r2_x']:.3f}/{m_dev['r2_y']:.3f}  "
              f"Corr_total={m_dev['corr_total']:.3f}")

    # ── Optional JSON output ──────────────────────────────────────────────────
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        per_sample = []
        for i in range(len(all_preds)):
            if is_absolute_target:
                entry = {
                    "sequence_dir": all_seq_dirs[i],
                    "pred_CoPX": float(all_preds[i, 0]),
                    "pred_CoPY": float(all_preds[i, 1]),
                    "gt_CoPX":   float(all_targets[i, 0]),
                    "gt_CoPY":   float(all_targets[i, 1]),
                }
            else:
                entry = {
                    "sequence_dir": all_seq_dirs[i],
                    "pred_dev_CoPX": float(all_preds[i, 0]),
                    "pred_dev_CoPY": float(all_preds[i, 1]),
                    "gt_dev_CoPX":   float(all_targets[i, 0]),
                    "gt_dev_CoPY":   float(all_targets[i, 1]),
                }
            if has_cop_mean and (not is_absolute_target):
                entry["pred_abs_CoPX"] = float(abs_preds[i, 0])
                entry["pred_abs_CoPY"] = float(abs_preds[i, 1])
                entry["gt_abs_CoPX"]   = float(abs_targets[i, 0])
                entry["gt_abs_CoPY"]   = float(abs_targets[i, 1])
            per_sample.append(entry)

        out = {
            "metrics_deviation": metrics_dev,
            "metrics_per_subject": per_subject_metrics,
            "per_sample": per_sample,
        }
        if has_cop_mean and (not is_absolute_target):
            out["metrics_absolute"] = metrics_abs

        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
