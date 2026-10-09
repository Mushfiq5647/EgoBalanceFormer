"""
Train the Transformer CoP predictor.

Example:
    cd <repo root>

    python balance_prediction/train.py \
        --train_list utils/train.txt \
        --val_list   utils/test.txt  \
        --log_dir    log/cop_transformer \
        --epochs     100 \
        --batch_size 64
"""

import argparse
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.dataset      import BalanceDataset  # noqa
from data.dataset_gt19_root import BalanceDatasetGT19Root  # noqa
from data.dataset_gt19_root_aligned import BalanceDatasetGT19RootAligned  # noqa
from data.dataset_gt19_enhanced import BalanceDatasetGT19Enhanced  # noqa
from data.dataset_estimated15 import BalanceDatasetEstimated15  # noqa
from data.normalization import Normalizer                          # noqa
from models.transformer import TransformerCoPPredictor             # noqa
from models.gru_regressor import GRUCoPPredictor                   # noqa
from models.gru_enhanced_regressor import GRUEnhancedCoPPredictor  # noqa
from models.lstm_regressor import LSTMCoPPredictor                 # noqa
from models.lstm_enhanced_regressor import LSTMEnhancedCoPPredictor # noqa
from models.linear_regressor import LinearCoPPredictor              # noqa
from models.deeptcn_regressor import DeepTCNCoPPredictor            # noqa
from models.cnn_lstm_regressor import CNNLSTMCoPPredictor            # noqa
from models.cnn_lstm_enhanced_regressor import CNNLSTMEnhancedCoPPredictor  # noqa
from models.stgcn_regressor import STGCNCoPPredictor                 # noqa
from models.st_transformer import SpatioTemporalCoPPredictor       # noqa
from models.st_transformer_enhanced import SpatioTemporalCoPPredictorEnhanced  # noqa
from utils.metrics      import compute_cop_metrics, print_metrics  # noqa
from pose_estimation.utils.loss import CorrelationAwareLoss  # noqa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_list",   required=True,  help="List file of train sequence dirs")
    p.add_argument("--val_list",     default=None,   help="List file of val sequence dirs (optional)")
    p.add_argument("--window_size",  type=int,   default=11)
    p.add_argument("--last_k_frames", type=int, default=0,
                   help="Use only last K frames from each window for temporal-context ablation (0=use all)")
    p.add_argument("--fps",          type=float, default=22.0)
    p.add_argument("--smooth_window",type=int,   default=3)
    p.add_argument("--pose_source",  type=str,   default="json",
                   choices=["json", "ae_tensor"],
                   help="json: read estimated_poses.json; ae_tensor: infer pose from heatmaps + AE checkpoint")
    p.add_argument("--pose_filename",type=str,   default="estimated_poses.json",
                   help="Pose file name inside each sequence dir (falls back to ground_truth.json)")
    p.add_argument("--heatmap_subdir", type=str, default="custom_pred_heatmaps")
    p.add_argument("--ae_checkpoint", type=str, default=None,
                   help="Required when --pose_source ae_tensor")
    p.add_argument("--ae_device", type=str, default="cpu",
                   help="Device for offline pose inference inside dataset preload (cpu/cuda)")
    p.add_argument("--ae_hidden_size", type=int, default=20)
    p.add_argument("--ae_batch_size", type=int, default=128)
    p.add_argument("--use_body_part", action="store_true")
    p.add_argument("--vr_filename",  type=str,   default="vr_data.json",
                   help="VR tracking file name inside each sequence dir")
    p.add_argument("--dataset_type", type=str, default="default",
                   choices=["default", "gt19_root", "gt19_root_aligned", "gt19_enhanced", "estimated15"],
                   help="default: existing loader, gt19_root: raw 19 joints, "
                        "gt19_root_aligned: timestamp-aligned gt19, "
                        "gt19_enhanced: aligned + joint groups + CoM (pose-only or pose+VR), "
                        "estimated15: 15-joint estimated pose from AE")
    p.add_argument("--max_offset_ms", type=float, default=50.0,
                   help="Max pose↔CoP timestamp offset in ms (at 22fps nearest frame is <=23ms)")

    # Model
    p.add_argument("--d_model",         type=int,   default=128)
    p.add_argument("--nhead",           type=int,   default=8)
    p.add_argument("--num_layers",      type=int,   default=4)
    p.add_argument("--dim_feedforward", type=int,   default=512)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--model_type",      type=str, default="transformer",
                   choices=["transformer", "gru", "gru_enhanced", "lstm", "lstm_enhanced", "linear", "deeptcn", "cnn_lstm", "cnn_lstm_enhanced", "stgcn", "st_transformer", "st_transformer_enhanced"])
    p.add_argument("--gru_hidden_size", type=int, default=128)
    p.add_argument("--gru_layers",      type=int, default=2)
    p.add_argument("--tcn_levels",      type=int, default=4,
                   help="Number of temporal blocks in DeepTCN")
    p.add_argument("--tcn_kernel_size", type=int, default=3,
                   help="Kernel size for DeepTCN temporal convolutions")
    # st_transformer-specific
    p.add_argument("--d_joint",           type=int, default=64,
                   help="Spatial attention dim per joint (st_transformer* only)")
    p.add_argument("--n_spatial_layers",  type=int, default=2,
                   help="Spatial encoder layers (st_transformer* only)")
    p.add_argument("--n_temporal_layers", type=int, default=3,
                   help="Temporal encoder layers (st_transformer* only)")
    p.add_argument("--n_cross_layers",    type=int, default=2,
                   help="Cross-attention layers (st_transformer only)")
    p.add_argument("--n_fusion_layers",   type=int, default=2,
                   help="Fusion layers (st_transformer_enhanced only)")
    p.add_argument("--n_decoder_layers",  type=int, default=2,
                   help="Decoder layers (st_transformer* only)")
    p.add_argument("--pose_smooth_window", type=int, default=5,
                   help="Temporal smoothing window for pose/VR (gt19_enhanced only)")
    p.add_argument("--vel_smooth_window",  type=int, default=3,
                   help="Velocity smoothing window (gt19_enhanced only)")
    p.add_argument("--no_velocity", action="store_true",
                   help="Disable pose/VR velocities (pose_pos, vr_pos still used)")
    p.add_argument("--no_vr", action="store_true",
                   help="Disable VR inputs (pose-only: pose_pos/vel, joint_groups, com)")
    
    # Estimated15 dataset options (ae_checkpoint already defined above)
    p.add_argument("--heatmap_dir", type=str, default="custom_pred_heatmaps",
                   help="Directory containing heatmaps (estimated15 only)")
    p.add_argument("--align_to_gt", action="store_true",
                   help="Apply Procrustes alignment to GT before training (estimated15 only, default: enabled)")
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
                   help="Forecast horizon in CoP steps (2 Hz): target at t+H (estimated15 only)")
    p.add_argument("--butterworth", action="store_true",
                   help="Use Butterworth low-pass filtering for estimated15 inputs")
    p.add_argument("--bw_cutoff_hz", type=float, default=3.0,
                   help="Butterworth cutoff frequency in Hz (estimated15 only)")
    p.add_argument("--bw_order", type=int, default=2,
                   help="Butterworth order (estimated15 only)")
    p.add_argument("--rotation_source", type=str, default="gt",
                   choices=["gt", "pose"],
                   help="Rotation feature source for estimated15: gt or pose-derived")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--print_every",  type=int,   default=50)
    p.add_argument("--save_from_epoch", type=int, default=1,
                   help="Only save best model from this epoch onwards (default: 1)")
    p.add_argument("--no_normalize", action="store_true",
                   help="Disable input/target normalization during training and evaluation")
    p.add_argument("--cop_norm", type=str, default="zscore",
                   choices=["zscore", "minmax"],
                   help="Target CoP normalization mode")
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
    
    # Loss function
    p.add_argument("--loss_type", type=str, default="huber",
                   choices=["mse", "huber", "correlation"],
                   help="Loss function: mse, huber, or correlation (correlation-aware)")
    p.add_argument("--huber_delta", type=float, default=1.0,
                   help="Delta for Huber loss (default: 1.0)")
    p.add_argument("--corr_alpha", type=float, default=0.5,
                   help="Weight for correlation term in correlation-aware loss (default: 0.5)")
    
    # Data filtering
    p.add_argument("--mad_threshold", type=float, default=None,
                   help="MAD threshold for CoP outlier removal (e.g., 3.5). None=disabled")

    # Paths
    p.add_argument("--log_dir",      default="log/cop_transformer")
    p.add_argument("--pretrained",   default=None, help="Resume from checkpoint")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop if val_rmse does not improve for this many epochs (0=disabled)")

    return p.parse_args()


def _build_dataset_and_loader(
    list_file: str,
    args: argparse.Namespace,
    shuffle: bool,
) -> tuple:
    if args.dataset_type == "gt19_root_aligned":
        ds = BalanceDatasetGT19RootAligned(
            list_file=list_file,
            window_size=args.window_size,
            fps=args.fps,
            max_offset_ms=args.max_offset_ms,
        )
    elif args.dataset_type == "gt19_enhanced":
        ds = BalanceDatasetGT19Enhanced(
            list_file=list_file,
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
            list_file=list_file,
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
            list_file=list_file,
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
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
    return ds, dl


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _slice_last_k(x: torch.Tensor | None, k: int) -> torch.Tensor | None:
    if x is None or k <= 0:
        return x
    if x.dim() >= 2 and x.shape[1] >= k:
        return x[:, -k:, ...]
    return x


def _forward(model, batch, no_pose: bool = False, last_k_frames: int = 0):
    pose_pos = _slice_last_k(batch["pose_pos"], last_k_frames)
    pose_vel = _slice_last_k(batch.get("pose_vel"), last_k_frames)
    vr_pos = _slice_last_k(batch.get("vr_pos"), last_k_frames)
    vr_vel = _slice_last_k(batch.get("vr_vel"), last_k_frames)
    joint_groups = _slice_last_k(batch.get("joint_groups"), last_k_frames)
    joint_rotations = _slice_last_k(batch.get("joint_rotations"), last_k_frames)
    joint_rot_vel = _slice_last_k(batch.get("joint_rot_vel"), last_k_frames)
    com = _slice_last_k(batch.get("com"), last_k_frames)
    com_vel = _slice_last_k(batch.get("com_vel"), last_k_frames)

    if no_pose:
        pose_pos = torch.zeros_like(pose_pos)
        if pose_vel is not None:
            pose_vel = torch.zeros_like(pose_vel)

    if isinstance(model, (SpatioTemporalCoPPredictor, SpatioTemporalCoPPredictorEnhanced, GRUEnhancedCoPPredictor, LSTMEnhancedCoPPredictor, CNNLSTMEnhancedCoPPredictor)):
        return model(
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
    return model(
        pose_pos=pose_pos,
        pose_vel=pose_vel,
        vr_pos=vr_pos,
        vr_vel=vr_vel,
    )


def _plot_history(history: dict, out_path: str) -> None:
    if not _HAS_MPL:
        return
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
    if history.get("val_loss"):
        plt.plot(epochs[: len(history["val_loss"])], history["val_loss"], label="val_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def evaluate(model, loader, normalizer, device, criterion, no_pose: bool = False, last_k_frames: int = 0) -> tuple[float, dict]:
    model.eval()
    all_preds, all_targets = [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        batch = _batch_to_device(batch, device)
        if normalizer and normalizer.is_fitted():
            batch = normalizer.normalize_batch(batch)

        pred = _forward(model, batch, no_pose=no_pose, last_k_frames=last_k_frames)            # (B, 2)
        loss = criterion(pred, batch["cop"])
        total_loss += loss.item() * pred.shape[0]
        n           += pred.shape[0]

        # Denormalize before metric computation
        pred_np = pred.cpu().numpy()
        gt_np   = batch["cop"].cpu().numpy()
        if normalizer and normalizer.is_fitted():
            pred_np = normalizer.denormalize_cop(torch.from_numpy(pred_np)).numpy()
            gt_np   = normalizer.denormalize_cop(torch.from_numpy(gt_np)).numpy()

        all_preds.append(pred_np)
        all_targets.append(gt_np)

    model.train()
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_cop_metrics(preds, targets)
    metrics["loss"] = total_loss / max(n, 1)
    # Diagnostic: if pred_std << gt_std, model may be collapsing toward the mean
    metrics["pred_std"] = float(np.std(preds))
    metrics["gt_std"] = float(np.std(targets))
    return metrics["loss"], metrics


def main() -> None:
    args = parse_args()
    if args.last_k_frames < 0:
        raise ValueError("--last_k_frames must be >= 0")
    if args.last_k_frames > args.window_size:
        raise ValueError("--last_k_frames cannot exceed --window_size")
    if args.forecast_horizon < 0:
        raise ValueError("--forecast_horizon must be >= 0")
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, train_dl = _build_dataset_and_loader(args.train_list, args, shuffle=True)

    val_dl = None
    if args.val_list:
        try:
            _, val_dl = _build_dataset_and_loader(args.val_list, args, shuffle=False)
        except ValueError as e:
            print(f"[warn] Skipping validation: {e}")

    # ── Normalizer ────────────────────────────────────────────────────────────
    norm_path = os.path.join(args.log_dir, "normalizer.npz")
    normalizer = None
    if not args.no_normalize:
        normalizer = Normalizer(cop_mode=args.cop_norm)
        normalizer.fit(train_ds)
        normalizer.save(norm_path)
    else:
        print("[Normalizer] Disabled (--no_normalize)", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    sample = train_ds[0]
    eff_window = args.last_k_frames if args.last_k_frames > 0 else args.window_size
    n_pose_joints = int(sample["pose_pos"].shape[1])
    n_vel_joints  = int(sample["pose_vel"].shape[1]) if "pose_vel" in sample else 0
    n_vr_coords   = 0 if args.no_vr else int(sample["vr_pos"].shape[-1])
    n_groups      = int(sample["joint_groups"].shape[1]) if "joint_groups" in sample else 5
    use_vr_vel    = ("vr_vel" in sample) and (not args.no_vr)
    print(f"Dataset config: n_pose_joints={n_pose_joints}, n_vr_coords={n_vr_coords}, "
          f"use_pose_vel={'pose_vel' in sample}, use_vr_vel={use_vr_vel}")

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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if args.pretrained and os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Loaded pretrained weights from {args.pretrained}")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Loss function
    if args.loss_type == "correlation":
        criterion = CorrelationAwareLoss(
            base_loss="huber",
            alpha=args.corr_alpha,
            huber_delta=args.huber_delta
        )
        print(f"Using CorrelationAwareLoss (alpha={args.corr_alpha}, delta={args.huber_delta})")
    elif args.loss_type == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta)
        print(f"Using HuberLoss (delta={args.huber_delta})")
    elif args.loss_type == "mse":
        criterion = nn.MSELoss()
        print("Using MSELoss")
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    epochs_no_improve_rmse = 0
    history = {"train_loss": [], "val_loss": []}

    model.train()
    last_epoch_trained = 0
    for epoch in range(1, args.epochs + 1):
        last_epoch_trained = epoch
        running_loss = 0.0
        seen = 0
        steps_per_epoch = len(train_dl)
        print(f"epoch {epoch}/{args.epochs} started  steps={steps_per_epoch}", flush=True)

        for step, batch in enumerate(train_dl, start=1):
            batch = _batch_to_device(batch, device)
            if normalizer and normalizer.is_fitted():
                batch = normalizer.normalize_batch(batch)

            pred = _forward(model, batch, no_pose=args.no_pose, last_k_frames=args.last_k_frames)
            loss = criterion(pred, batch["cop"])

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * pred.shape[0]
            seen          += pred.shape[0]

            if args.print_every > 0 and (step % args.print_every == 0 or step == steps_per_epoch):
                print(f"  epoch {epoch} step {step}/{len(train_dl)}  "
                      f"loss={loss.item():.6f}", flush=True)

        train_loss = running_loss / max(seen, 1)
        history["train_loss"].append(train_loss)
        scheduler.step()

        msg = f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}"

        if val_dl is not None:
            val_loss, val_metrics = evaluate(
                model, val_dl, normalizer, device, criterion,
                no_pose=args.no_pose, last_k_frames=args.last_k_frames
            )
            history["val_loss"].append(val_loss)
            val_rmse = val_metrics["rmse_total"]
            msg += f"  val_loss={val_loss:.6f}  val_rmse={val_rmse:.2f} cm"
            msg += f"  (pred_std={val_metrics['pred_std']:.2f}  gt_std={val_metrics['gt_std']:.2f})"

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improve_rmse = 0
                # Always save best model whenever val_rmse improves.
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "metrics": val_metrics},
                    os.path.join(args.log_dir, "best_model.pth"),
                )
                msg += "  ← best val_rmse (saved)"
            else:
                epochs_no_improve_rmse += 1
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        print(msg, flush=True)

        if args.early_stop_patience > 0 and val_dl is not None and epochs_no_improve_rmse >= args.early_stop_patience:
            print(f"Early stopping: no val_rmse improvement for {args.early_stop_patience} epochs.", flush=True)
            break

        # Periodic checkpoint (save every 10 epochs if past save_from_epoch)
        if epoch >= args.save_from_epoch and epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                os.path.join(args.log_dir, f"epoch_{epoch:04d}.pth"),
            )

    # Save final epoch checkpoint using the actual last trained epoch.
    torch.save(
        {"epoch": last_epoch_trained, "model": model.state_dict()},
        os.path.join(args.log_dir, f"epoch_{last_epoch_trained:04d}.pth"),
    )
    print(f"Saved final checkpoint: epoch_{last_epoch_trained:04d}.pth", flush=True)

    # Save training history
    history_path = os.path.join(args.log_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    _plot_history(history, os.path.join(args.log_dir, "loss_curve.png"))
    print(f"Saved history to {history_path}", flush=True)
    print(f"Saved loss plot to {os.path.join(args.log_dir, 'loss_curve.png')}", flush=True)

    print("Training done.", flush=True)


if __name__ == "__main__":
    main()
