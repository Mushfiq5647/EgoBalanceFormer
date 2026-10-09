"""
Test EgoGlass AutoEncoder (38ch: heatmaps + body part) on egocentric dataset.
MPJPE, PA-MPJPE, and joint-wise error in UnrealEgo and your convention.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloader.egocentric_pose_dataset import EgocentricPoseFromListFile  # noqa: E402
from model.network import AutoEncoder  # noqa: E402
from utils.loss import LossFuncMPJPE  # noqa: E402
from utils.util import batch_compute_similarity_transform_torch  # noqa: E402

UNREALEGO_JOINT_NAMES = [
    "head", "neck_01",
    "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "hand_l", "hand_r",
    "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r", "ball_l", "ball_r",
]

PREFERRED_JOINT_NAMES_16 = [
    "Head", "Neck", "L_Humerus", "R_Humerus", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Femur", "R_Femur", "L_Tibia", "R_Tibia", "L_Foot", "R_Foot", "L_Toe", "R_Toe",
]


def _make_opt(num_heatmap: int = 15, ae_hidden_size: int = 20):
    return SimpleNamespace(num_heatmap=num_heatmap, ae_hidden_size=ae_hidden_size)


def main():
    parser = argparse.ArgumentParser(description="Test EgoGlass AutoEncoder (38ch with body part)")
    parser.add_argument("--test_list", type=str, default="pose_estimation/utils/test.txt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ae_hidden_size", type=int, default=20)
    parser.add_argument("--limit_per_sequence", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--heatmap_subdir",
        type=str,
        default="egoglass_pred_heatmaps",
        help="EgoGlass heatmaps subdir (default: egoglass_pred_heatmaps).",
    )
    parser.add_argument(
        "--outlier_std",
        type=float,
        default=2.0,
        help="Frames with MPJPE > mean + k*std are outliers (default: 2.0).",
    )
    parser.add_argument(
        "--max_outlier_list",
        type=int,
        default=20,
        help="Max number of outlier frames to print (default: 20, 0=all).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = EgocentricPoseFromListFile(
        list_file=args.test_list,
        heatmap_subdir=args.heatmap_subdir,
        limit_per_sequence=args.limit_per_sequence,
        use_body_part=True,  # EgoGlass always uses body part
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    opt = _make_opt(num_heatmap=15, ae_hidden_size=args.ae_hidden_size)
    net = AutoEncoder(opt, input_channel_scale=2, extra_input_channels=8).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: '{args.checkpoint}'")
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.eval()

    loss_mpjpe = LossFuncMPJPE()
    total = 0.0
    total_pa = 0.0
    n = 0
    joint_mpjpe_sum = np.zeros(16, dtype=np.float64)
    per_frame = []

    def _to_ae_input(batch):
        hm15 = batch["heatmap15"].to(device)
        hm30 = torch.cat([hm15, hm15], dim=1)
        bp4 = batch["body_part4"].to(device)
        bp8 = torch.cat([bp4, bp4], dim=1)
        return torch.cat([hm30, bp8], dim=1)

    with torch.no_grad():
        for batch in dl:
            ae_input = _to_ae_input(batch)
            gt = batch["gt_pose16"].to(device)
            pred = net.predict_pose(ae_input)

            S1_hat = batch_compute_similarity_transform_torch(pred, gt)

            mpjpe_val = loss_mpjpe(pred, gt).item()
            pa_mpjpe_val = loss_mpjpe(S1_hat, gt).item()
            total += mpjpe_val * ae_input.shape[0]
            total_pa += pa_mpjpe_val * ae_input.shape[0]
            n += ae_input.shape[0]

            d = torch.linalg.norm(pred - gt, dim=-1)
            d_pa = torch.linalg.norm(S1_hat - gt, dim=-1)
            joint_mpjpe_sum += d.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            d_mean = d.mean(dim=1).detach().cpu().numpy().astype(np.float32)
            d_pa_mean = d_pa.mean(dim=1).detach().cpu().numpy().astype(np.float32)

            for i in range(len(batch["sequence_dir"])):
                per_frame.append({
                    "sequence_dir": str(batch["sequence_dir"][i]),
                    "image_stem": str(batch["image_stem"][i]),
                    "mpjpe": float(d_mean[i]),
                    "pa_mpjpe": float(d_pa_mean[i]),
                })

    mean_mpjpe = total / max(n, 1)
    mean_pa_mpjpe = total_pa / max(n, 1)
    print(f"test_samples={n} mean_mpjpe={mean_mpjpe:.6f} mean_pa_mpjpe={mean_pa_mpjpe:.6f}")

    joint_mpjpe = joint_mpjpe_sum / max(n, 1)
    print("\nJoint-wise MPJPE (UnrealEgo):")
    for j in range(16):
        print(f"  {UNREALEGO_JOINT_NAMES[j]:12s}: {joint_mpjpe[j]:.4f}")
    print("\nJoint-wise MPJPE (your convention):")
    for j in range(16):
        print(f"  {PREFERRED_JOINT_NAMES_16[j]:12s}: {joint_mpjpe[j]:.4f}")

    # Outlier analysis (per-frame MPJPE distribution)
    mpjpe_arr = np.array([f["mpjpe"] for f in per_frame], dtype=np.float64)
    std_mpjpe = float(np.std(mpjpe_arr)) if len(mpjpe_arr) > 1 else 0.0
    p25, p50, p75, p95, p99 = float(np.percentile(mpjpe_arr, 25)), float(np.percentile(mpjpe_arr, 50)), float(np.percentile(mpjpe_arr, 75)), float(np.percentile(mpjpe_arr, 95)), float(np.percentile(mpjpe_arr, 99))
    threshold = mean_mpjpe + args.outlier_std * std_mpjpe
    outlier_mask = mpjpe_arr > threshold
    num_outliers = int(np.sum(outlier_mask))
    outlier_frames = [(per_frame[i]["sequence_dir"], per_frame[i]["image_stem"], float(per_frame[i]["mpjpe"])) for i in np.where(outlier_mask)[0]]
    outlier_frames.sort(key=lambda x: -x[2])

    print("\nPer-frame MPJPE distribution:")
    print(f"  std: {std_mpjpe:.4f}  |  percentiles: p25={p25:.2f}  p50={p50:.2f}  p75={p75:.2f}  p95={p95:.2f}  p99={p99:.2f}")
    print(f"  Outliers (MPJPE > mean + {args.outlier_std}*std = {threshold:.2f}): {num_outliers} / {n} ({100.0 * num_outliers / max(n, 1):.2f}%)")
    if num_outliers > 0:
        to_show = outlier_frames[: args.max_outlier_list] if args.max_outlier_list > 0 else outlier_frames
        print(f"  Top outlier frames (sequence_dir, image_stem, mpjpe):")
        for seq, stem, mpj in to_show:
            print(f"    {seq} / {stem}: {mpj:.2f}")

    if args.output_json:
        import json
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        joint_mpjpe_unrealego = {name: float(joint_mpjpe[j]) for j, name in enumerate(UNREALEGO_JOINT_NAMES)}
        joint_mpjpe_preferred = {name: float(joint_mpjpe[j]) for j, name in enumerate(PREFERRED_JOINT_NAMES_16)}
        outlier_list = [{"sequence_dir": seq, "image_stem": stem, "mpjpe": mpj} for seq, stem, mpj in outlier_frames]
        with open(args.output_json, "w") as f:
            json.dump({
                "test_list": args.test_list,
                "checkpoint": args.checkpoint,
                "num_samples": n,
                "mean_mpjpe": mean_mpjpe,
                "mean_pa_mpjpe": mean_pa_mpjpe,
                "std_mpjpe": std_mpjpe,
                "percentiles": {"p25": p25, "p50": p50, "p75": p75, "p95": p95, "p99": p99},
                "outlier_threshold": threshold,
                "num_outliers": num_outliers,
                "outlier_frames": outlier_list,
                "joint_wise_mpjpe_unrealego": joint_mpjpe_unrealego,
                "joint_wise_mpjpe_your_convention": joint_mpjpe_preferred,
                "per_frame": per_frame,
            }, f, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
