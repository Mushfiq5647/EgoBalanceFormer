#!/usr/bin/env python3
"""
Integrated Gradients (Captum) modality attribution for CoP prediction.

This script computes modality-level attributions for CoPX and CoPY separately
using a trained st_transformer_enhanced checkpoint on estimated15 data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from captum.attr import IntegratedGradients
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: captum. Install it first:\n"
        "  pip install captum\n"
        f"Import error: {exc}"
    )

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.dataset_estimated15 import BalanceDatasetEstimated15  # noqa
from data.normalization import Normalizer  # noqa
from models.st_transformer_enhanced import SpatioTemporalCoPPredictorEnhanced  # noqa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test_list", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--normalizer", required=True)
    p.add_argument("--ae_checkpoint", required=True)
    p.add_argument("--use_body_part", action="store_true")
    p.add_argument("--heatmap_dir", default="custom_pred_heatmaps")
    p.add_argument("--output_dir", default="log/ig_analysis")

    p.add_argument("--window_size", type=int, default=11)
    p.add_argument("--fps", type=float, default=22.0)
    p.add_argument("--max_offset_ms", type=float, default=50.0)
    p.add_argument("--mad_threshold", type=float, default=4.0)
    p.add_argument("--cop_target_mode", type=str, default="k4", choices=["k4", "seqmean", "absolute"])
    p.add_argument("--cop_k", type=int, default=10)
    p.add_argument("--vr_mode", type=str, default="hmd_only", choices=["rel_lr", "hmd_only", "hmd_lr"])

    p.add_argument("--no_vr", action="store_true")
    p.add_argument("--no_velocity", action="store_true")
    p.add_argument("--no_joint_groups", action="store_true")
    p.add_argument("--no_joint_rotations", action="store_true")
    p.add_argument("--no_com", action="store_true")
    p.add_argument("--no_pose", action="store_true")

    p.add_argument("--d_joint", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--n_spatial_layers", type=int, default=2)
    p.add_argument("--n_temporal_layers", type=int, default=3)
    p.add_argument("--n_fusion_layers", type=int, default=2)
    p.add_argument("--n_decoder_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--explain_n", type=int, default=64)
    p.add_argument("--ig_steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def _pretty_group_name(key: str) -> str:
    mapping = {
        "pose_pos": "Estimated Lower-Body Joint Positions",
        "pose_vel": "Estimated Lower-Body Joint Velocities",
        "joint_rotations": "Joint Rotations",
        "joint_rot_vel": "Rotational Velocities",
        "com": "Center of Mass (CoM)",
        "com_vel": "CoM Velocity",
        "vr_pos": "HMD Position",
        "vr_vel": "HMD Velocity",
        "joint_groups": "Joint Groups",
    }
    return mapping.get(key, key)


def _active_keys(args: argparse.Namespace, sample: Dict[str, torch.Tensor]) -> List[str]:
    keys: List[str] = []
    if not args.no_pose:
        keys.append("pose_pos")
        if (not args.no_velocity) and ("pose_vel" in sample):
            keys.append("pose_vel")
    if (not args.no_joint_groups) and ("joint_groups" in sample):
        keys.append("joint_groups")
    if (not args.no_joint_rotations) and ("joint_rotations" in sample):
        keys.append("joint_rotations")
        if (not args.no_velocity) and ("joint_rot_vel" in sample):
            keys.append("joint_rot_vel")
    if (not args.no_com) and ("com" in sample):
        keys.append("com")
        if (not args.no_velocity) and ("com_vel" in sample):
            keys.append("com_vel")
    if (not args.no_vr) and ("vr_pos" in sample):
        keys.append("vr_pos")
        if (not args.no_velocity) and ("vr_vel" in sample):
            keys.append("vr_vel")
    return keys


def _plot_modality(values: Dict[str, float], title: str, out_path: str) -> None:
    rows = sorted(values.items(), key=lambda x: x[1], reverse=True)
    labels = [_pretty_group_name(k) for k, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(labels, vals)
    plt.xlabel("mean(|Integrated Gradients|)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    ds = BalanceDatasetEstimated15(
        list_file=args.test_list,
        window_size=args.window_size,
        fps=args.fps,
        max_offset_ms=args.max_offset_ms,
        use_velocity=not args.no_velocity,
        use_vr=not args.no_vr,
        ae_checkpoint=args.ae_checkpoint,
        heatmap_dir=args.heatmap_dir,
        mad_threshold=args.mad_threshold,
        vr_mode=args.vr_mode,
        cop_target_mode=args.cop_target_mode,
        cop_k=args.cop_k,
        use_body_part=args.use_body_part,
    )

    normalizer = Normalizer.load(args.normalizer)
    sample0 = ds[0]
    active_keys = _active_keys(args, sample0)
    print("Active keys:", active_keys)
    if not active_keys:
        raise ValueError("No active modalities selected.")

    n_pose_joints = int(sample0["pose_pos"].shape[1])
    n_groups = int(sample0["joint_groups"].shape[1]) if "joint_groups" in sample0 else 4
    n_vr_coords = 0 if args.no_vr else int(sample0["vr_pos"].shape[-1])

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
        window_size=args.window_size,
        use_joint_groups=not args.no_joint_groups,
        use_joint_rotations=not args.no_joint_rotations,
        use_rotation_vel=not args.no_velocity,
        use_pose_vel=(not args.no_velocity) and (not args.no_pose),
        use_com=not args.no_com,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    idx_all = np.arange(len(ds))
    np.random.shuffle(idx_all)
    idx_use = idx_all[: min(args.explain_n, len(ds))]
    print(f"Explain samples: {len(idx_use)}")

    # Prepare base containers
    attr_sum_x = {k: 0.0 for k in active_keys}
    attr_sum_y = {k: 0.0 for k in active_keys}

    def _forward_func(*inputs):
        # inputs order follows active_keys
        mapped: Dict[str, torch.Tensor | None] = {k: None for k in [
            "pose_pos", "pose_vel", "joint_groups", "joint_rotations", "joint_rot_vel",
            "com", "com_vel", "vr_pos", "vr_vel",
        ]}
        for k, t in zip(active_keys, inputs):
            mapped[k] = t
        out = model(
            pose_pos=mapped["pose_pos"] if mapped["pose_pos"] is not None else torch.zeros_like(sample0["pose_pos"]).unsqueeze(0).to(device),
            pose_vel=mapped["pose_vel"],
            vr_pos=mapped["vr_pos"],
            vr_vel=mapped["vr_vel"],
            joint_groups=mapped["joint_groups"] if getattr(model, "use_joint_groups", True) else None,
            joint_rotations=mapped["joint_rotations"] if getattr(model, "use_joint_rotations", True) else None,
            joint_rot_vel=mapped["joint_rot_vel"] if getattr(model, "use_joint_rotations", True) else None,
            com=mapped["com"] if getattr(model, "use_com", True) else None,
            com_vel=mapped["com_vel"] if getattr(model, "use_com", True) else None,
        )
        return out

    ig = IntegratedGradients(_forward_func)

    for idx in idx_use:
        s = ds[int(idx)]
        b = {k: s[k].unsqueeze(0).to(device) for k in s if isinstance(s[k], torch.Tensor)}
        b = normalizer.normalize_batch(b)

        inputs = []
        baselines = []
        for k in active_keys:
            t = b[k].detach().clone().requires_grad_(True)
            inputs.append(t)
            baselines.append(torch.zeros_like(t))

        attrs_x = ig.attribute(tuple(inputs), baselines=tuple(baselines), target=0, n_steps=args.ig_steps)
        attrs_y = ig.attribute(tuple(inputs), baselines=tuple(baselines), target=1, n_steps=args.ig_steps)
        if not isinstance(attrs_x, tuple):
            attrs_x = (attrs_x,)
        if not isinstance(attrs_y, tuple):
            attrs_y = (attrs_y,)

        for k, ax, ay in zip(active_keys, attrs_x, attrs_y):
            attr_sum_x[k] += float(ax.detach().abs().mean().item())
            attr_sum_y[k] += float(ay.detach().abs().mean().item())

    n = float(len(idx_use))
    modality_x = {k: v / n for k, v in attr_sum_x.items()}
    modality_y = {k: v / n for k, v in attr_sum_y.items()}
    modality_avg = {k: 0.5 * (modality_x[k] + modality_y[k]) for k in active_keys}

    _plot_modality(modality_x, "Integrated Gradients by Modality (CoPX)", os.path.join(args.output_dir, "ig_modality_copx.png"))
    _plot_modality(modality_y, "Integrated Gradients by Modality (CoPY)", os.path.join(args.output_dir, "ig_modality_copy.png"))
    _plot_modality(modality_avg, "Integrated Gradients by Modality (Average)", os.path.join(args.output_dir, "ig_modality_avg.png"))

    # Percent view
    denom = sum(modality_avg.values()) if sum(modality_avg.values()) > 0 else 1.0
    pct = {k: (100.0 * v / denom) for k, v in modality_avg.items()}

    out = {
        "active_modalities": active_keys,
        "explain_n": int(len(idx_use)),
        "ig_steps": int(args.ig_steps),
        "modality_mean_abs_ig_copx": modality_x,
        "modality_mean_abs_ig_copy": modality_y,
        "modality_mean_abs_ig_avg": modality_avg,
        "modality_percent_avg": pct,
    }
    with open(os.path.join(args.output_dir, "ig_modality_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved IG modality analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
