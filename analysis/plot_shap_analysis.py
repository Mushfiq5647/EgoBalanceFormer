#!/usr/bin/env python3
"""
SHAP-style global and local explanation plots for CoP prediction.

This script is designed for:
  - dataset_type=estimated15
  - model_type=st_transformer_enhanced

Outputs:
  - global_feature_importance_copx.png
  - global_feature_importance_copy.png
  - global_modality_importance.png
  - local_explanation_copx.png
  - local_explanation_copy.png
  - shap_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    import shap  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: shap. Install it first in your env, e.g.:\n"
        "  pip install shap\n"
        f"Import error: {exc}"
    )

import matplotlib.pyplot as plt

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
    p.add_argument("--output_dir", default="log/shap_analysis")

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

    p.add_argument("--d_joint", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--n_spatial_layers", type=int, default=2)
    p.add_argument("--n_temporal_layers", type=int, default=3)
    p.add_argument("--n_fusion_layers", type=int, default=2)
    p.add_argument("--n_decoder_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--background_n", type=int, default=32)
    p.add_argument("--explain_n", type=int, default=64)
    p.add_argument("--nsamples", type=int, default=100)
    p.add_argument(
        "--l1_reg",
        type=str,
        default="num_features(20)",
        help="KernelSHAP l1 regularization (e.g., num_features(20), aic, bic, or 0)",
    )
    p.add_argument(
        "--grouped_only",
        action="store_true",
        help="Only save grouped modality SHAP outputs (no indexed per-feature plots).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _active_keys(args: argparse.Namespace, sample: Dict[str, torch.Tensor]) -> List[str]:
    keys: List[str] = ["pose_pos"]
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


def _flatten_sample(
    sample: Dict[str, torch.Tensor],
    keys: List[str],
    with_names: bool = False,
) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[int, int]]]:
    vals: List[np.ndarray] = []
    names: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    start = 0

    for k in keys:
        arr = sample[k].detach().cpu().numpy()
        flat = arr.reshape(-1).astype(np.float32)
        vals.append(flat)
        end = start + flat.shape[0]
        spans[k] = (start, end)
        if with_names:
            for i in range(flat.shape[0]):
                names.append(f"{k}[{i}]")
        start = end
    return np.concatenate(vals, axis=0), names, spans


def _unflatten_to_batch(
    x: np.ndarray,  # (B, F)
    template: Dict[str, torch.Tensor],
    keys: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    bsz = x.shape[0]
    out: Dict[str, torch.Tensor] = {}
    cursor = 0
    for k in keys:
        shp = tuple(template[k].shape)
        numel = int(np.prod(shp))
        chunk = x[:, cursor:cursor + numel].reshape((bsz,) + shp)
        out[k] = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
        cursor += numel
    return out


def _predict_factory(
    model: SpatioTemporalCoPPredictorEnhanced,
    template: Dict[str, torch.Tensor],
    keys: List[str],
    device: torch.device,
    normalizer: Normalizer,
    output_idx: int,
):
    def _predict(x_np: np.ndarray) -> np.ndarray:
        if x_np.ndim == 1:
            x_np_local = x_np[None, :]
        else:
            x_np_local = x_np
        with torch.no_grad():
            batch = _unflatten_to_batch(x_np_local.astype(np.float32), template, keys, device)
            batch = normalizer.normalize_batch(batch)
            pred = model(
                pose_pos=batch["pose_pos"],
                pose_vel=batch.get("pose_vel"),
                vr_pos=batch.get("vr_pos"),
                vr_vel=batch.get("vr_vel"),
                joint_groups=None if getattr(model, "use_joint_groups", True) is False else batch.get("joint_groups"),
                joint_rotations=None if getattr(model, "use_joint_rotations", True) is False else batch.get("joint_rotations"),
                joint_rot_vel=None if getattr(model, "use_joint_rotations", True) is False else batch.get("joint_rot_vel"),
                com=None if getattr(model, "use_com", True) is False else batch.get("com"),
                com_vel=None if getattr(model, "use_com", True) is False else batch.get("com_vel"),
            )
            pred = normalizer.denormalize_cop(pred).cpu().numpy()
            return pred[:, output_idx]
    return _predict


def _barh_top(
    labels: List[str],
    values: np.ndarray,
    title: str,
    out_path: str,
    topk: int = 20,
) -> None:
    idx = np.argsort(values)[::-1][:topk]
    vals = values[idx][::-1]
    labs = [labels[i] for i in idx[::-1]]
    plt.figure(figsize=(8, 6))
    plt.barh(labs, vals)
    plt.title(title)
    plt.xlabel("mean(|SHAP|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_modality_csv(
    modality_names: List[str],
    imp_x: List[float],
    imp_y: List[float],
    out_path: str,
) -> None:
    rows = []
    for i, m in enumerate(modality_names):
        mx = float(imp_x[i])
        my = float(imp_y[i])
        rows.append((m, mx, my, 0.5 * (mx + my)))
    rows.sort(key=lambda r: r[3], reverse=True)
    with open(out_path, "w") as f:
        f.write("modality,importance_copx,importance_copy,importance_avg\n")
        for m, mx, my, ma in rows:
            f.write(f"{_pretty_group_name(m)},{mx:.8f},{my:.8f},{ma:.8f}\n")


def _save_modality_percent_csv(
    modality_names: List[str],
    imp_x: List[float],
    imp_y: List[float],
    out_path: str,
) -> None:
    rows = []
    avg = np.array([0.5 * (float(imp_x[i]) + float(imp_y[i])) for i in range(len(modality_names))], dtype=np.float64)
    denom = float(avg.sum()) if float(avg.sum()) > 0 else 1.0
    pct = (avg / denom) * 100.0
    for i, m in enumerate(modality_names):
        rows.append((m, float(pct[i])))
    rows.sort(key=lambda r: r[1], reverse=True)
    with open(out_path, "w") as f:
        f.write("modality,importance_percent\n")
        for m, p in rows:
            f.write(f"{m},{p:.6f}\n")


def _plot_modality_ranked(
    modality_names: List[str],
    imp_x: List[float],
    imp_y: List[float],
    out_path: str,
) -> None:
    rows = []
    for i, m in enumerate(modality_names):
        mx = float(imp_x[i])
        my = float(imp_y[i])
        rows.append((m, mx, my, 0.5 * (mx + my)))
    rows.sort(key=lambda r: r[3], reverse=True)
    labels = [r[0] for r in rows][::-1]
    vals = [r[3] for r in rows][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(labels, vals)
    plt.xlabel("mean(|SHAP|), averaged over CoPX and CoPY")
    plt.title("Generalized modality ranking")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_modality_single_axis(
    modality_names: List[str],
    values: List[float],
    title: str,
    out_path: str,
) -> None:
    rows = [(_pretty_group_name(modality_names[i]), float(values[i])) for i in range(len(modality_names))]
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(labels, vals)
    plt.xlabel("mean(|SHAP value|)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_modality_ranked_percent(
    modality_names: List[str],
    imp_x: List[float],
    imp_y: List[float],
    out_path: str,
) -> None:
    avg = np.array([0.5 * (float(imp_x[i]) + float(imp_y[i])) for i in range(len(modality_names))], dtype=np.float64)
    denom = float(avg.sum()) if float(avg.sum()) > 0 else 1.0
    pct = (avg / denom) * 100.0
    rows = [(_pretty_group_name(modality_names[i]), float(pct[i])) for i in range(len(modality_names))]
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(labels, vals)
    plt.xlabel("Contribution (%) of total mean(|SHAP|)")
    plt.title("Normalized Modality Importance (SHAP %)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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


def _plot_mean_shap_feature_groups(
    modality_names: List[str],
    imp_x: List[float],
    imp_y: List[float],
    out_path: str,
) -> None:
    rows = []
    for i, m in enumerate(modality_names):
        mx = float(imp_x[i])
        my = float(imp_y[i])
        rows.append((_pretty_group_name(m), mx, my, 0.5 * (mx + my)))
    rows.sort(key=lambda r: r[3], reverse=True)
    labels = [r[0] for r in rows][::-1]
    vals = [r[3] for r in rows][::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals)
    plt.xlabel("mean(|SHAP value|) (average impact on model output)")
    plt.title("Mean SHAP Feature Importance (Grouped)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _local_plot(
    feature_names: List[str],
    x_row: np.ndarray,
    shap_row: np.ndarray,
    out_path: str,
    title: str,
    topk: int = 15,
) -> None:
    idx = np.argsort(np.abs(shap_row))[::-1][:topk]
    vals = shap_row[idx][::-1]
    labs = [f"{feature_names[i]} ({x_row[i]:.3f})" for i in idx[::-1]]
    colors = ["#d9534f" if v > 0 else "#5bc0de" for v in vals]
    plt.figure(figsize=(10, 6))
    plt.barh(labs, vals, color=colors)
    plt.title(title)
    plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = BalanceDatasetEstimated15(
        list_file=args.test_list,
        window_size=args.window_size,
        fps=args.fps,
        max_offset_ms=args.max_offset_ms,
        use_velocity=not args.no_velocity,
        use_vr=not args.no_vr,
        ae_checkpoint=args.ae_checkpoint,
        use_body_part=args.use_body_part,
        heatmap_dir=args.heatmap_dir,
        mad_threshold=args.mad_threshold,
        vr_mode=args.vr_mode,
        cop_target_mode=args.cop_target_mode,
        cop_k=args.cop_k,
    )

    n_total = len(ds)
    if n_total < 2:
        raise ValueError("Not enough samples for SHAP.")
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)
    bg_n = min(args.background_n, n_total // 2)
    ex_n = min(args.explain_n, n_total - bg_n)
    bg_idx = all_idx[:bg_n]
    ex_idx = all_idx[bg_n:bg_n + ex_n]
    print(f"Using background={bg_n}, explain={ex_n}, total={n_total}")

    sample0 = ds[int(bg_idx[0])]
    keys = _active_keys(args, sample0)
    print("Active feature keys:", keys)

    # Build model
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
        use_pose_vel=not args.no_velocity,
        use_com=not args.no_com,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    normalizer = Normalizer.load(args.normalizer)

    # Build flattened matrices
    bg_feats, ex_feats = [], []
    feature_names: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}

    for i, idx in enumerate(bg_idx):
        x, names, mod_spans = _flatten_sample(ds[int(idx)], keys, with_names=(i == 0))
        bg_feats.append(x)
        if i == 0:
            feature_names = names
            spans = mod_spans
    for idx in ex_idx:
        x, _, _ = _flatten_sample(ds[int(idx)], keys, with_names=False)
        ex_feats.append(x)

    bg_arr = np.stack(bg_feats, axis=0)
    ex_arr = np.stack(ex_feats, axis=0)

    # Template for unflattening
    template = {k: ds[int(bg_idx[0])][k].clone() for k in keys}

    pred_x = _predict_factory(model, template, keys, device, normalizer, output_idx=0)
    pred_y = _predict_factory(model, template, keys, device, normalizer, output_idx=1)

    print("Computing SHAP for CoPX ...")
    explainer_x = shap.KernelExplainer(pred_x, bg_arr)
    shap_x = explainer_x.shap_values(ex_arr, nsamples=args.nsamples, l1_reg=args.l1_reg)
    shap_x = np.asarray(shap_x, dtype=np.float64)  # (N, F)

    print("Computing SHAP for CoPY ...")
    explainer_y = shap.KernelExplainer(pred_y, bg_arr)
    shap_y = explainer_y.shap_values(ex_arr, nsamples=args.nsamples, l1_reg=args.l1_reg)
    shap_y = np.asarray(shap_y, dtype=np.float64)  # (N, F)

    # Global feature ranking (indexed flattened features) can be skipped.
    imp_x = np.mean(np.abs(shap_x), axis=0)
    imp_y = np.mean(np.abs(shap_y), axis=0)
    if not args.grouped_only:
        _barh_top(
            feature_names,
            imp_x,
            "Global SHAP ranking (CoPX)",
            os.path.join(args.output_dir, "global_feature_importance_copx.png"),
            topk=20,
        )
        _barh_top(
            feature_names,
            imp_y,
            "Global SHAP ranking (CoPY)",
            os.path.join(args.output_dir, "global_feature_importance_copy.png"),
            topk=20,
        )

    # Global modality ranking
    modality_names = list(spans.keys())
    modality_imp_x = []
    modality_imp_y = []
    for m in modality_names:
        s, e = spans[m]
        modality_imp_x.append(float(np.mean(np.abs(shap_x[:, s:e]))))
        modality_imp_y.append(float(np.mean(np.abs(shap_y[:, s:e]))))

    xloc = np.arange(len(modality_names))
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(xloc - width / 2, modality_imp_x, width=width, label="CoPX")
    plt.bar(xloc + width / 2, modality_imp_y, width=width, label="CoPY")
    plt.xticks(xloc, modality_names, rotation=20)
    plt.ylabel("mean(|SHAP|)")
    plt.title("Global modality importance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "global_modality_importance.png"), dpi=200)
    plt.close()
    _plot_modality_ranked(
        modality_names=modality_names,
        imp_x=modality_imp_x,
        imp_y=modality_imp_y,
        out_path=os.path.join(args.output_dir, "global_modality_importance_ranked.png"),
    )
    _save_modality_csv(
        modality_names=modality_names,
        imp_x=modality_imp_x,
        imp_y=modality_imp_y,
        out_path=os.path.join(args.output_dir, "global_modality_importance.csv"),
    )
    _plot_mean_shap_feature_groups(
        modality_names=modality_names,
        imp_x=modality_imp_x,
        imp_y=modality_imp_y,
        out_path=os.path.join(args.output_dir, "mean_shap_feature_groups.png"),
    )
    _plot_modality_ranked_percent(
        modality_names=modality_names,
        imp_x=modality_imp_x,
        imp_y=modality_imp_y,
        out_path=os.path.join(args.output_dir, "global_modality_importance_percent.png"),
    )
    _save_modality_percent_csv(
        modality_names=modality_names,
        imp_x=modality_imp_x,
        imp_y=modality_imp_y,
        out_path=os.path.join(args.output_dir, "global_modality_importance_percent.csv"),
    )
    _plot_modality_single_axis(
        modality_names=modality_names,
        values=modality_imp_x,
        title="Modality Mean SHAP (CoPX)",
        out_path=os.path.join(args.output_dir, "global_modality_importance_copx.png"),
    )
    _plot_modality_single_axis(
        modality_names=modality_names,
        values=modality_imp_y,
        title="Modality Mean SHAP (CoPY)",
        out_path=os.path.join(args.output_dir, "global_modality_importance_copy.png"),
    )

    # Local sample plots (first explained sample)
    if not args.grouped_only:
        _local_plot(
            feature_names=feature_names,
            x_row=ex_arr[0],
            shap_row=shap_x[0],
            out_path=os.path.join(args.output_dir, "local_explanation_copx.png"),
            title="Local explanation (CoPX) - sample 0",
            topk=15,
        )
        _local_plot(
            feature_names=feature_names,
            x_row=ex_arr[0],
            shap_row=shap_y[0],
            out_path=os.path.join(args.output_dir, "local_explanation_copy.png"),
            title="Local explanation (CoPY) - sample 0",
            topk=15,
        )

    summary = {
        "background_n": int(bg_n),
        "explain_n": int(ex_n),
        "num_features": int(ex_arr.shape[1]),
        "kernel_l1_reg": args.l1_reg,
        "feature_keys": keys,
        "modality_importance": {
            m: {"copx": modality_imp_x[i], "copy": modality_imp_y[i]}
            for i, m in enumerate(modality_names)
        },
        "top20_features_copx": [
            {"name": feature_names[i], "importance": float(imp_x[i])}
            for i in np.argsort(imp_x)[::-1][:20]
        ],
        "top20_features_copy": [
            {"name": feature_names[i], "importance": float(imp_y[i])}
            for i in np.argsort(imp_y)[::-1][:20]
        ],
    }
    with open(os.path.join(args.output_dir, "shap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved SHAP plots and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
