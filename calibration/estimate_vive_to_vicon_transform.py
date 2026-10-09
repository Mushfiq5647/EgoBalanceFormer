"""
Estimate the similarity transformation (scale, rotation, translation) from
VIVE coordinate system to VICON coordinate system using the Umeyama algorithm.

Given N corresponding point pairs {p_vive_i, p_vicon_i}, solves for:
    p_vicon = s * R @ p_vive + t

where s is a uniform scale factor, R is a 3x3 rotation matrix, and t is a
3-vector translation.

Inputs:
    vr_data_vive.json  - 50 timestamps, each with HMD/Left/Right positions in VIVE space
    vr_data_vicon.json - 50 timestamps, each with HMD/Left/Right positions in VICON space

Output:
    vive_to_vicon_umeyama.json - estimated transformation parameters
"""

import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Umeyama similarity-transform estimator
# ---------------------------------------------------------------------------

def umeyama(src: np.ndarray, dst: np.ndarray):
    """
    Estimate the similarity transform T = (s, R, t) such that
        dst ≈ s * R @ src + t
    using the closed-form Umeyama (1991) algorithm.

    Parameters
    ----------
    src : (N, 3) array  – source points (VIVE)
    dst : (N, 3) array  – target points (VICON)

    Returns
    -------
    s   : float         – scale factor
    R   : (3, 3) array  – rotation matrix
    t   : (3,)  array   – translation vector
    """
    assert src.shape == dst.shape, "src and dst must have the same shape"
    n, m = src.shape  # n points, m dimensions

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = (src_c ** 2).sum() / n

    # Cross-covariance matrix
    H = dst_c.T @ src_c / n  # (m, m)

    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Correct for reflection
    d = np.ones(m)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        d[-1] = -1

    R = U @ np.diag(d) @ Vt

    s = (S * d).sum() / var_src

    t = mu_dst - s * R @ mu_src

    return s, R, t


# ---------------------------------------------------------------------------
# Load data and assemble point clouds
# ---------------------------------------------------------------------------

def load_points(vive_path: Path, vicon_path: Path):
    """
    Parse both JSON files and return stacked arrays of corresponding points.
    Automatically detects which devices are present (HMD, Left, Right).
    """
    with open(vive_path) as f:
        vive_data = json.load(f)
    with open(vicon_path) as f:
        vicon_data = json.load(f)

    assert len(vive_data) == len(vicon_data), (
        f"Length mismatch: {len(vive_data)} VIVE vs {len(vicon_data)} VICON entries"
    )

    vive_pts, vicon_pts = [], []

    # Detect which devices are present
    first_vive = vive_data[0]
    devices = []
    for device in ["HMD", "Left", "Right"]:
        if f"vr{device}_X" in first_vive:
            devices.append(device)
    
    print(f"Detected devices: {', '.join(devices)}")

    for vive, vicon in zip(vive_data, vicon_data):
        assert vive["timestamp"] == vicon["timestamp"], (
            f"Timestamp mismatch: {vive['timestamp']} vs {vicon['timestamp']}"
        )

        for device in devices:
            prefix_vr = f"vr{device}"
            # Check both naming conventions for vicon
            if f"vc{device}_X" in vicon:
                prefix_vc = f"vc{device}"
            else:
                # Assume same prefix as vive
                prefix_vc = prefix_vr
            
            vive_pts.append([vive[f"{prefix_vr}_X"],
                              vive[f"{prefix_vr}_Y"],
                              vive[f"{prefix_vr}_Z"]])
            vicon_pts.append([vicon[f"{prefix_vc}_X"],
                               vicon[f"{prefix_vc}_Y"],
                               vicon[f"{prefix_vc}_Z"]])

    return np.array(vive_pts, dtype=np.float64), np.array(vicon_pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Evaluate quality of the fit
# ---------------------------------------------------------------------------

def reprojection_stats(src, dst, s, R, t):
    """
    Apply the estimated transform and compute per-point residuals.
    """
    dst_est = s * (src @ R.T) + t
    residuals = np.linalg.norm(dst_est - dst, axis=1)
    return {
        "mean_error_m":   float(residuals.mean()),
        "median_error_m": float(np.median(residuals)),
        "max_error_m":    float(residuals.max()),
        "std_error_m":    float(residuals.std()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    vive_path  = HERE / "vr_data_vive.json"
    vicon_path = HERE / "vr_data_vicon.json"
    out_path   = HERE / "vive_to_vicon_umeyama.json"

    print(f"Loading VIVE data  : {vive_path}")
    print(f"Loading VICON data : {vicon_path}")

    src, dst = load_points(vive_path, vicon_path)
    n_timestamps = len(vive_data) if 'vive_data' in locals() else len(src) // 2
    n_devices = len(src) // len(json.load(open(vive_path)))
    print(f"Assembled {len(src)} corresponding point pairs "
          f"({len(src) // n_devices} timestamps × {n_devices} trackers)")

    # ---- Umeyama similarity transform ----
    s, R, t = umeyama(src, dst)

    stats = reprojection_stats(src, dst, s, R, t)

    # Build the 4×4 homogeneous similarity-transform matrix
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3,  3] = t

    print("\n--- Estimated Transform (VIVE → VICON) ---")
    print(f"Scale factor  s = {s:.8f}")
    print(f"Rotation  R =\n{R}")
    print(f"Translation  t = {t}")
    print(f"\n4×4 similarity matrix T:\n{T}")
    print(f"\nFit quality (residuals after applying transform):")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")

    # ---- Save results ----
    result = {
        "description": (
            "Similarity transform from VIVE to VICON coordinate system. "
            "Apply as: p_vicon = s * R @ p_vive + t  "
            "(equivalently, T_4x4 @ [p_vive; 1] gives [p_vicon; 1/s] — "
            "use the explicit s, R, t fields for clean application)."
        ),
        "n_point_pairs": int(len(src)),
        "n_timestamps":  int(len(src) // 3),
        "scale": float(s),
        "rotation_3x3": R.tolist(),
        "translation": t.tolist(),
        "transform_4x4_similarity": T.tolist(),
        "fit_quality": stats,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nSaved transformation to: {out_path}")


if __name__ == "__main__":
    main()
