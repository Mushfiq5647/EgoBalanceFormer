"""
Estimate global transformation from VIVE to VICON coordinate system
by pooling all VR data from all directories.

Reads:
  - vive_coordinates/*/vr_data.json  (VR data in VIVE coordinate system)
  - vicon_coordinates/*/vr_data.json (VR data in VICON coordinate system)

Outputs:
  - global_vive_to_vicon_transform.json (transformation parameters)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


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
# Load VR data from directories
# ---------------------------------------------------------------------------

def load_vr_data_from_dir(vr_data_path: Path, prefix="vr"):
    """
    Load VR data from a single vr_data.json file.
    Returns dict mapping timestamp -> {HMD, Left, Right} coordinates.
    Skips entries where ANY coordinate is zero.
    
    Parameters
    ----------
    vr_data_path : Path
        Path to vr_data.json file
    prefix : str
        Prefix for keys: "vr" for VIVE data, "vc" for VICON data
    """
    with open(vr_data_path) as f:
        data = json.load(f)
    
    vr_dict = {}
    skipped = 0
    
    for entry in data:
        timestamp = entry["timestamp"]
        
        # Extract coordinates with appropriate prefix
        try:
            hmd = np.array([entry[f"{prefix}HMD_X"], entry[f"{prefix}HMD_Y"], entry[f"{prefix}HMD_Z"]])
            left = np.array([entry[f"{prefix}Left_X"], entry[f"{prefix}Left_Y"], entry[f"{prefix}Left_Z"]])
            right = np.array([entry[f"{prefix}Right_X"], entry[f"{prefix}Right_Y"], entry[f"{prefix}Right_Z"]])
        except KeyError:
            # Skip if keys don't match expected prefix
            skipped += 1
            continue
        
        # Check if ANY coordinate is zero - if so, skip this entry
        all_coords = np.concatenate([hmd, left, right])
        if np.any(all_coords == 0.0):
            skipped += 1
            continue
        
        vr_dict[timestamp] = {
            "HMD": hmd,
            "Left": left,
            "Right": right,
        }
    
    return vr_dict, skipped


def load_all_vr_data(base_dir: Path, prefix="vr"):
    """
    Load all vr_data.json files from subdirectories.
    Returns dict: subject_name -> timestamp -> {HMD, Left, Right}
    Also returns total skipped count.
    
    Parameters
    ----------
    base_dir : Path
        Base directory containing subject subdirectories
    prefix : str
        Prefix for keys: "vr" for VIVE data, "vc" for VICON data
    """
    all_data = {}
    total_skipped = 0
    
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        vr_data_file = subdir / "vr_data.json"
        if not vr_data_file.exists():
            continue
        
        subject_name = subdir.name
        subject_data, skipped = load_vr_data_from_dir(vr_data_file, prefix=prefix)
        all_data[subject_name] = subject_data
        total_skipped += skipped
    
    return all_data, total_skipped


def match_corresponding_points(vive_data, vicon_data):
    """
    Match corresponding points between VIVE and VICON data.
    Only includes timestamps where BOTH VIVE and VICON have valid (non-zero) data.
    Returns two arrays: src_points (VIVE), dst_points (VICON).
    """
    src_points = []
    dst_points = []
    
    stats = {
        "total_subjects_vive": len(vive_data),
        "total_subjects_vicon": len(vicon_data),
        "matched_subjects": 0,
        "total_timestamps_checked": 0,
        "matched_timestamps": 0,
        "total_points": 0,
    }
    
    # Find common subjects
    common_subjects = set(vive_data.keys()) & set(vicon_data.keys())
    stats["matched_subjects"] = len(common_subjects)
    
    print(f"\nProcessing {len(common_subjects)} common subjects...")
    
    for subject in sorted(common_subjects):
        vive_subject = vive_data[subject]
        vicon_subject = vicon_data[subject]
        
        # Find common timestamps (both already filtered for non-zero values)
        common_timestamps = set(vive_subject.keys()) & set(vicon_subject.keys())
        stats["total_timestamps_checked"] += max(len(vive_subject), len(vicon_subject))
        stats["matched_timestamps"] += len(common_timestamps)
        
        for timestamp in sorted(common_timestamps):
            vive_frame = vive_subject[timestamp]
            vicon_frame = vicon_subject[timestamp]
            
            # Add all three devices (HMD, Left, Right)
            # Both have already been filtered for zeros, so we can safely add them
            for device in ["HMD", "Left", "Right"]:
                src_points.append(vive_frame[device])
                dst_points.append(vicon_frame[device])
                stats["total_points"] += 1
        
        # Progress indicator
        if len(common_subjects) > 10 and (list(sorted(common_subjects)).index(subject) + 1) % 10 == 0:
            print(f"  Processed {list(sorted(common_subjects)).index(subject) + 1}/{len(common_subjects)} subjects...")
    
    return np.array(src_points), np.array(dst_points), stats


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
        "min_error_m":    float(residuals.min()),
        "std_error_m":    float(residuals.std()),
        "percentile_95_m": float(np.percentile(residuals, 95)),
        "percentile_99_m": float(np.percentile(residuals, 99)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys
    
    # Paths - can be overridden via command line arguments
    if len(sys.argv) >= 3:
        vive_dir = Path(sys.argv[1])
        vicon_dir = Path(sys.argv[2])
        output_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path("global_vive_to_vicon_transform.json")
    else:
        vive_dir = Path("/data/My_Backup/Dataset/vive_coordinates")
        vicon_dir = Path("/data/My_Backup/Dataset/vicon_coordinates")
        output_path = Path(__file__).parent / "global_vive_to_vicon_transform.json"
    
    # Check if directories exist
    if not vive_dir.exists():
        print(f"\n✗ ERROR: VIVE directory not found: {vive_dir}")
        print("\nUsage: python estimate_global_vive_to_vicon.py <vive_dir> <vicon_dir> [output_file]")
        return
    
    if not vicon_dir.exists():
        print(f"\n✗ ERROR: VICON directory not found: {vicon_dir}")
        print("\nUsage: python estimate_global_vive_to_vicon.py <vive_dir> <vicon_dir> [output_file]")
        return
    
    print("=" * 80)
    print("GLOBAL VIVE → VICON TRANSFORMATION ESTIMATION")
    print("=" * 80)
    
    # Load all VR data
    print(f"\nLoading VIVE data from:  {vive_dir}")
    vive_data, vive_skipped = load_all_vr_data(vive_dir, prefix="vr")
    print(f"  Loaded {len(vive_data)} subjects")
    print(f"  Skipped {vive_skipped} entries with zero values")
    
    print(f"\nLoading VICON data from: {vicon_dir}")
    vicon_data, vicon_skipped = load_all_vr_data(vicon_dir, prefix="vc")
    print(f"  Loaded {len(vicon_data)} subjects")
    print(f"  Skipped {vicon_skipped} entries with zero values")
    
    # Match corresponding points
    print("\nMatching corresponding points...")
    src, dst, stats = match_corresponding_points(vive_data, vicon_data)
    
    print(f"\nMatching statistics:")
    print(f"  Total subjects in VIVE:      {stats['total_subjects_vive']}")
    print(f"  Total subjects in VICON:     {stats['total_subjects_vicon']}")
    print(f"  Matched subjects:            {stats['matched_subjects']}")
    print(f"  Matched timestamps:          {stats['matched_timestamps']}")
    print(f"  Total point correspondences: {stats['total_points']}")
    print(f"    ({stats['total_points'] // 3} frames × 3 devices)")
    print(f"\n  Note: Only timestamps with ALL non-zero values in BOTH systems are used.")
    
    if len(src) == 0:
        print("\n✗ ERROR: No matching data found!")
        return
    
    # Estimate transformation using Umeyama
    print("\n" + "=" * 80)
    print("ESTIMATING TRANSFORMATION")
    print("=" * 80)
    
    s, R, t = umeyama(src, dst)
    
    # Build 4×4 homogeneous similarity-transform matrix
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3,  3] = t
    
    print(f"\nScale factor  s = {s:.8f}")
    print(f"\nRotation matrix R =")
    print(R)
    print(f"\nTranslation vector t = {t}")
    print(f"\n4×4 similarity matrix T =")
    print(T)
    
    # Evaluate fit quality
    fit_stats = reprojection_stats(src, dst, s, R, t)
    
    print("\n" + "=" * 80)
    print("FIT QUALITY (Reprojection Errors)")
    print("=" * 80)
    print(f"\n  Mean error:       {fit_stats['mean_error_m']*1000:.2f} mm ({fit_stats['mean_error_m']*100:.2f} cm)")
    print(f"  Median error:     {fit_stats['median_error_m']*1000:.2f} mm ({fit_stats['median_error_m']*100:.2f} cm)")
    print(f"  Std error:        {fit_stats['std_error_m']*1000:.2f} mm ({fit_stats['std_error_m']*100:.2f} cm)")
    print(f"  Min error:        {fit_stats['min_error_m']*1000:.2f} mm")
    print(f"  Max error:        {fit_stats['max_error_m']*1000:.2f} mm ({fit_stats['max_error_m']*100:.2f} cm)")
    print(f"  95th percentile:  {fit_stats['percentile_95_m']*1000:.2f} mm")
    print(f"  99th percentile:  {fit_stats['percentile_99_m']*1000:.2f} mm")
    
    # Save results
    result = {
        "description": (
            "Global similarity transform from VIVE to VICON coordinate system, "
            "estimated from all subjects and trials. "
            "Apply as: p_vicon = s * R @ p_vive + t"
        ),
        "matching_stats": stats,
        "scale": float(s),
        "rotation_3x3": R.tolist(),
        "translation": t.tolist(),
        "transform_4x4_similarity": T.tolist(),
        "fit_quality": fit_stats,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ Transformation saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
