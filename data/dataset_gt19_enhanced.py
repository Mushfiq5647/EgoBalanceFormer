"""
BalanceDatasetGT19Enhanced:
  Enhanced variant with better feature engineering for CoP prediction.

Key improvements over dataset_gt19_root_aligned:
  1. Temporal smoothing of pose/VR (reduce jitter)
  2. Velocity features (motion is more predictive than static position)
  3. Joint grouping (aggregate stable features from anatomical groups)
  4. Approximate center-of-mass computation (weighted by joint proxies)
  5. Per-CoP timestamp alignment instead of fixed 11-frame indexing

Units (consistent for model and loss):
  ── Positions (length): all in CENTIMETERS (cm) ──
  pose_pos    : (W, 11, 3)  joint positions, smoothed, root-relative, lower-body only
                Source: ground_truth joints "translation" in mm → ÷10 → cm
  vr_pos      : (W, 6)      [Left-HMD, Right-HMD], smoothed, HMD-relative
                Source: VR in meters → (left-hmd)*100, (right-hmd)*100 → cm
  joint_groups: (W, 5, 3)  aggregated positions [pelvis, torso, left_leg, right_leg, arms] in cm
  com         : (W, 3)     center of mass in cm
  cop         : (2,)       [CoPX, CoPY] target, in cm (from balance.json)

  ── Velocities: all in CENTIMETERS PER SECOND (cm/s) ──
  pose_vel    : (W, 11, 3)  central-diff velocity of pose_pos, then smoothed
  vr_vel      : (W, 6)      central-diff velocity of vr_pos, then smoothed
  Formula: vel[t] = (pos[t+1] - pos[t-1])/2 * fps  (with endpoints handled)
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import uniform_filter1d

from data.dataset import _frame_number


COP_FPS = 2.0


@dataclass(frozen=True)
class EnhancedSample:
    sequence_dir: str
    start_frame:  int
    cop_index:    int


def _parse_ts(ts: str) -> float:
    """Parse 'YYYY-MM-DD-HH-MM-SS-mmm' → seconds as float."""
    parts = ts.split("-")
    if len(parts) == 7:
        dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                      int(parts[3]), int(parts[4]), int(parts[5]))
        return dt.timestamp() + int(parts[6]) / 1000.0
    raise ValueError(f"Unexpected timestamp format: {ts!r}")


def _smooth_temporal(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth along time axis (axis=0) with uniform filter.
    Args:
        arr: (N, ...) array
        window: smoothing window size
    Returns:
        smoothed array same shape
    """
    if window <= 1:
        return arr
    return uniform_filter1d(arr, size=window, axis=0, mode="nearest")


def _compute_velocity(pos: np.ndarray, fps: float = 22.0, smooth_window: int = 3) -> np.ndarray:
    """
    Central difference velocity with optional smoothing.
    Args:
        pos: (N, ...) positions (e.g. in cm)
        fps: frame rate (frames per second)
        smooth_window: smoothing window for velocity
    Returns:
        vel: (N, ...) velocities in (position_units)/s (e.g. cm/s if pos in cm)
    """
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0 * fps  # central diff
    vel[0]    = (pos[1] - pos[0]) * fps           # forward diff
    vel[-1]   = (pos[-1] - pos[-2]) * fps         # backward diff
    if smooth_window > 1:
        vel = _smooth_temporal(vel, smooth_window)
    return vel


def _extract_lower_body_rotations_from_gt19(gt_rotations_19: np.ndarray, joint_indices: List[int]) -> np.ndarray:
    """
    Extract lower-body joint rotations (Euler angles) from 19-joint GT rotations.
    
    Args:
        gt_rotations_19: (N, 19, 3) Euler angles in degrees
        joint_indices: List of 8 joint indices to extract (e.g., [7, 8, 9, 10, 15, 16, 17, 18])
    
    Returns:
        rotations_8: (N, 8, 3) Euler angles in radians (converted from degrees)
    """
    # Extract and convert to radians
    rotations_8 = gt_rotations_19[:, joint_indices, :] * (np.pi / 180.0)  # (N, 8, 3)
    
    return rotations_8


def _compute_joint_angles_lower_body_19(pose_19: np.ndarray) -> np.ndarray:
    """
    Compute 6 lower-body joint angles from 19-joint pose.
    
    NOTE: This function is now deprecated in favor of using GT rotations directly.
    It is kept for backward compatibility.
    
    Args:
        pose_19: (N, 19, 3) full 19-joint pose, sequence-start-relative
    
    Returns:
        angles: (N, 6) [R_hip, R_knee, R_ankle, L_hip, L_knee, L_ankle] in radians
    """
    N = pose_19.shape[0]
    angles = np.zeros((N, 6), dtype=np.float32)
    
    # Extract lower-body joints (indices in 19-joint format)
    # Left leg: 7=L_Femur, 8=L_Tibia, 9=L_Foot, 10=L_Toe
    # Right leg: 15=R_Femur, 16=R_Tibia, 17=R_Foot, 18=R_Toe
    
    # Right leg
    r_hip = pose_19[:, 15, :]    # R_Femur (N, 3)
    r_knee = pose_19[:, 16, :]   # R_Tibia
    r_ankle = pose_19[:, 17, :]  # R_Foot
    r_foot = pose_19[:, 18, :]   # R_Toe
    
    # Left leg
    l_hip = pose_19[:, 7, :]     # L_Femur
    l_knee = pose_19[:, 8, :]    # L_Tibia
    l_ankle = pose_19[:, 9, :]   # L_Foot
    l_foot = pose_19[:, 10, :]   # L_Toe
    
    # Vertical reference
    vertical = np.array([0, 0, 1], dtype=np.float32)
    
    # Helper function to compute angle between two vectors
    def angle_between(v1, v2):
        """Compute angle in radians between vectors v1 and v2."""
        v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8
        v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8
        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm
        cos_angle = np.clip(np.sum(v1_unit * v2_unit, axis=-1), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    # Right hip angle (thigh vs. vertical)
    r_thigh = r_knee - r_hip  # (N, 3)
    angles[:, 0] = angle_between(r_thigh, np.tile(vertical, (N, 1)))
    
    # Right knee angle (thigh vs. shank)
    r_shank = r_ankle - r_knee
    angles[:, 1] = angle_between(r_thigh, r_shank)
    
    # Right ankle angle (shank vs. foot)
    r_foot_vec = r_foot - r_ankle
    angles[:, 2] = angle_between(r_shank, r_foot_vec)
    
    # Left hip angle
    l_thigh = l_knee - l_hip
    angles[:, 3] = angle_between(l_thigh, np.tile(vertical, (N, 1)))
    
    # Left knee angle
    l_shank = l_ankle - l_knee
    angles[:, 4] = angle_between(l_thigh, l_shank)
    
    # Left ankle angle
    l_foot_vec = l_foot - l_ankle
    angles[:, 5] = angle_between(l_shank, l_foot_vec)
    
    return angles


def _filter_outliers_mad(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Filter outliers using Median Absolute Deviation (MAD).
    
    Args:
        data: (N, D) array (e.g. CoP samples)
        threshold: MAD threshold (default 3.5, typical for outlier detection)
    
    Returns:
        mask: (N,) boolean array, True for inliers
    """
    if len(data) == 0:
        return np.ones(len(data), dtype=bool)
    
    # Compute MAD per dimension
    median = np.median(data, axis=0)  # (D,)
    abs_dev = np.abs(data - median)   # (N, D)
    mad = np.median(abs_dev, axis=0)  # (D,)
    
    # Modified z-score per dimension
    # Use 1.4826 as the constant to make MAD comparable to std for normal distributions
    mad_z_score = abs_dev / (1.4826 * mad + 1e-8)  # (N, D)
    
    # Keep samples where ALL dimensions are within threshold
    mask = np.all(mad_z_score < threshold, axis=1)  # (N,)
    
    return mask


# Segment mass fractions (approximate adult human proportions)
# Based on biomechanics literature (Winter 2009, de Leva 1996)
SEGMENT_MASSES = {
    "head":       0.081,
    "torso":      0.497,  # trunk + neck
    "upper_arm":  0.028,  # per arm
    "forearm":    0.016,  # per arm
    "hand":       0.006,  # per hand
    "thigh":      0.100,  # per thigh
    "shank":      0.0465, # per shank
    "foot":       0.0145, # per foot
}

# Joint index → segment mapping for raw 19-joint skeleton
# Joint order:
#   0=Root, 1=LowerBack, 2=Head,
#   3=L_Collar, 4=L_Humerus, 5=L_Elbow, 6=L_Wrist,
#   7=L_Femur, 8=L_Tibia, 9=L_Foot, 10=L_Toe,
#   11=R_Collar, 12=R_Humerus, 13=R_Elbow, 14=R_Wrist,
#   15=R_Femur, 16=R_Tibia, 17=R_Foot, 18=R_Toe
JOINT_SEGMENT_MASS = np.array([
    0.497,  # 0: Root / pelvis
    0.497,  # 1: LowerBack / torso
    0.081,  # 2: Head
    0.028,  # 3: L_Collar / shoulder girdle proxy
    0.028,  # 4: L_Humerus
    0.016,  # 5: L_Elbow / forearm proxy
    0.006,  # 6: L_Wrist / hand proxy
    0.100,  # 7: L_Femur
    0.0465, # 8: L_Tibia
    0.0145, # 9: L_Foot
    0.0145, # 10: L_Toe
    0.028,  # 11: R_Collar / shoulder girdle proxy
    0.028,  # 12: R_Humerus
    0.016,  # 13: R_Elbow / forearm proxy
    0.006,  # 14: R_Wrist / hand proxy
    0.100,  # 15: R_Femur
    0.0465, # 16: R_Tibia
    0.0145, # 17: R_Foot
    0.0145, # 18: R_Toe
], dtype=np.float32)
JOINT_SEGMENT_MASS /= JOINT_SEGMENT_MASS.sum()  # normalize to sum=1


def _compute_center_of_mass(pose: np.ndarray) -> np.ndarray:
    """
    Compute weighted center of mass from 19-joint pose.
    Args:
        pose: (N, 19, 3) in cm
    Returns:
        com: (N, 3) in cm
    """
    return np.einsum("njd,j->nd", pose, JOINT_SEGMENT_MASS)


# Lower-body joint indices for balance prediction (legs only, 8 joints)
# Mapping from 19-joint skeleton to lower-body subset
LOWER_BODY_JOINT_INDICES = [7, 8, 9, 10, 15, 16, 17, 18]
LOWER_BODY_NAMES = [
    "L_Femur", "L_Tibia", "L_Foot", "L_Toe",
    "R_Femur", "R_Tibia", "R_Foot", "R_Toe",
]

# Joint groups for aggregated features
JOINT_GROUPS = {
    "pelvis":    [0, 1],                 # Root, LowerBack
    "torso":     [2, 3, 11],             # Head + collars
    "left_leg":  [7, 8, 9, 10],          # L_Femur, L_Tibia, L_Foot, L_Toe
    "right_leg": [15, 16, 17, 18],       # R_Femur, R_Tibia, R_Foot, R_Toe
    "arms":      [4, 5, 6, 12, 13, 14],  # humerus, elbow, wrist
}


def _aggregate_joint_groups(pose: np.ndarray) -> np.ndarray:
    """
    Aggregate joints into 5 stable groups by averaging.
    Args:
        pose: (N, 19, 3)
    Returns:
        groups: (N, 5, 3)  [pelvis, torso, left_leg, right_leg, arms]
    """
    N = pose.shape[0]
    groups = np.zeros((N, 5, 3), dtype=np.float32)
    for i, (name, indices) in enumerate(JOINT_GROUPS.items()):
        groups[:, i, :] = pose[:, indices, :].mean(axis=1)
    return groups


class BalanceDatasetGT19Enhanced(Dataset):
    """
    Enhanced 19-joint dataset with:
      - Temporal smoothing
      - Sequence-start-relative pose (root joint = pelvis lean trajectory)
      - CoM computed from ABSOLUTE pose then made sequence-start-relative
        → horizontal displacement directly correlates with CoP
      - Velocities (optional, --no_velocity to disable)
      - Joint grouping (seq-start-relative)
      - max_offset_ms=50 for tight temporal alignment (at 22fps, nearest frame ≤23ms)
    """

    def __init__(
        self,
        list_file: str,
        window_size: int = 11,
        fps: float = 22.0,
        pose_smooth_window: int = 5,
        vel_smooth_window: int = 3,
        max_offset_ms: float = 50.0,
        joint_indices: Optional[List[int]] = None,
        use_velocity: bool = True,
        use_vr: bool = True,
        mad_threshold: Optional[float] = None,
        **kwargs,
    ):
        self.window_size = window_size
        self.fps = fps
        self.pose_smooth_window = pose_smooth_window
        self.vel_smooth_window = vel_smooth_window
        self.max_offset_ms = max_offset_ms
        self.use_velocity = use_velocity
        self.use_vr = use_vr
        self.mad_threshold = mad_threshold
        self.joint_indices: List[int] = (
            joint_indices if joint_indices is not None else LOWER_BODY_JOINT_INDICES
        )

        with open(list_file) as f:
            self.sequence_dirs = [
                l.strip() for l in f if l.strip() and not l.startswith("#")
            ]

        self._poses:        Dict[str, np.ndarray] = {}  # (N, 8, 3) lower-body only
        self._pose_vel:     Dict[str, np.ndarray] = {}  # (N, 8, 3) lower-body vel
        self._vr:           Dict[str, np.ndarray] = {}  # (N,6) smoothed
        self._vr_vel:       Dict[str, np.ndarray] = {}  # (N,6) smoothed vel
        self._joint_groups: Dict[str, np.ndarray] = {}  # (N,5,3)
        self._joint_rotations: Dict[str, np.ndarray] = {}  # (N,8,3) lower-body rotations (Euler, radians)
        self._com:          Dict[str, np.ndarray] = {}  # (N,3)
        self._com_vel:      Dict[str, np.ndarray] = {}  # (N,3) CoM velocity cm/s
        self._cop:          Dict[str, np.ndarray] = {}  # (M,2) mean-subtracted deviations
        self._cop_mean:     Dict[str, np.ndarray] = {}  # (2,) per-sequence CoP mean (cm)

        self.samples: List[EnhancedSample] = []
        skipped_seqs = 0
        total_pose_skipped = 0
        total_cop_skipped = 0

        for seq_dir in self.sequence_dirs:
            try:
                result = self._load_and_align(seq_dir)
                (poses, pose_vel, vr, vr_vel, groups, com, com_vel, rotations, cop,
                 samples, pose_skip, cop_skip) = result
                if not samples:
                    print(f"  [skip] {os.path.basename(seq_dir)}: 0 aligned windows")
                    skipped_seqs += 1
                    continue
                self._poses[seq_dir] = poses
                self._pose_vel[seq_dir] = pose_vel
                self._vr[seq_dir] = vr
                self._vr_vel[seq_dir] = vr_vel
                self._joint_groups[seq_dir] = groups
                self._joint_rotations[seq_dir] = rotations
                self._com[seq_dir] = com
                self._com_vel[seq_dir] = com_vel
                
                # Compute per-sequence mean and deviation
                cop_mean = cop.mean(axis=0)                  # (2,) per-sequence neutral CoP
                self._cop_mean[seq_dir] = cop_mean
                self._cop[seq_dir] = cop - cop_mean          # (M,2) deviation from neutral
                self.samples.extend(samples)
                total_pose_skipped += pose_skip
                total_cop_skipped += cop_skip
            except (FileNotFoundError, ValueError) as e:
                print(f"  [skip] {os.path.basename(seq_dir)}: {e}")
                skipped_seqs += 1

        if not self.samples:
            raise ValueError(f"No aligned windows from '{list_file}'")

        n_j = len(self.joint_indices)
        vel_str = "pos+vel" if self.use_velocity else "pos only"
        mad_str = f", MAD={self.mad_threshold}" if self.mad_threshold else ""
        print(
            f"[Enhanced] {len(self.samples)} windows from "
            f"{len(self._poses)}/{len(self.sequence_dirs)} sequences "
            f"(joints={n_j} lower-body, {vel_str}, skipped {skipped_seqs} seqs, "
            f"pose_smooth={pose_smooth_window}, vel_smooth={vel_smooth_window}, "
            f"max_offset_ms={self.max_offset_ms:.1f}{mad_str})"
        )

    # ------------------------------------------------------------------
    def _load_and_align(self, seq_dir: str):
        """
        Returns (poses, pose_vel, vr, vr_vel, joint_groups, com, com_vel, cop,
                 samples, pose_frames_skipped, cop_samples_skipped).
        """
        gt_path  = os.path.join(seq_dir, "ground_truth.json")
        bal_path = os.path.join(seq_dir, "balance.json")
        if not os.path.exists(gt_path):
            raise FileNotFoundError("Missing ground_truth.json")
        if not os.path.exists(bal_path):
            raise FileNotFoundError("Missing balance.json")

        # ---- metadata timestamps (optional fallback) ----
        frame_to_ts: Dict[int, str] = {}
        meta_files = list(Path(seq_dir).glob("metadata_*.txt"))
        if meta_files:
            try:
                with open(meta_files[0], newline="", errors="ignore") as f:
                    for row in csv.DictReader(f):
                        frame_to_ts[int(row["FrameIndex"])] = row["ColorTimestamp"]
            except (csv.Error, KeyError, ValueError):
                frame_to_ts = {}

        # ---- pose + VR from ground_truth.json ----
        with open(gt_path) as f:
            gt_entries = json.load(f)

        valid = []
        for entry in gt_entries:
            if not isinstance(entry, dict):
                continue
            image_name = entry.get("image_name")
            if not image_name:
                continue
            fid = _frame_number(str(image_name))

            joints = entry.get("joints")
            if not isinstance(joints, dict) or "translation" not in joints:
                continue
            pose = np.asarray(joints["translation"], dtype=np.float32)
            if pose.shape != (19, 3):
                continue
            pose = pose / 10.0  # mm → cm (NOT root-relative yet)
            
            # Load rotations (Euler angles in degrees)
            rotation = np.asarray(joints.get("rotation", np.zeros((19, 3))), dtype=np.float32)
            if rotation.shape != (19, 3):
                rotation = np.zeros((19, 3), dtype=np.float32)

            hmd   = entry.get("vrHMD")
            left  = entry.get("vrLeft")
            right = entry.get("vrRight")
            if hmd is None or left is None or right is None:
                continue
            hmd   = np.asarray(hmd,   dtype=np.float32)
            left  = np.asarray(left,  dtype=np.float32)
            right = np.asarray(right, dtype=np.float32)
            if hmd.shape != (3,) or left.shape != (3,) or right.shape != (3,):
                continue
            # Convert VR to cm first
            hmd_cm   = hmd * 100.0
            left_cm  = left * 100.0
            right_cm = right * 100.0

            if not self.use_vr:
                # Pose-only experiment: VR branch disabled.
                vr = np.zeros(6, dtype=np.float32)
            else:
                # Keep only HMD-relative controllers in cm.
                left_rel  = left_cm  - hmd_cm
                right_rel = right_cm - hmd_cm
                vr = np.concatenate([left_rel, right_rel]).astype(np.float32)

            ts_str = entry.get("timestamp") or frame_to_ts.get(fid)
            if not ts_str:
                continue
            try:
                ts_sec = _parse_ts(ts_str)
            except ValueError:
                continue
            valid.append((fid, ts_sec, pose, vr, rotation))

        if not valid:
            raise ValueError("No valid pose+VR frames")
        valid.sort(key=lambda x: x[1])

        pose_ts = np.asarray([v[1] for v in valid], dtype=np.float64)
        poses_raw = np.stack([v[2] for v in valid])     # (N, 19, 3) cm, NOT root-relative
        vr_raw    = np.stack([v[3] for v in valid])     # (N, 6)
        rotations_raw = np.stack([v[4] for v in valid]) # (N, 19, 3) Euler angles in degrees

        # ---- CoP from balance.json ----
        with open(bal_path) as f:
            cop_data = json.load(f)

        cop_list: List[List[float]] = []
        cop_ts_list: List[float] = []
        for entry in cop_data:
            if not isinstance(entry, dict):
                continue
            x = float(entry.get("copX") or entry.get("CoPX", 0.0))
            y = float(entry.get("copY") or entry.get("CoPY", 0.0))
            ts = entry.get("timestamp")
            if not ts:
                continue
            try:
                t = _parse_ts(ts)
            except ValueError:
                continue
            cop_ts_list.append(t)
            cop_list.append([x, y])

        if not cop_list:
            raise ValueError("No valid CoP samples")
        cop_all = np.array(cop_list, dtype=np.float32)  # (M, 2)
        cop_ts = np.asarray(cop_ts_list, dtype=np.float64)
        
        # ---- MAD outlier filtering (BEFORE alignment) ----
        if self.mad_threshold is not None and self.mad_threshold > 0:
            inlier_mask = _filter_outliers_mad(cop_all, self.mad_threshold)
            n_outliers = (~inlier_mask).sum()
            if n_outliers > 0:
                cop_all = cop_all[inlier_mask]
                cop_ts = cop_ts[inlier_mask]
                print(f"    [MAD] {os.path.basename(seq_dir)}: removed {n_outliers}/{len(inlier_mask)} CoP outliers (before alignment)")

        # ---- 1. Temporal smoothing ----
        poses_smooth = _smooth_temporal(poses_raw, self.pose_smooth_window)  # (N,19,3)
        vr_smooth    = _smooth_temporal(vr_raw, self.pose_smooth_window)     # (N,6)

        # ---- 2. Compute CoM from ABSOLUTE pose (BEFORE any subtraction) ----
        # Per-frame root-relative CoM would always be near zero and carry no lean signal.
        # Here we compute absolute CoM then make it sequence-start-relative so the
        # horizontal displacement over time captures lean — the primary predictor of CoP.
        com_abs = _compute_center_of_mass(poses_smooth)          # (N,3) absolute world coords
        com = com_abs - com_abs[0:1, :]                          # (N,3) displacement from start
        com_vel = _compute_velocity(com, self.fps, self.vel_smooth_window)  # (N,3) cm/s

        # ---- 3. Sequence-start-relative pose (REPLACES per-frame root-relative) ----
        # Per-frame root-relative: each frame's root is (0,0,0) → lean is erased.
        # Sequence-start-relative: subtract only the FIRST frame's root position from
        # all frames, so root joint trajectory = lean signal over time.
        seq_origin = poses_smooth[0:1, 0:1, :].copy()           # (1,1,3) first frame root
        poses_smooth = poses_smooth - seq_origin                  # (N,19,3) seq-start-relative

        # ---- 4. Joint grouping (from seq-start-relative pose) ----
        joint_groups = _aggregate_joint_groups(poses_smooth)     # (N,5,3)

        # ---- 5. Extract lower-body rotations from GT (8 joints, 3 Euler angles each) ----
        joint_rotations = _extract_lower_body_rotations_from_gt19(rotations_raw, self.joint_indices)  # (N, 8, 3) radians

        # ---- 6. Extract only lower-body joints (8 joints: legs only) ----
        pose_pos_out = poses_smooth[:, self.joint_indices, :]    # (N, 8, 3)

        # ---- 7. Compute velocities (or zeros if disabled) ----
        if self.use_velocity:
            pose_vel = _compute_velocity(poses_smooth, self.fps, self.vel_smooth_window)
            pose_vel_out = pose_vel[:, self.joint_indices, :]    # (N, 8, 3)
            vr_vel = _compute_velocity(vr_smooth, self.fps, self.vel_smooth_window)
        else:
            pose_vel_out = np.zeros_like(pose_pos_out)
            vr_vel = np.zeros_like(vr_smooth)

        # ---- 8. Timestamp-aligned windowing ----
        cop = cop_all
        samples: List[EnhancedSample] = []
        pose_skip = max(0, self.window_size - 1)
        cop_skip = 0
        max_offset_s = self.max_offset_ms / 1000.0

        for cop_index, t_cop in enumerate(cop_ts):
            insert = int(np.searchsorted(pose_ts, t_cop))
            candidate_indices = []
            if insert < len(pose_ts):
                candidate_indices.append(insert)
            if insert > 0:
                candidate_indices.append(insert - 1)
            if not candidate_indices:
                cop_skip += 1
                continue

            end_idx = min(candidate_indices, key=lambda j: abs(pose_ts[j] - t_cop))
            offset_s = abs(float(pose_ts[end_idx] - t_cop))
            if offset_s > max_offset_s:
                cop_skip += 1
                continue

            start_idx = end_idx - self.window_size + 1
            if start_idx < 0:
                cop_skip += 1
                continue

            samples.append(
                EnhancedSample(
                    sequence_dir=seq_dir,
                    start_frame=start_idx,
                    cop_index=cop_index,
                )
            )

        return (pose_pos_out, pose_vel_out, vr_smooth, vr_vel, joint_groups, com, com_vel,
                joint_rotations, cop, samples, pose_skip, cop_skip)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns dict with enhanced features (all cm):
          pose_pos      : (W, 8, 3)   lower-body joints only, SEQ-START-relative
          pose_vel      : (W, 8, 3)   lower-body velocities
          vr_pos        : (W, 6)      smoothed [Left-HMD, Right-HMD] (HMD-relative)
          vr_vel        : (W, 6)      smoothed velocities
          joint_groups  : (W, 5, 3)   [pelvis, torso, left_leg, right_leg, arms] seq-start-relative
          joint_angles  : (W, 6)      [R_hip, R_knee, R_ankle, L_hip, L_knee, L_ankle] radians
          com           : (W, 3)      CoM displacement from sequence start (captures lean)
          com_vel       : (W, 3)      CoM velocity (cm/s), central-diff then smoothed
          cop           : (2,)        [CoPX, CoPY] target in cm
        """
        s   = self.samples[idx]
        seq = s.sequence_dir
        sl  = slice(s.start_frame, s.start_frame + self.window_size)

        return {
            "pose_pos":     torch.from_numpy(self._poses[seq][sl].copy()),
            "pose_vel":     torch.from_numpy(self._pose_vel[seq][sl].copy()),
            "vr_pos":       torch.from_numpy(self._vr[seq][sl].copy()),
            "vr_vel":       torch.from_numpy(self._vr_vel[seq][sl].copy()),
            "joint_groups": torch.from_numpy(self._joint_groups[seq][sl].copy()),
            "joint_rotations": torch.from_numpy(self._joint_rotations[seq][sl].copy()),
            "com":          torch.from_numpy(self._com[seq][sl].copy()),
            "com_vel":      torch.from_numpy(self._com_vel[seq][sl].copy()),
            # cop = deviation from per-sequence neutral (mean-subtracted, cm)
            "cop":          torch.from_numpy(self._cop[seq][s.cop_index].copy()),
            # cop_mean = per-sequence neutral CoP (cm); add back for absolute prediction
            "cop_mean":     torch.from_numpy(self._cop_mean[seq].copy()),
            "sequence_dir": seq,
        }
