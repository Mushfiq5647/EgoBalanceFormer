"""
BalanceDatasetEstimated15: Balance prediction using 15-joint estimated pose.

This dataset uses the 15-joint pose output from the monocular AutoEncoder
(trained via pose_estimation/scripts/autoencoder/train_egocentric_autoencoder_from_heatmaps.py) instead
of the 19-joint ground_truth.json.

Key differences from BalanceDatasetGT19Enhanced:
  - Pose source: Estimated at runtime from heatmaps using AE checkpoint.
  - CoM: computed from 15 joints using a 15-joint mass model.
  - Joint groups: defined over the 15-joint skeleton.
  - VR data: still loaded from ground_truth.json (if --no_vr is not set).
  - CoP targets: configurable target mode:
      * k4: deviation from initial K-sample baseline mean (K from cop_k)
      * seqmean: deviation from per-sequence mean
      * absolute: direct absolute CoP
  - Optional: Procrustes alignment to GT (if align_to_gt=True).

Units (consistent with gt19_enhanced):
  ── Positions (length): all in CENTIMETERS (cm) ──
  pose_pos    : (W, n_joints, 3)  15 joints, smoothed, pelvis-relative (pelvis at origin)
  vr_pos      : (W, 3/6/9)        configurable VR features, smoothed
  joint_groups: (W, 4, 3)         [torso, left_leg, right_leg, arms] in cm
  com         : (W, 3)            center of mass, pelvis-relative, in cm
  cop         : (2,)              [CoPX, CoPY] target (cm), per selected mode

  ── Velocities: all in CENTIMETERS PER SECOND (cm/s) ──
  pose_vel    : (W, n_joints, 3)  central-diff velocity of pose_pos, then smoothed
  vr_vel      : (W, 3/6/9)        central-diff velocity of vr_pos, then smoothed
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.spatial import procrustes
from torch.utils.data import Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.pose_estimator import AEPoseEstimator  # noqa
from pose_estimation.utils.joint_mapping import map_19_to_preferred_15  # noqa


# 15-joint segment mass fractions (anthropometric model)
# Order: Neck, R_shoulder, R_elbow, R_wrist, L_shoulder, L_elbow, L_wrist,
#        R_hip, R_knee, R_ankle, R_foot, L_hip, L_knee, L_ankle, L_foot
JOINT_SEGMENT_MASS_15 = np.array([
    0.0694,  # 0: Neck (head+neck combined)
    0.0271,  # 1: R_shoulder
    0.0162,  # 2: R_elbow
    0.0061,  # 3: R_wrist
    0.0271,  # 4: L_shoulder
    0.0162,  # 5: L_elbow
    0.0061,  # 6: L_wrist
    0.1416,  # 7: R_hip (thigh)
    0.0433,  # 8: R_knee (shank)
    0.0137,  # 9: R_ankle (foot)
    0.0145,  # 10: R_foot (toe)
    0.1416,  # 11: L_hip (thigh)
    0.0433,  # 12: L_knee (shank)
    0.0137,  # 13: L_ankle (foot)
    0.0145,  # 14: L_foot (toe)
], dtype=np.float32)
JOINT_SEGMENT_MASS_15 /= JOINT_SEGMENT_MASS_15.sum()

# Lower-body joint indices in 15-joint skeleton (for balance prediction)
LOWER_BODY_INDICES_15 = [7, 8, 9, 10, 11, 12, 13, 14]  # Both legs only
LOWER_BODY_NAMES_15 = [
    "Right_hip", "Right_knee", "Right_ankle", "Right_foot",
    "Left_hip", "Left_knee", "Left_ankle", "Left_foot",
]

# Joint groups for 15-joint skeleton
JOINT_GROUPS_15 = {
    "torso":     [0, 1, 4],              # Neck + shoulders
    "left_leg":  [11, 12, 13, 14],       # L_hip, L_knee, L_ankle, L_foot
    "right_leg": [7, 8, 9, 10],          # R_hip, R_knee, R_ankle, R_foot
    "arms":      [2, 3, 5, 6],           # elbows + wrists
}


def _compute_center_of_mass_15(pose: np.ndarray) -> np.ndarray:
    """
    Compute weighted CoM from 15-joint pose.
    Args:
        pose: (N, 15, 3) in cm, pelvis-relative
    Returns:
        com: (N, 3) in cm, pelvis-relative
    """
    return np.einsum("njd,j->nd", pose, JOINT_SEGMENT_MASS_15)


def _aggregate_joint_groups_15(pose: np.ndarray) -> np.ndarray:
    """
    Aggregate 15 joints into 4 stable groups by averaging.
    Args:
        pose: (N, 15, 3)
    Returns:
        groups: (N, 4, 3) [torso, left_leg, right_leg, arms]
    """
    groups = []
    for name in ["torso", "left_leg", "right_leg", "arms"]:
        indices = JOINT_GROUPS_15[name]
        groups.append(pose[:, indices, :].mean(axis=1))  # (N, 3)
    return np.stack(groups, axis=1)  # (N, 4, 3)


def _smooth_temporal(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Temporal smoothing via uniform_filter1d along axis=0.
    Args:
        arr: (N, ...) any shape
        window: int, smoothing window size
    Returns:
        smoothed: same shape as arr
    """
    if window <= 1:
        return arr
    return uniform_filter1d(arr, size=window, axis=0, mode="nearest")


def _butterworth_lowpass(
    arr: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int = 2,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter along time axis (axis=0).
    Args:
        arr: (N, ...) any shape
        fps: sampling rate in Hz
        cutoff_hz: cutoff frequency in Hz
        order: filter order
    Returns:
        filtered array with same shape
    """
    # Lazy import so non-Butterworth runs do not depend on scipy.signal.
    from scipy.signal import butter, filtfilt

    if arr.shape[0] < 4:
        return arr
    nyquist = 0.5 * float(fps)
    if nyquist <= 0:
        return arr
    cutoff = float(cutoff_hz)
    if cutoff <= 0:
        return arr
    wn = min(cutoff / nyquist, 0.999)
    b, a = butter(int(order), wn, btype="low")
    # filtfilt pad requirement; skip filtering if sequence is too short
    padlen = 3 * (max(len(a), len(b)) - 1)
    if arr.shape[0] <= padlen:
        return arr
    flat = arr.reshape(arr.shape[0], -1)
    filt = filtfilt(b, a, flat, axis=0)
    return filt.reshape(arr.shape).astype(arr.dtype, copy=False)


def _compute_velocity(pos: np.ndarray, fps: float, smooth_window: int = 3) -> np.ndarray:
    """
    Central-difference velocity, then smooth.
    Args:
        pos: (N, ...) positions in cm
        fps: frame rate
        smooth_window: temporal smoothing window for velocity
    Returns:
        vel: (N, ...) velocities in cm/s
    """
    N = pos.shape[0]
    if N < 2:
        return np.zeros_like(pos)
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0 * fps
    vel[0] = (pos[1] - pos[0]) * fps
    vel[-1] = (pos[-1] - pos[-2]) * fps
    return _smooth_temporal(vel, smooth_window)


def _extract_lower_body_rotations_from_gt19(gt_rotations_19: np.ndarray) -> np.ndarray:
    """
    Extract 8 lower-body joint rotations (Euler angles) from 19-joint GT rotations.
    
    Mapping from 19-joint to 8 lower-body joints:
    - Right_hip (15-joint idx 7) → R_Femur (19-joint idx 15)
    - Right_knee (15-joint idx 8) → R_Tibia (19-joint idx 16)
    - Right_ankle (15-joint idx 9) → R_Foot (19-joint idx 17)
    - Right_foot (15-joint idx 10) → R_Toe (19-joint idx 18)
    - Left_hip (15-joint idx 11) → L_Femur (19-joint idx 7)
    - Left_knee (15-joint idx 12) → L_Tibia (19-joint idx 8)
    - Left_ankle (15-joint idx 13) → L_Foot (19-joint idx 9)
    - Left_foot (15-joint idx 14) → L_Toe (19-joint idx 10)
    
    Args:
        gt_rotations_19: (N, 19, 3) Euler angles in degrees
    
    Returns:
        rotations_8: (N, 8, 3) Euler angles in radians (converted from degrees)
    """
    gt_rotations_19 = np.asarray(gt_rotations_19, dtype=np.float32)
    if gt_rotations_19.ndim == 2 and gt_rotations_19.shape == (19, 3):
        gt_rotations_19 = gt_rotations_19[None, ...]
    if gt_rotations_19.ndim != 3 or gt_rotations_19.shape[1:] != (19, 3):
        raise ValueError(
            f"Expected GT rotations shape (N,19,3), got {gt_rotations_19.shape}"
        )

    # 19-joint indices for the 8 lower-body joints
    indices_19 = [15, 16, 17, 18, 7, 8, 9, 10]  # R_leg then L_leg

    # Extract and convert to radians
    rotations_8 = gt_rotations_19[:, indices_19, :] * (np.pi / 180.0)  # (N, 8, 3)
    
    return rotations_8


def _compute_pose_derived_rotations_from_pose15(pose_15: np.ndarray) -> np.ndarray:
    """
    Build lower-body kinematic "rotation-like" features from estimated pose.

    Output layout matches joint_rotations expected by the model: (N, 8, 3), where
    the 8 joints are [R_hip, R_knee, R_ankle, R_foot, L_hip, L_knee, L_ankle, L_foot].
    Each 3-vector is represented as:
      [yaw_from_bone_direction, pitch_from_bone_direction, signed_flexion_angle]
    in radians.

    The third channel is a signed kinematic angle (not Euler roll), computed from
    neighboring segments to avoid the previous degenerate constant-zero channel.
    """
    pose_15 = np.asarray(pose_15, dtype=np.float32)
    if pose_15.ndim != 3 or pose_15.shape[1:] != (15, 3):
        raise ValueError(f"Expected pose_15 shape (N,15,3), got {pose_15.shape}")

    # Lower-body joints in 15-joint order
    r_hip, r_knee, r_ankle, r_foot = pose_15[:, 7], pose_15[:, 8], pose_15[:, 9], pose_15[:, 10]
    l_hip, l_knee, l_ankle, l_foot = pose_15[:, 11], pose_15[:, 12], pose_15[:, 13], pose_15[:, 14]

    # Bone directions
    r_thigh = r_knee - r_hip
    r_shank = r_ankle - r_knee
    r_foot_vec = r_foot - r_ankle
    l_thigh = l_knee - l_hip
    l_shank = l_ankle - l_knee
    l_foot_vec = l_foot - l_ankle

    def _safe_unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        return v / n

    def _vec_to_ypr(v: np.ndarray) -> np.ndarray:
        eps = 1e-8
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        yaw = np.arctan2(vy, vx)
        horiz = np.sqrt(vx * vx + vy * vy + eps)
        pitch = np.arctan2(vz, horiz)
        dummy = np.zeros_like(yaw)
        return np.stack([yaw, pitch, dummy], axis=1).astype(np.float32)

    def _signed_angle(v1: np.ndarray, v2: np.ndarray, ref_axis: np.ndarray) -> np.ndarray:
        """
        Signed angle from v1 to v2 around ref_axis.
        """
        u1 = _safe_unit(v1)
        u2 = _safe_unit(v2)
        cross = np.cross(u1, u2)  # (N,3)
        dot = np.clip(np.sum(u1 * u2, axis=1), -1.0, 1.0)
        ang = np.arccos(dot)
        sign = np.sign(np.sum(cross * ref_axis.reshape(1, 3), axis=1) + 1e-8)
        return (ang * sign).astype(np.float32)

    # Signed flexion proxy angles (sagittal-style sign with x-axis reference)
    # If your coordinate frame differs, this sign may flip globally but remains informative.
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    down = -z_axis.reshape(1, 3)

    r_hip_flex = _signed_angle(np.repeat(down, len(pose_15), axis=0), r_thigh, x_axis)
    r_knee_flex = _signed_angle(r_thigh, r_shank, x_axis)
    r_ankle_flex = _signed_angle(r_shank, r_foot_vec, x_axis)

    l_hip_flex = _signed_angle(np.repeat(down, len(pose_15), axis=0), l_thigh, x_axis)
    l_knee_flex = _signed_angle(l_thigh, l_shank, x_axis)
    l_ankle_flex = _signed_angle(l_shank, l_foot_vec, x_axis)

    # Match expected 8-joint order
    r_hip_ypr = _vec_to_ypr(r_thigh)
    r_knee_ypr = _vec_to_ypr(r_shank)
    r_ankle_ypr = _vec_to_ypr(r_foot_vec)
    r_foot_ypr = _vec_to_ypr(r_foot_vec)  # reuse distal segment
    l_hip_ypr = _vec_to_ypr(l_thigh)
    l_knee_ypr = _vec_to_ypr(l_shank)
    l_ankle_ypr = _vec_to_ypr(l_foot_vec)
    l_foot_ypr = _vec_to_ypr(l_foot_vec)  # reuse distal segment

    # Replace 3rd component with signed flexion proxies
    r_hip_ypr[:, 2] = r_hip_flex
    r_knee_ypr[:, 2] = r_knee_flex
    r_ankle_ypr[:, 2] = r_ankle_flex
    r_foot_ypr[:, 2] = r_ankle_flex
    l_hip_ypr[:, 2] = l_hip_flex
    l_knee_ypr[:, 2] = l_knee_flex
    l_ankle_ypr[:, 2] = l_ankle_flex
    l_foot_ypr[:, 2] = l_ankle_flex

    ypr = [
        r_hip_ypr, r_knee_ypr, r_ankle_ypr, r_foot_ypr,
        l_hip_ypr, l_knee_ypr, l_ankle_ypr, l_foot_ypr,
    ]
    return np.stack(ypr, axis=1)  # (N, 8, 3)


def _compute_joint_angles_lower_body(pose_15: np.ndarray) -> np.ndarray:
    """
    Compute 6 lower-body joint angles from 15-joint pose.
    
    NOTE: This function is now deprecated in favor of using GT rotations directly.
    It is kept for backward compatibility.
    
    Args:
        pose_15: (N, 15, 3) full 15-joint pose, pelvis-relative
    
    Returns:
        angles: (N, 6) [R_hip, R_knee, R_ankle, L_hip, L_knee, L_ankle] in radians
    """
    N = pose_15.shape[0]
    angles = np.zeros((N, 6), dtype=np.float32)
    
    # Extract lower-body joints (indices in 15-joint format)
    # 7: R_hip, 8: R_knee, 9: R_ankle, 10: R_foot
    # 11: L_hip, 12: L_knee, 13: L_ankle, 14: L_foot
    
    # Right leg
    r_hip = pose_15[:, 7, :]    # (N, 3)
    r_knee = pose_15[:, 8, :]
    r_ankle = pose_15[:, 9, :]
    r_foot = pose_15[:, 10, :]
    
    # Left leg
    l_hip = pose_15[:, 11, :]
    l_knee = pose_15[:, 12, :]
    l_ankle = pose_15[:, 13, :]
    l_foot = pose_15[:, 14, :]
    
    # Vertical reference (pelvis is at origin in pelvis-relative pose)
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


def _procrustes_align_sequence(pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    """
    Align predicted pose to GT using Procrustes (per-frame similarity transform).
    Args:
        pred_pose: (N, 15, 3) predicted pose, pelvis-relative
        gt_pose: (N, 15, 3) GT pose, pelvis-relative
    Returns:
        aligned_pose: (N, 15, 3) Procrustes-aligned to GT
    """
    N = pred_pose.shape[0]
    aligned = np.zeros_like(pred_pose)
    for i in range(N):
        # scipy.spatial.procrustes returns (mtx1, mtx2, disparity)
        # mtx2 is the transformed version of the second matrix (pred) to align with first (gt)
        _, aligned[i], _ = procrustes(gt_pose[i], pred_pose[i])
    return aligned


@dataclass(frozen=True)
class EstimatedSample:
    sequence_dir: str
    start_frame: int
    cop_index: int
    target_cop_index: int


class BalanceDatasetEstimated15(Dataset):
    """
    Balance dataset using 15-joint estimated pose from monocular AutoEncoder.

    Expects each sequence_dir to contain:
      - estimated_poses.json (15 joints, pelvis-relative, in preferred 15 order)
      - balance.json (CoP samples)
      - ground_truth.json (for VR data and timestamps)

    Features produced:
      - pose_pos, pose_vel: from estimated 15 joints
      - joint_groups: 4 groups (torso, left_leg, right_leg, arms)
      - com: pelvis-relative CoM from 15 joints
      - vr_pos, vr_vel: optional (from ground_truth.json)
      - cop: deviation from first-K baseline mean (K=cop_k)
      - cop_mean: first-K baseline CoP mean
    """

    def __init__(
        self,
        list_file: str,
        window_size: int = 11,
        fps: float = 22.0,
        pose_smooth_window: int = 5,
        vel_smooth_window: int = 3,
        max_offset_ms: float = 50.0,
        use_velocity: bool = True,
        use_vr: bool = True,
        include_hmd: bool = False,
        vr_mode: Optional[str] = None,
        # Runtime pose estimation
        ae_checkpoint: Optional[str] = None,
        heatmap_dir: str = "custom_pred_heatmaps",
        use_body_part: bool = True,
        ae_hidden_size: int = 20,
        ae_batch_size: int = 128,
        # Procrustes alignment
        align_to_gt: bool = False,
        mad_threshold: Optional[float] = None,
        cop_target_mode: str = "k4",
        cop_k: int = 4,
        forecast_horizon: int = 0,
        butterworth: bool = False,
        bw_cutoff_hz: float = 3.0,
        bw_order: int = 2,
        rotation_source: str = "gt",
        **kwargs,
    ):
        self.window_size = window_size
        self.fps = fps
        self.pose_smooth_window = pose_smooth_window
        self.vel_smooth_window = vel_smooth_window
        self.max_offset_ms = max_offset_ms
        self.use_velocity = use_velocity
        self.use_vr = use_vr
        # VR feature mode:
        # - rel_lr: [left-hmd, right-hmd] (6D)
        # - hmd_only: [hmd] (3D)
        # - hmd_lr: [hmd, left-hmd, right-hmd] (9D)
        # keep backward compatibility with include_hmd flag
        if vr_mode is None:
            self.vr_mode = "hmd_lr" if bool(include_hmd) else "rel_lr"
        else:
            self.vr_mode = str(vr_mode).lower()
        if self.vr_mode not in {"rel_lr", "hmd_only", "hmd_lr"}:
            raise ValueError(f"Unsupported vr_mode: {vr_mode}")
        self.align_to_gt = align_to_gt
        self.heatmap_dir = heatmap_dir
        self.ae_batch_size = ae_batch_size
        self.mad_threshold = mad_threshold
        self.cop_target_mode = str(cop_target_mode).lower()
        if self.cop_target_mode not in {"k4", "seqmean", "absolute"}:
            raise ValueError(f"Unsupported cop_target_mode: {cop_target_mode}")
        self.cop_k = int(cop_k)
        if self.cop_k < 1:
            raise ValueError(f"cop_k must be >= 1, got {cop_k}")
        self.forecast_horizon = int(forecast_horizon)
        if self.forecast_horizon < 0:
            raise ValueError(f"forecast_horizon must be >= 0, got {forecast_horizon}")
        self.butterworth = bool(butterworth)
        self.bw_cutoff_hz = float(bw_cutoff_hz)
        self.bw_order = int(bw_order)
        self.rotation_source = str(rotation_source).lower()
        if self.rotation_source not in {"gt", "pose"}:
            raise ValueError(f"Unsupported rotation_source: {rotation_source}")

        # Initialize pose estimator if checkpoint provided
        self.pose_estimator: Optional[AEPoseEstimator] = None
        if ae_checkpoint is not None:
            print(f"[Estimated15] Loading AE checkpoint: {ae_checkpoint}")
            self.pose_estimator = AEPoseEstimator(
                checkpoint=ae_checkpoint,
                use_body_part=use_body_part,
                ae_hidden_size=ae_hidden_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"[Estimated15] Pose estimator ready (device={self.pose_estimator.device})")

        with open(list_file) as f:
            self.sequence_dirs = [
                l.strip() for l in f if l.strip() and not l.startswith("#")
            ]

        self._poses:        Dict[str, np.ndarray] = {}  # (N, 8, 3) lower-body only
        self._pose_vel:     Dict[str, np.ndarray] = {}  # (N, 8, 3) lower-body vel
        self._vr:           Dict[str, np.ndarray] = {}  # (N,6/9) smoothed
        self._vr_vel:       Dict[str, np.ndarray] = {}  # (N,6/9) smoothed vel
        self._joint_groups: Dict[str, np.ndarray] = {}  # (N,4,3)
        self._joint_rotations: Dict[str, np.ndarray] = {}  # (N,8,3) lower-body rotations (Euler, radians)
        self._joint_rot_vel: Dict[str, np.ndarray] = {}  # (N,8,3) lower-body rotation velocity (rad/s)
        self._com:          Dict[str, np.ndarray] = {}  # (N,3)
        self._com_vel:      Dict[str, np.ndarray] = {}  # (N,3) CoM velocity cm/s
        self._cop:          Dict[str, np.ndarray] = {}  # (M,2) first-K-baseline-subtracted deviations
        self._cop_mean:     Dict[str, np.ndarray] = {}  # (2,) first-K baseline CoP mean (cm)

        self.samples: List[EstimatedSample] = []
        skipped_seqs = 0
        total_pose_skipped = 0
        total_cop_skipped = 0

        for seq_dir in self.sequence_dirs:
            try:
                result = self._load_and_align(seq_dir)
                (poses, pose_vel, vr, vr_vel, groups, com, com_vel, rotations, rot_vel, cop,
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
                self._joint_rot_vel[seq_dir] = rot_vel
                self._com[seq_dir] = com
                self._com_vel[seq_dir] = com_vel
                
                # CoP target mode:
                # - k4: deviation from first-K samples mean
                # - seqmean: deviation from per-sequence mean
                # - absolute: direct CoP target
                if self.cop_target_mode == "k4":
                    k = min(self.cop_k, len(cop))
                    cop_mean = cop[:k].mean(axis=0)
                    cop_target = cop - cop_mean
                elif self.cop_target_mode == "seqmean":
                    cop_mean = cop.mean(axis=0)
                    cop_target = cop - cop_mean
                else:  # absolute
                    cop_mean = np.zeros((2,), dtype=np.float32)
                    cop_target = cop
                self._cop_mean[seq_dir] = cop_mean
                self._cop[seq_dir] = cop_target
                self.samples.extend(samples)
                total_pose_skipped += pose_skip
                total_cop_skipped += cop_skip
            except (FileNotFoundError, ValueError) as e:
                print(f"  [skip] {os.path.basename(seq_dir)}: {e}")
                skipped_seqs += 1

        if not self.samples:
            raise ValueError(f"No aligned windows from '{list_file}'")

        n_seqs = len(self.sequence_dirs) - skipped_seqs
        mad_str = f", MAD={self.mad_threshold}" if self.mad_threshold else ""
        bw_str = f", BW={self.bw_order}@{self.bw_cutoff_hz:.1f}Hz" if self.butterworth else ""
        hmd_str = f", VR={self.vr_mode}"
        print(
            f"[Estimated15] {len(self.samples)} windows from {n_seqs}/{len(self.sequence_dirs)} sequences "
            f"(joints=15 estimated, pos+vel, skipped {skipped_seqs} seqs, "
            f"pose_smooth={self.pose_smooth_window}, vel_smooth={self.vel_smooth_window}, "
            f"max_offset_ms={self.max_offset_ms:.1f}{mad_str}{bw_str}{hmd_str}, "
            f"rot_source={self.rotation_source}, "
            f"cop_target={self.cop_target_mode}, cop_k={self.cop_k}, "
            f"horizon={self.forecast_horizon})"
        )

    def _load_and_align(self, seq_dir: str):
        # ---- Load ground_truth.json for timestamps, GT pose (if align_to_gt), and VR ----
        gt_path = os.path.join(seq_dir, "ground_truth.json")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground_truth.json")
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        if not isinstance(gt_data, list):
            raise ValueError(f"Expected list in {gt_path}")

        # ---- Estimate pose at runtime (15 joints, pelvis-relative, cm) ----
        if self.pose_estimator is None:
            raise ValueError("ae_checkpoint must be provided for runtime pose estimation")

        # Find heatmap files for this sequence
        # Try two locations: 1) inside sequence dir, 2) in central heatmap_dir
        heatmap_pattern_1 = os.path.join(seq_dir, self.heatmap_dir, "*_hm15x64x64.npy")
        heatmap_files = sorted(glob.glob(heatmap_pattern_1))
        
        if not heatmap_files:
            # Try central location: heatmap_dir/<seq_basename>/*_hm15x64x64.npy
            seq_basename = os.path.basename(seq_dir.rstrip("/"))
            heatmap_pattern_2 = os.path.join(self.heatmap_dir, f"{seq_basename}/*_hm15x64x64.npy")
            heatmap_files = sorted(glob.glob(heatmap_pattern_2))
        
        if not heatmap_files:
            raise FileNotFoundError(f"No heatmaps found in: {heatmap_pattern_1} or {heatmap_pattern_2}")

        # Load heatmaps
        heatmaps15 = np.stack([np.load(f) for f in heatmap_files], axis=0)  # (N,15,64,64)

        # Load body parts if needed
        body_parts4 = None
        if self.pose_estimator.use_body_part:
            # Try same directory as heatmaps
            heatmap_dir_actual = os.path.dirname(heatmap_files[0])
            bp_pattern = os.path.join(heatmap_dir_actual, "*_bp4x64x64.npy")
            bp_files = sorted(glob.glob(bp_pattern))
            if len(bp_files) == len(heatmap_files):
                body_parts4 = np.stack([np.load(f) for f in bp_files], axis=0)  # (N,4,64,64)

        # Estimate pose
        poses_raw = self.pose_estimator.estimate(heatmaps15, body_parts4, batch_size=self.ae_batch_size)  # (N,15,3) cm, pelvis-relative

        # Match heatmap files to GT by image_name
        pose_frames = []
        gt_by_name = {item.get("image_name"): item for item in gt_data if "image_name" in item}
        for i, hm_file in enumerate(heatmap_files):
            image_stem = os.path.basename(hm_file).replace("_hm15x64x64.npy", "")
            if image_stem not in gt_by_name:
                continue  # Skip if no GT match
            pose_frames.append({
                "image_name": image_stem,
                "translation": poses_raw[i],  # (15,3)
                "gt_item": gt_by_name[image_stem],
            })

        if len(pose_frames) < self.window_size:
            raise ValueError(f"Not enough aligned pose frames: {len(pose_frames)} < {self.window_size}")

        poses_raw = np.stack([f["translation"] for f in pose_frames], axis=0)  # (N,15,3)

        # Extract timestamps
        pose_ts_list = []
        for pf in pose_frames:
            gt_item = pf["gt_item"]
            ts_str = gt_item.get("timestamp", "")
            if not ts_str:
                raise ValueError(f"Missing timestamp for {pf['image_name']}")
            pose_ts_list.append(self._parse_timestamp(ts_str))
        pose_ts = np.array(pose_ts_list, dtype=np.float64)

        # ---- Load GT rotations only if requested ----
        gt_rotations_19 = None
        if self.rotation_source == "gt":
            gt_rotations_19_list = []
            invalid_rot_count = 0
            for pf in pose_frames:
                gt_item = pf["gt_item"]
                rot_raw = gt_item.get("joints", {}).get("rotation", None)
                rot_19 = np.asarray(rot_raw, dtype=np.float32) if rot_raw is not None else np.empty((0,), dtype=np.float32)
                if rot_19.shape != (19, 3):
                    # Keep frame alignment stable; use neutral rotation fallback.
                    rot_19 = np.zeros((19, 3), dtype=np.float32)
                    invalid_rot_count += 1
                gt_rotations_19_list.append(rot_19)
            gt_rotations_19 = np.stack(gt_rotations_19_list, axis=0)  # (N, 19, 3)
            if invalid_rot_count > 0:
                raise ValueError(
                    f"Invalid GT rotation shape in {invalid_rot_count}/{len(gt_rotations_19_list)} frames"
                )

        # ---- Optional: Procrustes alignment to GT ----
        if self.align_to_gt:
            # Load GT 19-joint pose and map to 15 joints
            gt_pose15_list = []
            for pf in pose_frames:
                gt_item = pf["gt_item"]
                joints_19 = np.array(gt_item["joints"]["translation"], dtype=np.float32) / 10.0  # mm→cm
                gt_pose15 = map_19_to_preferred_15(joints_19)  # (15,3) pelvis-relative
                gt_pose15_list.append(gt_pose15)
            gt_pose15 = np.stack(gt_pose15_list, axis=0)  # (N,15,3)

            # Align predicted pose to GT using Procrustes
            poses_raw = _procrustes_align_sequence(poses_raw, gt_pose15)

        # ---- Load VR data (optional) ----
        if self.use_vr:
            vr_raw = []
            for pf in pose_frames:
                gt_item = pf["gt_item"]
                hmd = np.array(gt_item.get("vrHMD", [0, 0, 0]), dtype=np.float32) * 100.0  # m→cm
                left = np.array(gt_item.get("vrLeft", [0, 0, 0]), dtype=np.float32) * 100.0
                right = np.array(gt_item.get("vrRight", [0, 0, 0]), dtype=np.float32) * 100.0
                left_rel = left - hmd
                right_rel = right - hmd
                if self.vr_mode == "hmd_only":
                    vr_raw.append(hmd)
                elif self.vr_mode == "hmd_lr":
                    vr_raw.append(np.concatenate([hmd, left_rel, right_rel]))
                else:
                    vr_raw.append(np.concatenate([left_rel, right_rel]))
            vr_raw = np.stack(vr_raw, axis=0)  # (N,3/6/9)
        else:
            if self.vr_mode == "hmd_only":
                vr_dims = 3
            elif self.vr_mode == "hmd_lr":
                vr_dims = 9
            else:
                vr_dims = 6
            vr_raw = np.zeros((len(pose_frames), vr_dims), dtype=np.float32)

        # ---- Load CoP ----
        bal_path = os.path.join(seq_dir, "balance.json")
        if not os.path.exists(bal_path):
            raise FileNotFoundError(f"Missing balance.json")
        with open(bal_path, "r") as f:
            bal_data = json.load(f)
        if not isinstance(bal_data, list):
            raise ValueError(f"Expected list in {bal_path}")

        cop_all = []
        cop_ts_list = []
        for item in bal_data:
            ts_str = item.get("timestamp", "")
            if not ts_str:
                continue
            cop_all.append([item.get("copX", 0.0), item.get("copY", 0.0)])
            cop_ts_list.append(self._parse_timestamp(ts_str))

        cop_all = np.array(cop_all, dtype=np.float32)
        cop_ts = np.array(cop_ts_list, dtype=np.float64)

        if len(cop_all) == 0:
            raise ValueError("No valid CoP samples")
        
        # ---- MAD outlier filtering (BEFORE alignment) ----
        if self.mad_threshold is not None and self.mad_threshold > 0:
            inlier_mask = _filter_outliers_mad(cop_all, self.mad_threshold)
            n_outliers = (~inlier_mask).sum()
            if n_outliers > 0:
                cop_all = cop_all[inlier_mask]
                cop_ts = cop_ts[inlier_mask]
                print(f"    [MAD] {os.path.basename(seq_dir)}: removed {n_outliers}/{len(inlier_mask)} CoP outliers (before alignment)")

        # ---- Temporal smoothing ----
        if self.butterworth:
            poses_smooth = _butterworth_lowpass(
                poses_raw, self.fps, self.bw_cutoff_hz, self.bw_order
            )  # (N,15,3)
            vr_smooth = _butterworth_lowpass(
                vr_raw, self.fps, self.bw_cutoff_hz, self.bw_order
            )  # (N,6)
        else:
            poses_smooth = _smooth_temporal(poses_raw, self.pose_smooth_window)  # (N,15,3)
            vr_smooth = _smooth_temporal(vr_raw, self.pose_smooth_window)  # (N,6)

        # ---- Compute CoM from pelvis-relative pose ----
        # Since estimated pose is already pelvis-relative, CoM is also pelvis-relative.
        # For balance prediction (deviation targets), this is acceptable.
        com = _compute_center_of_mass_15(poses_smooth)  # (N,3)
        com_vel = _compute_velocity(com, self.fps, self.vel_smooth_window)  # (N,3) cm/s

        # ---- Joint grouping ----
        joint_groups = _aggregate_joint_groups_15(poses_smooth)  # (N,4,3)

        # ---- Rotation features: GT Euler or pose-derived kinematics ----
        if self.rotation_source == "gt":
            joint_rotations = _extract_lower_body_rotations_from_gt19(gt_rotations_19)  # (N, 8, 3) radians
        else:
            joint_rotations = _compute_pose_derived_rotations_from_pose15(poses_smooth)  # (N, 8, 3) radians
        if self.butterworth:
            joint_rotations = _butterworth_lowpass(
                joint_rotations, self.fps, self.bw_cutoff_hz, self.bw_order
            )

        # ---- Rotation velocity (rad/s), optionally smoothed ----
        if self.use_velocity:
            joint_rot_vel = _compute_velocity(joint_rotations, self.fps, self.vel_smooth_window)
        else:
            joint_rot_vel = np.zeros_like(joint_rotations)

        # ---- Extract only lower-body joints (8 joints: indices 7-14) ----
        pose_pos_out = poses_smooth[:, LOWER_BODY_INDICES_15, :]  # (N, 8, 3)

        # ---- Compute velocities (or zeros if disabled) ----
        if self.use_velocity:
            pose_vel = _compute_velocity(poses_smooth, self.fps, self.vel_smooth_window)
            pose_vel_out = pose_vel[:, LOWER_BODY_INDICES_15, :]  # (N, 8, 3)
            vr_vel = _compute_velocity(vr_smooth, self.fps, self.vel_smooth_window)  # (N, 3/6/9)
        else:
            pose_vel_out = np.zeros_like(pose_pos_out)
            vr_vel = np.zeros_like(vr_smooth)

        # ---- Timestamp-aligned windowing ----
        cop = cop_all
        samples: List[EstimatedSample] = []
        pose_skip = max(0, self.window_size - 1)
        cop_skip = 0
        max_offset_s = self.max_offset_ms / 1000.0

        for cop_index, t_cop in enumerate(cop_ts):
            target_cop_index = cop_index + self.forecast_horizon
            if target_cop_index >= len(cop):
                cop_skip += 1
                continue
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
                EstimatedSample(
                    sequence_dir=seq_dir,
                    start_frame=start_idx,
                    cop_index=cop_index,
                    target_cop_index=target_cop_index,
                )
            )

        return (pose_pos_out, pose_vel_out, vr_smooth, vr_vel, joint_groups, com, com_vel,
                joint_rotations, joint_rot_vel, cop, samples, pose_skip, cop_skip)

    def _parse_timestamp(self, ts_str: str) -> float:
        """Parse timestamp string to seconds (simple heuristic)."""
        parts = ts_str.replace("-", " ").replace(":", " ").split()
        if len(parts) >= 6:
            h, m, s = int(parts[3]), int(parts[4]), int(parts[5])
            ms = int(parts[6]) if len(parts) > 6 else 0
            return h * 3600.0 + m * 60.0 + s + ms / 1000.0
        return 0.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns dict with features (all cm):
          pose_pos      : (W, 8, 3)   lower-body joints only, pelvis-relative
          pose_vel      : (W, 8, 3)   lower-body velocities
          vr_pos        : (W, 3/6/9)  smoothed VR features (mode-dependent)
          vr_vel        : (W, 3/6/9)  smoothed velocities
          joint_groups  : (W, 4, 3)   [torso, left_leg, right_leg, arms]
          joint_angles  : (W, 6)      [R_hip, R_knee, R_ankle, L_hip, L_knee, L_ankle] radians
          com           : (W, 3)      CoM, pelvis-relative
          com_vel       : (W, 3)      CoM velocity (cm/s), central-diff then smoothed
          cop           : (2,)        [CoPX, CoPY] target in cm (mode-dependent)
          cop_mean      : (2,)        baseline mean (0 for absolute mode)
        """
        s = self.samples[idx]
        seq = s.sequence_dir
        sl = slice(s.start_frame, s.start_frame + self.window_size)

        return {
            "pose_pos":     torch.from_numpy(self._poses[seq][sl].copy()),
            "pose_vel":     torch.from_numpy(self._pose_vel[seq][sl].copy()),
            "vr_pos":       torch.from_numpy(self._vr[seq][sl].copy()),
            "vr_vel":       torch.from_numpy(self._vr_vel[seq][sl].copy()),
            "joint_groups": torch.from_numpy(self._joint_groups[seq][sl].copy()),
            "joint_rotations": torch.from_numpy(self._joint_rotations[seq][sl].copy()),
            "joint_rot_vel": torch.from_numpy(self._joint_rot_vel[seq][sl].copy()),
            "com":          torch.from_numpy(self._com[seq][sl].copy()),
            "com_vel":      torch.from_numpy(self._com_vel[seq][sl].copy()),
            "cop":          torch.from_numpy(self._cop[seq][s.target_cop_index].copy()),
            "cop_mean":     torch.from_numpy(self._cop_mean[seq].copy()),
            "sequence_dir": seq,
        }
