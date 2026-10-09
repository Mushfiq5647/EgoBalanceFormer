"""
BalanceDataset — loads windowed (pose + VR) → CoP samples.

Data assumptions per sequence directory:
  pose source = "json":
    estimated_poses.json (or ground_truth.json fallback) — pose at 22 fps
  pose source = "ae_tensor":
    ground_truth.json for frame indexing + precomputed heatmaps under
    <sequence_dir>/<heatmap_subdir>/<image_name>_hm15x64x64.npy
  balance.json       — CoP GT at 2 fps, format:
                         [ {"CoPX": <float>, "CoPY": <float>}, ... ]
  ground_truth.json  — VR tracking at 22 fps, per-frame keys:
                         "vrHMD": [x, y, z],
                         "vrLeft": [x, y, z],
                         "vrRight":[x, y, z]
                       (legacy fallback to vr_data.json is still supported)

Temporal alignment:
  Both pose and VR are at 22 fps; CoP is at 2 fps.
  22 / 2 = 11  →  window of 11 pose/VR frames predicts one CoP sample.
  Window i  ↔  pose frames [i*11 .. (i+1)*11)  ↔  balance.json[i]

Lower-body joints used (indices in the preferred 15-joint convention):
  7: Right_hip   8: Right_knee   9: Right_ankle  10: Right_foot
  11: Left_hip  12: Left_knee   13: Left_ankle   14: Left_foot
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── make repo root importable ──────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pose_estimation.utils.joint_mapping import map_19_to_preferred_15  # noqa: E402
from data.velocity import VelocityComputer  # noqa: E402
from data.pose_estimator import AEPoseEstimator  # noqa: E402

# Indices of lower-body joints in the preferred-15 convention
LOWER_BODY_INDICES: List[int] = [7, 8, 9, 10, 11, 12, 13, 14]
LOWER_BODY_NAMES: List[str] = [
    "Right_hip", "Right_knee", "Right_ankle", "Right_foot",
    "Left_hip",  "Left_knee",  "Left_ankle",  "Left_foot",
]
N_LOWER_BODY = len(LOWER_BODY_INDICES)   # 8
N_VR_DEVICES = 3                          # HMD, left, right
N_VR_COORDS  = N_VR_DEVICES * 3          # 9
VR_M_TO_CM   = 100.0    # VR data is in meters → convert to cm
COP_CM_TO_M  = 0.01     # kept for reference but NO LONGER APPLIED — CoP stays in cm
POSE_ROOT_INDEX_15 = 0  # Neck in preferred 15-joint order


@dataclass(frozen=True)
class BalanceSample:
    sequence_dir: str
    start_frame:  int   # index into pre-loaded sequence arrays
    cop_index:    int   # index into this sequence's CoP array


def _read_list_file(path: str) -> List[str]:
    with open(path) as f:
        lines = [l.strip() for l in f]
    return [l for l in lines if l and not l.startswith("#")]


def _frame_number(image_name: str) -> int:
    """Extract trailing integer from e.g. 'color_frame_120' → 120."""
    parts = image_name.split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return -1


class BalanceDataset(Dataset):
    """
    Args:
        list_file:      path to a .txt file with one sequence_dir per line
        window_size:    frames per window (default 11 = 22 fps / 2 fps)
        fps:            pose / VR frame rate (default 22)
        smooth_window:  moving-average window for velocity smoothing (1 = off)
        joint_indices:  which joints from the 15-joint set to keep
                        (default: lower body, indices 7-14)
        pose_filename:  preferred pose JSON file inside each sequence dir
        pose_source:    "json" or "ae_tensor"
        vr_filename:    unused for current ground_truth.json VR loading path
    """

    def __init__(
        self,
        list_file: str,
        window_size:   int = 11,
        fps:           float = 22.0,
        smooth_window: int = 3,
        joint_indices: Optional[List[int]] = None,
        pose_filename: str = "estimated_poses.json",
        pose_source: str = "json",
        heatmap_subdir: str = "custom_pred_heatmaps",
        ae_checkpoint: Optional[str] = None,
        ae_device: str = "cpu",
        ae_hidden_size: int = 20,
        ae_batch_size: int = 128,
        use_body_part: bool = False,
        vr_filename:   str = "vr_data.json",
    ):
        self.window_size   = window_size
        self.fps           = fps
        self.smooth_window = smooth_window
        self.joint_indices = joint_indices if joint_indices is not None else LOWER_BODY_INDICES
        self.pose_filename = pose_filename
        self.pose_source   = pose_source
        self.heatmap_subdir = heatmap_subdir
        self.ae_checkpoint = ae_checkpoint
        self.ae_device = ae_device
        self.ae_hidden_size = ae_hidden_size
        self.ae_batch_size = ae_batch_size
        self.use_body_part = use_body_part
        self.vr_filename   = vr_filename
        self.vel_computer  = VelocityComputer(fps=fps)

        if self.pose_source not in {"json", "ae_tensor"}:
            raise ValueError(f"pose_source must be 'json' or 'ae_tensor', got '{self.pose_source}'")

        self.pose_estimator = None
        if self.pose_source == "ae_tensor":
            if not self.ae_checkpoint:
                raise ValueError("pose_source='ae_tensor' requires ae_checkpoint path.")
            self.pose_estimator = AEPoseEstimator(
                checkpoint=self.ae_checkpoint,
                device=self.ae_device,
                ae_hidden_size=self.ae_hidden_size,
                use_body_part=self.use_body_part,
            )

        self.sequence_dirs = _read_list_file(list_file)

        # Per-sequence pre-loaded data (in memory)
        # key → seq_dir string
        self._poses: Dict[str, np.ndarray] = {}   # (N, J_or_15, 3)
        self._vr:    Dict[str, np.ndarray] = {}   # (N, 9)
        self._cop:   Dict[str, np.ndarray] = {}   # (M, 2)

        self.samples: List[BalanceSample] = []
        self._load_all_sequences()
        # Free heavy model after preloading sequence pose tensors.
        self.pose_estimator = None

        if len(self.samples) == 0:
            raise ValueError(f"No valid windows found from list_file='{list_file}'")
        print(f"[BalanceDataset] {len(self.samples)} windows from {len(self.sequence_dirs)} sequences.")

    # ──────────────────────────────────────────────────────────────────────────
    # Loading helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_all_sequences(self) -> None:
        for seq_dir in self.sequence_dirs:
            try:
                if self.pose_source == "ae_tensor":
                    poses = self._load_poses_from_ae(seq_dir)   # (N, J, 3)
                else:
                    poses = self._load_poses(seq_dir)           # (N, J, 3)
                vr    = self._load_vr(seq_dir)        # (N, 9)
                cop   = self._load_cop(seq_dir)       # (M, 2)
            except (FileNotFoundError, ValueError) as e:
                print(f"[warn] Skipping '{seq_dir}': {e}")
                continue

            # Align lengths: N pose frames must be divisible into windows
            n_windows = min(len(poses) // self.window_size,
                            len(vr)    // self.window_size,
                            len(cop))
            if n_windows == 0:
                print(f"[warn] Skipping '{seq_dir}': insufficient data "
                      f"(poses={len(poses)}, vr={len(vr)}, cop={len(cop)})")
                continue

            # Trim to aligned length
            n_frames = n_windows * self.window_size
            self._poses[seq_dir] = poses[:n_frames]
            self._vr[seq_dir]    = vr[:n_frames]
            self._cop[seq_dir]   = cop[:n_windows]

            for i in range(n_windows):
                self.samples.append(BalanceSample(
                    sequence_dir=seq_dir,
                    start_frame=i * self.window_size,
                    cop_index=i,
                ))

    def _resolve_pose_path(self, seq_dir: str) -> str:
        candidates = []
        for name in [self.pose_filename, "estimated_poses.json", "ground_truth.json"]:
            if name and name not in candidates:
                candidates.append(name)
        for name in candidates:
            p = os.path.join(seq_dir, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"Missing pose file in '{seq_dir}'. Tried: {', '.join(candidates)}"
        )

    @staticmethod
    def _extract_xyz_dict(entry: dict, base_key: str) -> Optional[np.ndarray]:
        x_key = f"{base_key}_X"
        y_key = f"{base_key}_Y"
        z_key = f"{base_key}_Z"
        if x_key in entry and y_key in entry and z_key in entry:
            return np.asarray([entry[x_key], entry[y_key], entry[z_key]], dtype=np.float32)
        return None

    @staticmethod
    def _extract_device_vec(entry: dict, aliases: List[str]) -> Optional[np.ndarray]:
        for alias in aliases:
            if alias in entry:
                arr = np.asarray(entry[alias], dtype=np.float32)
                if arr.shape == (3,):
                    return arr
            xyz = BalanceDataset._extract_xyz_dict(entry, alias)
            if xyz is not None:
                return xyz
        return None

    @staticmethod
    def _pick_numeric(entry: dict, keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in entry and entry[key] is not None:
                return float(entry[key])
        return float(default)

    @staticmethod
    def _root_relative_15(poses15: np.ndarray) -> np.ndarray:
        """
        Make preferred-15 poses root-relative using one root joint:
        Neck (idx 0 in preferred 15-joint order).
        Args:
            poses15: (N, 15, 3)
        Returns:
            (N, 15, 3) root-relative
        """
        if poses15.ndim != 3 or poses15.shape[1:] != (15, 3):
            raise ValueError(f"Expected pose shape (N,15,3), got {poses15.shape}")
        root = poses15[:, POSE_ROOT_INDEX_15, :]  # (N,3)
        return poses15 - root[:, None, :]

    def _load_poses(self, seq_dir: str) -> np.ndarray:
        """
        Load pose JSON (estimated_poses.json preferred, ground_truth.json fallback)
        map 19 joints to preferred 15 when needed →
        subset to self.joint_indices → return (N, J, 3) float32.
        """
        pose_path = self._resolve_pose_path(seq_dir)

        with open(pose_path) as f:
            raw = json.load(f)

        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, dict):
            entries = raw.get("frames") or raw.get("data") or raw.get("poses")
            if entries is None and all(isinstance(v, dict) for v in raw.values()):
                entries = list(raw.values())
            if entries is None:
                raise ValueError(f"Unsupported pose format in '{pose_path}'")
        else:
            raise ValueError(f"Unsupported pose JSON type in '{pose_path}': {type(raw)}")

        valid = []
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                continue
            name = item.get("image_name") or item.get("frame_name") or item.get("frame")
            frame_id = _frame_number(str(name)) if name is not None else idx

            joints = item.get("joints")
            translation = None
            if isinstance(joints, dict):
                translation = joints.get("translation")
            elif isinstance(joints, list):
                translation = joints
            if translation is None:
                translation = item.get("translation") or item.get("positions")
            if translation is None:
                continue

            arr = np.asarray(translation, dtype=np.float32)
            if arr.shape == (19, 3):
                arr = map_19_to_preferred_15(arr)
            elif arr.shape != (15, 3):
                continue
            valid.append((frame_id, arr))

        if not valid:
            raise ValueError(f"No valid pose frames in '{pose_path}'")

        valid.sort(key=lambda x: x[0])
        poses15 = np.stack([v[1] for v in valid])             # (N, 15, 3)
        poses15 = self._root_relative_15(poses15)
        return poses15[:, self.joint_indices, :]              # (N, J, 3)

    def _load_poses_from_ae(self, seq_dir: str) -> np.ndarray:
        """
        Run pretrained AutoEncoder on per-frame heatmaps and return
        predicted full 15-joint pose as (N, 15, 3).
        """
        if self.pose_estimator is None:
            raise ValueError("pose_estimator is not initialized for pose_source='ae_tensor'.")

        gt_path = os.path.join(seq_dir, "ground_truth.json")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground_truth.json in '{seq_dir}'")

        with open(gt_path) as f:
            gt = json.load(f)
        if not isinstance(gt, list):
            raise ValueError(f"Expected list in '{gt_path}', got {type(gt)}")

        rows = []
        for item in gt:
            if not isinstance(item, dict):
                continue
            image_name = item.get("image_name")
            if not image_name:
                continue
            rows.append((_frame_number(str(image_name)), str(image_name)))
        rows.sort(key=lambda x: x[0])

        heatmaps = []
        body_parts = []
        for _, image_name in rows:
            hm_path = os.path.join(seq_dir, self.heatmap_subdir, f"{image_name}_hm15x64x64.npy")
            if not os.path.exists(hm_path):
                continue
            hm = np.load(hm_path).astype(np.float32)
            if hm.shape != (15, 64, 64):
                continue
            heatmaps.append(hm)

            if self.use_body_part:
                bp_path = os.path.join(seq_dir, self.heatmap_subdir, f"{image_name}_bp4x64x64.npy")
                if not os.path.exists(bp_path):
                    raise FileNotFoundError(
                        f"Missing body part mask '{bp_path}' while use_body_part=True."
                    )
                bp = np.load(bp_path).astype(np.float32)
                if bp.shape != (4, 64, 64):
                    raise ValueError(f"Expected body part shape (4,64,64), got {bp.shape} at {bp_path}")
                body_parts.append(bp)

        if not heatmaps:
            raise ValueError(
                f"No valid heatmaps found in '{seq_dir}/{self.heatmap_subdir}' for pose inference."
            )

        hm_np = np.stack(heatmaps, axis=0)  # (N,15,64,64)
        bp_np = np.stack(body_parts, axis=0) if self.use_body_part else None
        poses15 = self.pose_estimator.estimate(
            heatmaps15=hm_np,
            body_parts4=bp_np,
            batch_size=self.ae_batch_size,
        )  # (N,15,3)
        poses15 = self._root_relative_15(poses15)
        return poses15  # (N,15,3)

    def _load_vr(self, seq_dir: str) -> np.ndarray:
        """
        Load VR tracking → (N, 9) float32.

        Source: ground_truth.json (one entry per 22-fps frame):
        [
          {"vrHMD": [x,y,z], "vrLeft": [x,y,z], "vrRight": [x,y,z]},
          ...
        ]

        Expects keys vrHMD / vrLeft / vrRight on each frame entry.
        """
        vr_path = os.path.join(seq_dir, "ground_truth.json")
        if not os.path.exists(vr_path):
            raise FileNotFoundError(f"Missing ground_truth.json in '{seq_dir}'")

        with open(vr_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("frames") or data.get("data") or data.get("vr") or []

        rows = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            hmd = self._extract_device_vec(entry, ["HMD", "hmd", "vrHMD", "vr_hmd"])
            left = self._extract_device_vec(entry, ["left", "Left", "vrLeft", "vr_left", "l_controller"])
            right = self._extract_device_vec(entry, ["right", "Right", "vrRight", "vr_right", "r_controller"])
            if hmd is None or left is None or right is None:
                continue
            rows.append(np.concatenate([hmd, left, right]) * VR_M_TO_CM)  # (9,), meters→cm

        if not rows:
            raise ValueError(f"No valid VR rows found in '{vr_path}'")
        return np.stack(rows, axis=0)   # (N, 9)

    def _load_cop(self, seq_dir: str) -> np.ndarray:
        """
        Load balance.json → (M, 2) float32  [CoPX, CoPY].
        CoP values are stored in centimeters and kept as-is (no unit conversion).
        All other modalities (pose, VR) are also converted to cm for consistency.

        Expected format (one entry per 2-fps CoP sample):
        [ {"CoPX": <float>, "CoPY": <float>}, ... ]
        """
        cop_path = os.path.join(seq_dir, "balance.json")
        if not os.path.exists(cop_path):
            raise FileNotFoundError(f"Missing balance.json in '{seq_dir}'")

        with open(cop_path) as f:
            data = json.load(f)

        # Handle both list-of-dicts and {"data": [...]} wrappers
        if isinstance(data, dict):
            data = data.get("data") or data.get("cop_data") or list(data.values())[0]

        rows = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            x = self._pick_numeric(entry, ["CoPX", "copX", "copx", "cop_x", "x"], default=0.0)
            y = self._pick_numeric(entry, ["CoPY", "copY", "copy", "cop_y", "y"], default=0.0)
            rows.append([x, y])
        if not rows:
            raise ValueError(f"No valid CoP rows found in '{cop_path}'")
        return np.array(rows, dtype=np.float32)   # (M, 2), already in cm

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict of tensors for one 11-frame window:

          pose_pos  : (W, Jp, 3) pose positions (pelvis-relative)
          pose_vel  : (W, Jp, 3) pose velocities (all loaded joints)
          vr_pos    : (W, 9)     VR device positions (mm)
          vr_vel    : (W, 9)     VR device velocities
          cop       : (2,)       ground-truth [CoPX, CoPY]

        where W = window_size (11).
        - pose_source='json':      Jp = len(joint_indices) (default 8 lower body)
        - pose_source='ae_tensor': Jp = 15 (full body)
        """
        s   = self.samples[idx]
        seq = s.sequence_dir
        sl  = slice(s.start_frame, s.start_frame + self.window_size)

        pose_pos = self._poses[seq][sl].copy()   # (W, J, 3)
        vr_pos   = self._vr[seq][sl].copy()      # (W, 9)
        cop      = self._cop[seq][s.cop_index]   # (2,)

        # Compute pose velocity for all currently loaded pose joints.
        pose_for_vel = pose_pos
        pose_vel = self.vel_computer.compute(
            pose_for_vel, method="central", smooth_window=self.smooth_window
        )   # (W, Jp, 3)

        vr_pos_3d = vr_pos.reshape(self.window_size, N_VR_DEVICES, 3)
        vr_vel_3d = self.vel_computer.compute(
            vr_pos_3d, method="central", smooth_window=self.smooth_window
        )   # (W, 3, 3)
        vr_vel = vr_vel_3d.reshape(self.window_size, N_VR_COORDS)   # (W, 9)

        return {
            "pose_pos": torch.from_numpy(pose_pos.astype(np.float32)),
            "pose_vel": torch.from_numpy(pose_vel.astype(np.float32)),
            "vr_pos":   torch.from_numpy(vr_pos.astype(np.float32)),
            "vr_vel":   torch.from_numpy(vr_vel.astype(np.float32)),
            "cop":      torch.from_numpy(cop.astype(np.float32)),
            "sequence_dir": seq,
        }
