import json
import os
from typing import List, Optional

import numpy as np

from data.dataset import BalanceDataset, _frame_number


class BalanceDatasetGT19Root(BalanceDataset):

    def __init__(
        self,
        list_file: str,
        window_size: int = 11,
        fps: float = 22.0,
        smooth_window: int = 3,
        joint_indices: Optional[List[int]] = None,
        pose_filename: str = "ground_truth.json",
        pose_source: str = "json",
        heatmap_subdir: str = "custom_pred_heatmaps",
        ae_checkpoint: Optional[str] = None,
        ae_device: str = "cpu",
        ae_hidden_size: int = 20,
        ae_batch_size: int = 128,
        use_body_part: bool = False,
        vr_filename: str = "vr_data.json",
    ):
        if pose_source != "json":
            raise ValueError("BalanceDatasetGT19Root only supports pose_source='json'.")
        if joint_indices is None:
            joint_indices = list(range(19))
        super().__init__(
            list_file=list_file,
            window_size=window_size,
            fps=fps,
            smooth_window=smooth_window,
            joint_indices=joint_indices,
            pose_filename=pose_filename,
            pose_source="json",
            heatmap_subdir=heatmap_subdir,
            ae_checkpoint=ae_checkpoint,
            ae_device=ae_device,
            ae_hidden_size=ae_hidden_size,
            ae_batch_size=ae_batch_size,
            use_body_part=use_body_part,
            vr_filename=vr_filename,
        )

    def _load_poses(self, seq_dir: str) -> np.ndarray:
        """
        Load ground_truth.json pose translation as raw 19 joints.
        Pose is converted from mm to meters after root-centering.
        Returns:
          (N, J, 3), where J = len(self.joint_indices) (default 19)
        """
        pose_path = os.path.join(seq_dir, "ground_truth.json")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing ground_truth.json in '{seq_dir}'")

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
            if arr.shape != (19, 3):
                continue

            # Root-relative with first joint (Root index 0)
            arr = arr - arr[0:1, :]
            # Convert pose from mm to meters
            arr = arr / 1000.0
            valid.append((frame_id, arr))

        if not valid:
            raise ValueError(f"No valid 19-joint pose frames in '{pose_path}'")

        valid.sort(key=lambda x: x[0])
        poses19 = np.stack([v[1] for v in valid])         # (N,19,3)
        return poses19[:, self.joint_indices, :]          # (N,J,3)

    def _load_vr(self, seq_dir: str) -> np.ndarray:
        """
        Load VR tracking from ground_truth.json in meters.
        Returns:
          (N, 9) float32 in meters
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
            rows.append(np.concatenate([hmd, left, right]))  # meters

        if not rows:
            raise ValueError(f"No valid VR rows found in '{vr_path}'")
        return np.stack(rows, axis=0)  # (N, 9), meters
