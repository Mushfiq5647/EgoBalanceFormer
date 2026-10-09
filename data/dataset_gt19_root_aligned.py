"""
BalanceDatasetGT19RootAligned:
  True timestamp-aligned 19-joint dataset.

  For each CoP sample timestamp, find the nearest pose/VR frame timestamp and
  build a causal window of `window_size` frames ending at that matched frame.
  Samples with pose↔CoP mismatch larger than `max_offset_ms` are discarded.

Unit summary (all cm):
  pose_pos : (W, 19, 3)  cm, root-relative
  vr_pos   : (W, 6)      cm, [Left-HMD, Right-HMD]
  cop      : (2,)        cm
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

from data.dataset import _frame_number


COP_FPS = 2.0
COP_INTERVAL = 1.0 / COP_FPS   # 0.5 s per CoP sample


@dataclass(frozen=True)
class AlignedSample:
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


class BalanceDatasetGT19RootAligned(Dataset):
    """
    Timestamp-aligned 19-joint dataset.

    Accepts the same kwargs as BalanceDataset / BalanceDatasetGT19Root so
    it can be used as a drop-in replacement.  Extra kwargs are ignored.
    """

    def __init__(
        self,
        list_file: str,
        window_size: int = 11,
        fps: float = 22.0,
        joint_indices: Optional[List[int]] = None,
        max_offset_ms: float = 250.0,
        **kwargs,
    ):
        self.window_size = window_size
        self.fps = fps
        self.max_offset_ms = max_offset_ms
        self.joint_indices = (
            joint_indices if joint_indices is not None else list(range(19))
        )

        with open(list_file) as f:
            self.sequence_dirs = [
                l.strip() for l in f if l.strip() and not l.startswith("#")
            ]

        self._poses: Dict[str, np.ndarray] = {}
        self._vr:    Dict[str, np.ndarray] = {}
        self._cop:   Dict[str, np.ndarray] = {}

        self.samples: List[AlignedSample] = []
        skipped_seqs = 0
        total_pose_skipped = 0
        total_cop_skipped = 0

        for seq_dir in self.sequence_dirs:
            try:
                result = self._load_and_align(seq_dir)
                poses, vr, cop, samples, pose_skip, cop_skip = result
                if not samples:
                    print(f"  [skip] {os.path.basename(seq_dir)}: "
                          f"0 aligned windows")
                    skipped_seqs += 1
                    continue
                self._poses[seq_dir] = poses
                self._vr[seq_dir] = vr
                self._cop[seq_dir] = cop
                self.samples.extend(samples)
                total_pose_skipped += pose_skip
                total_cop_skipped += cop_skip
            except (FileNotFoundError, ValueError) as e:
                print(f"  [skip] {os.path.basename(seq_dir)}: {e}")
                skipped_seqs += 1

        if not self.samples:
            raise ValueError(f"No aligned windows from '{list_file}'")

        print(
            f"[Aligned] {len(self.samples)} windows from "
            f"{len(self._poses)}/{len(self.sequence_dirs)} sequences "
            f"(skipped {skipped_seqs} seqs, "
            f"pose_frames_trimmed={total_pose_skipped}, "
            f"cop_samples_trimmed={total_cop_skipped}, "
            f"max_offset_ms={self.max_offset_ms:.1f})"
        )

    # ------------------------------------------------------------------
    def _load_and_align(self, seq_dir: str):
        """
        Returns (poses, vr, cop, samples, pose_frames_skipped, cop_samples_skipped).
        """
        gt_path  = os.path.join(seq_dir, "ground_truth.json")
        bal_path = os.path.join(seq_dir, "balance.json")
        if not os.path.exists(gt_path):
            raise FileNotFoundError("Missing ground_truth.json")
        if not os.path.exists(bal_path):
            raise FileNotFoundError("Missing balance.json")

        # ---- optional metadata timestamps (fallback only) ----
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
            pose = (pose - pose[0:1, :]) / 10.0        # root-relative, mm → cm

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
            left_rel  = (left  - hmd) * 100.0           # m → cm
            right_rel = (right - hmd) * 100.0
            vr = np.concatenate([left_rel, right_rel]).astype(np.float32)

            # Prefer timestamp from JSON entry; fall back to metadata map
            ts_str = entry.get("timestamp") or frame_to_ts.get(fid)
            if not ts_str:
                continue
            try:
                ts_sec = _parse_ts(ts_str)
            except ValueError:
                continue
            valid.append((fid, ts_sec, pose, vr))

        if not valid:
            raise ValueError("No valid pose+VR frames")
        valid.sort(key=lambda x: x[1])

        pose_ts = np.asarray([v[1] for v in valid], dtype=np.float64)
        poses_all = np.stack([v[2] for v in valid])     # (N, 19, 3)
        vr_all    = np.stack([v[3] for v in valid])     # (N, 6)

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

        samples: List[AlignedSample] = []
        pose_skip = 0
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
                AlignedSample(
                    sequence_dir=seq_dir,
                    start_frame=start_idx,
                    cop_index=cop_index,
                )
            )

        poses = poses_all
        vr = vr_all
        cop = cop_all
        pose_skip = max(0, self.window_size - 1)
        return poses, vr, cop, samples, pose_skip, cop_skip

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Units (all cm):
          pose_pos : (W, 19, 3)  root-relative
          vr_pos   : (W, 6)      [Left-HMD, Right-HMD]
          cop      : (2,)        [CoPX, CoPY]
        """
        s   = self.samples[idx]
        seq = s.sequence_dir
        sl  = slice(s.start_frame, s.start_frame + self.window_size)

        return {
            "pose_pos": torch.from_numpy(
                self._poses[seq][sl].copy().astype(np.float32)
            ),
            "vr_pos": torch.from_numpy(
                self._vr[seq][sl].copy().astype(np.float32)
            ),
            "cop": torch.from_numpy(
                self._cop[seq][s.cop_index].copy().astype(np.float32)
            ),
            "sequence_dir": seq,
        }
