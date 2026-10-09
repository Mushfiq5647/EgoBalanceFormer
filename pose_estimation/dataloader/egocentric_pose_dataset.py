import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.joint_mapping import map_19_to_preferred_15, map_19_to_unrealego_16


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EgocentricPoseFrame:
    sequence_dir: str
    image_stem: str  # e.g. "color_frame_120"
    image_path: str
    heatmap_path: str
    body_part_path: Optional[str]  # optional: bp4x64x64.npy for 38ch AE
    joints19_translation: np.ndarray  # (19, 3) in the same units as your json


def _read_list_file(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _resolve_image_path(sequence_dir: str, image_stem: str) -> str:
    candidates = [
        os.path.join(sequence_dir, f"{image_stem}.png"),
        os.path.join(sequence_dir, f"{image_stem}.jpg"),
        os.path.join(sequence_dir, f"{image_stem}.jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    p2 = os.path.join(sequence_dir, image_stem)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Could not find image '{image_stem}' in '{sequence_dir}'.")


class EgocentricPoseFromListFile(Dataset):
    """
    Loads (image, predicted 15x64x64 heatmap, mapped 16x3 GT pose) for your dataset.

    Expects each sequence_dir to contain:
      - ground_truth.json (list of entries with image_name + joints.translation[19,3])
      - images named like color_frame_*.png (or jpg/jpeg)
      - predicted heatmaps saved by our script under:
           <sequence_dir>/<heatmap_subdir>/<image_name>_hm15x64x64.npy
    """

    def __init__(
        self,
        list_file: str,
        heatmap_subdir: str = "unrealego_pred_heatmaps",
        image_size: int = 256,
        only_prefix: str = "color_frame",
        limit_per_sequence: Optional[int] = None,
        use_body_part: bool = False,
    ):
        self.list_file = list_file
        self.sequence_dirs = _read_list_file(list_file)
        self.heatmap_subdir = heatmap_subdir
        self.only_prefix = only_prefix
        self.limit_per_sequence = limit_per_sequence
        self.use_body_part = use_body_part

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.frames: List[EgocentricPoseFrame] = []
        self._index_frames()

    def _index_frames(self) -> None:
        frames: List[EgocentricPoseFrame] = []

        for seq_dir in self.sequence_dirs:
            gt_path = os.path.join(seq_dir, "ground_truth.json")
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Missing ground_truth.json at '{gt_path}'")

            with open(gt_path, "r") as f:
                gt = json.load(f)

            if not isinstance(gt, list):
                raise ValueError(f"Expected list in '{gt_path}', got {type(gt)}")

            count = 0
            for item in gt:
                image_name = item.get("image_name")
                if not image_name:
                    continue
                if self.only_prefix and not str(image_name).startswith(self.only_prefix):
                    continue

                joints = item.get("joints", {})
                translation = joints.get("translation")
                if translation is None:
                    continue

                joints19 = np.asarray(translation, dtype=np.float32)
                if joints19.shape != (19, 3):
                    # Skip malformed GT entries instead of crashing the entire split.
                    continue

                try:
                    image_path = _resolve_image_path(seq_dir, str(image_name))
                except FileNotFoundError:
                    continue  # skip ground truth when image is missing

                heatmap_path = os.path.join(
                    seq_dir,
                    self.heatmap_subdir,
                    f"{str(image_name)}_hm15x64x64.npy",
                )
                if not os.path.exists(heatmap_path):
                    raise FileNotFoundError(
                        f"Missing predicted heatmap at '{heatmap_path}'. "
                        f"Run generate_unrealego_heatmaps_from_folders.py or generate_egoglass_heatmaps_from_folders.py first."
                    )

                body_part_path = None
                if self.use_body_part:
                    bp_path = os.path.join(
                        seq_dir,
                        self.heatmap_subdir,
                        f"{str(image_name)}_bp4x64x64.npy",
                    )
                    if not os.path.exists(bp_path):
                        raise FileNotFoundError(
                            f"Missing body part mask at '{bp_path}'. "
                            f"Run generate_egoglass_heatmaps_from_folders.py with --save_body_part first."
                        )
                    body_part_path = bp_path

                frames.append(
                    EgocentricPoseFrame(
                        sequence_dir=seq_dir,
                        image_stem=str(image_name),
                        image_path=image_path,
                        heatmap_path=heatmap_path,
                        body_part_path=body_part_path,
                        joints19_translation=joints19,
                    )
                )

                count += 1
                if self.limit_per_sequence is not None and count >= self.limit_per_sequence:
                    break

        if len(frames) == 0:
            raise ValueError(f"No frames found from list_file='{self.list_file}'")

        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        frame = self.frames[idx]

        # Image (not strictly needed for AE training, but you requested to load it)
        img = Image.open(frame.image_path).convert("RGB")
        img_t = self.transform(img)  # [3,256,256]

        # Heatmap (15,64,64) predicted by pretrained HeatMap net
        hm = np.load(frame.heatmap_path).astype(np.float32)
        if hm.shape != (15, 64, 64):
            raise ValueError(f"Expected heatmap shape (15,64,64), got {hm.shape} at {frame.heatmap_path}")
        hm_t = torch.from_numpy(hm)  # [15,64,64]

        # Optional body part (4,64,64) for 38ch AE fusion
        bp_t = None
        if frame.body_part_path is not None:
            bp = np.load(frame.body_part_path).astype(np.float32)
            if bp.shape != (4, 64, 64):
                raise ValueError(f"Expected body part shape (4,64,64), got {bp.shape} at {frame.body_part_path}")
            bp_t = torch.from_numpy(bp)

        # Map GT joints 19 -> UnrealEgo 16 and preferred 15 (pelvis-relative inside mappings)
        pose16 = map_19_to_unrealego_16(frame.joints19_translation)  # (16,3)
        pose16_t = torch.from_numpy(pose16.astype(np.float32))  # [16,3]
        pose15 = map_19_to_preferred_15(frame.joints19_translation)  # (15,3)
        pose15_t = torch.from_numpy(pose15.astype(np.float32))  # [15,3]

        out = {
            "image": img_t,
            "heatmap15": hm_t,
            "gt_pose16": pose16_t,
            "gt_pose15": pose15_t,
            "sequence_dir": frame.sequence_dir,
            "image_stem": frame.image_stem,
            "image_path": frame.image_path,
            "heatmap_path": frame.heatmap_path,
        }
        if bp_t is not None:
            out["body_part4"] = bp_t
        return out

