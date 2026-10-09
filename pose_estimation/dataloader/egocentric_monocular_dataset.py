import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EgocentricFrame:
    sequence_dir: str
    image_stem: str  # e.g. "color_frame_120" (no extension)
    image_path: str  # resolved file path


def _read_folders_list(path_to_folders_txt: str) -> List[str]:
    with open(path_to_folders_txt, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _resolve_image_path(sequence_dir: str, image_stem: str) -> str:
    # Most common: .png
    candidates = [
        os.path.join(sequence_dir, f"{image_stem}.png"),
        os.path.join(sequence_dir, f"{image_stem}.jpg"),
        os.path.join(sequence_dir, f"{image_stem}.jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback: allow image_name already includes extension
    p2 = os.path.join(sequence_dir, image_stem)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(
        f"Could not find image for image_name='{image_stem}' in '{sequence_dir}'. "
        f"Tried: {candidates + [p2]}"
    )


class EgocentricMonocularFromFoldersTxt(Dataset):
    """
    Dataset that reads sequence directories from a folders.txt file.
    Each directory must contain a ground_truth.json list with entries like:
      { "image_name": "color_frame_121", "joints": { "translation": [[...], ...] } }

    This dataset yields images (preprocessed) and enough metadata to save outputs.
    """

    def __init__(
        self,
        folders_txt: str,
        image_size: int = 256,
        only_prefix: str = "color_frame",
        limit_per_sequence: Optional[int] = None,
    ):
        self.folders_txt = folders_txt
        self.sequence_dirs = _read_folders_list(folders_txt)
        self.only_prefix = only_prefix
        self.limit_per_sequence = limit_per_sequence

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.frames: List[EgocentricFrame] = []
        self._index_frames()

    def _index_frames(self) -> None:
        frames: List[EgocentricFrame] = []
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

                try:
                    image_path = _resolve_image_path(seq_dir, str(image_name))
                except FileNotFoundError:
                    continue  # skip ground truth when image is missing

                frames.append(
                    EgocentricFrame(
                        sequence_dir=seq_dir,
                        image_stem=str(image_name),
                        image_path=image_path,
                    )
                )

                count += 1
                if self.limit_per_sequence is not None and count >= self.limit_per_sequence:
                    break

        if len(frames) == 0:
            raise ValueError(f"No frames found from folders_txt='{self.folders_txt}'")
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        # Robust loading: skip corrupted / unreadable images.
        num_frames = len(self.frames)
        start_idx = idx % num_frames
        cur_idx = start_idx

        while True:
            frame = self.frames[cur_idx]
            try:
                img = Image.open(frame.image_path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                # Log once per bad file and move to the next frame.
                print(f"[EgocentricMonocularFromFoldersTxt] Skipping unreadable image: {frame.image_path}")
                cur_idx = (cur_idx + 1) % num_frames
                if cur_idx == start_idx:
                    # All images are unreadable; give up.
                    raise RuntimeError(
                        f"All images in dataset from '{self.folders_txt}' are unreadable."
                    )
                continue

            img_t = self.transform(img)  # [3, H, W] ImageNet-normalized

            return {
                "image": img_t,
                "sequence_dir": frame.sequence_dir,
                "image_stem": frame.image_stem,
                "image_path": frame.image_path,
            }

