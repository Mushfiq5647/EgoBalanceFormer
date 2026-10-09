"""
Generate EgoGlass heatmaps for your monocular egocentric dataset.

Uses pretrained EgoGlass HeatMap_left and HeatMap_right. For monocular input,
feeds the same image to both and uses the left-view heatmaps (15 ch).

Output: <sequence_dir>/egoglass_pred_heatmaps/<image_stem>_hm15x64x64.npy
       optionally <image_stem>_bp4x64x64.npy when --save_body_part
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloader.egocentric_monocular_dataset import EgocentricMonocularFromFoldersTxt
from model.network import HeatMap_EgoGlass


def _make_opt(num_heatmap: int = 15, init_imagenet: bool = False):
    return SimpleNamespace(
        num_heatmap=num_heatmap,
        init_ImageNet=init_imagenet,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders_txt",
        type=str,
        default="pose_estimation/utils/folders.txt",
        help="Path to folders.txt (one sequence directory per line).",
    )
    parser.add_argument(
        "--weights_left",
        type=str,
        default="log/egoglass_B16/best_net_HeatMap_left.pth",
        help="Path to pretrained EgoGlass HeatMap_left weights.",
    )
    parser.add_argument(
        "--weights_right",
        type=str,
        default="log/egoglass_B16/best_net_HeatMap_right.pth",
        help="Path to pretrained EgoGlass HeatMap_right weights.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="egoglass_pred_heatmaps",
        help="Subdirectory under each sequence_dir to store .npy heatmaps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--limit_per_sequence",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_body_part",
        action="store_true",
        help="Also save 4ch body part masks (bp4x64x64.npy) for AE training with fusion.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = EgocentricMonocularFromFoldersTxt(
        folders_txt=args.folders_txt,
        image_size=256,
        only_prefix="color_frame",
        limit_per_sequence=args.limit_per_sequence,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    opt = _make_opt(num_heatmap=15, init_imagenet=False)
    net_left = HeatMap_EgoGlass(opt).to(device)
    net_right = HeatMap_EgoGlass(opt).to(device)

    for p in (args.weights_left, args.weights_right):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Weights not found: '{p}'")

    net_left.load_state_dict(torch.load(args.weights_left, map_location="cpu"), strict=False)
    net_right.load_state_dict(torch.load(args.weights_right, map_location="cpu"), strict=False)
    net_left.eval()
    net_right.eval()

    with torch.no_grad():
        for batch in dl:
            images = batch["image"].to(device)
            seq_dirs = batch["sequence_dir"]
            image_stems = batch["image_stem"]

            # Same image as left and right (monocular)
            out_left = net_left(images)
            out_right = net_right(images)
            hm_left = out_left[0] if isinstance(out_left, tuple) else out_left
            hm_right = out_right[0] if isinstance(out_right, tuple) else out_right
            bp_left = out_left[1] if isinstance(out_left, tuple) and len(out_left) > 1 else None

            heatmaps = hm_left.detach().cpu().numpy().astype(np.float32)
            body_parts = bp_left.detach().cpu().numpy().astype(np.float32) if bp_left is not None else None

            for i in range(heatmaps.shape[0]):
                out_dir = os.path.join(seq_dirs[i], args.output_subdir)
                os.makedirs(out_dir, exist_ok=True)

                hm_path = os.path.join(out_dir, f"{image_stems[i]}_hm15x64x64.npy")
                if args.overwrite or not os.path.exists(hm_path):
                    np.save(hm_path, heatmaps[i])

                if args.save_body_part and body_parts is not None:
                    bp_path = os.path.join(out_dir, f"{image_stems[i]}_bp4x64x64.npy")
                    if args.overwrite or not os.path.exists(bp_path):
                        np.save(bp_path, body_parts[i])

    print("Done.")


if __name__ == "__main__":
    main()
