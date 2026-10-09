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

from dataloader.egocentric_monocular_dataset import EgocentricMonocularFromFoldersTxt  # noqa: E402
from model.network import HeatMap_UnrealEgo_Shared  # noqa: E402


def _make_opt(num_heatmap: int = 15, init_imagenet: bool = False):
    # Minimal opt object needed by HeatMap_UnrealEgo_Shared / Encoder_Block
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
        "--weights",
        type=str,
        default="log/unrealego_heatmap_shared_B16/best_net_HeatMap.pth",
        help="Path to pretrained UnrealEgo HeatMap weights.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for heatmap inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="unrealego_pred_heatmaps",
        help="Subdirectory created under each sequence_dir to store .npy heatmaps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing heatmap .npy files if present.",
    )
    parser.add_argument(
        "--limit_per_sequence",
        type=int,
        default=None,
        help="Optional: limit number of frames processed per sequence directory.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + loader
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

    # Load pretrained HeatMap net (stereo). We'll feed the same monocular image twice.
    opt = _make_opt(num_heatmap=15, init_imagenet=False)
    net = HeatMap_UnrealEgo_Shared(opt).to(device)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Heatmap weights not found at '{args.weights}'")

    state = torch.load(args.weights, map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    # Inference + save
    with torch.no_grad():
        for batch in dl:
            images = batch["image"].to(device)  # [B,3,256,256]
            seq_dirs = batch["sequence_dir"]
            image_stems = batch["image_stem"]

            # Use same image as left/right to run stereo net
            pred_cat = net(images, images)  # [B, 2*num_heatmap, 64, 64]
            pred_left, pred_right = torch.chunk(pred_cat, 2, dim=1)  # each [B, 15, 64, 64]

            # For monocular use, left/right should be identical if inputs are identical.
            heatmaps = pred_left.detach().cpu().numpy().astype(np.float32)  # [B,15,64,64]

            for i in range(heatmaps.shape[0]):
                out_dir = os.path.join(seq_dirs[i], args.output_subdir)
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, f"{image_stems[i]}_hm15x64x64.npy")
                if (not args.overwrite) and os.path.exists(out_path):
                    continue
                np.save(out_path, heatmaps[i])

    print("Done.")


if __name__ == "__main__":
    main()

