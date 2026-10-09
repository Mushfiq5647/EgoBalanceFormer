"""
Generate heatmaps with your B=16, T=32 trained checkpoint from `heatmaps/network_heatmap.py`,
but in a **frame-by-frame** way for your custom dataset.

- Model: `HeatMap_Network` from `heatmaps/network_heatmap.py`
- Checkpoint: e.g. `log/egopwsceneego_heatmap_shared_B16/heatmap_best.ckpt`
- Output per frame: `(15, 64, 64)` logits saved as `<image_stem>_hm15x64x64.npy`

These .npy heatmaps can then be used as input to your own model.
"""

import argparse
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


# Make repo root importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloader.egocentric_monocular_dataset import EgocentricMonocularFromFoldersTxt  # noqa: E402
from heatmaps.network_heatmap import HeatMap_Network  # noqa: E402


def _make_opt(num_heatmap: int = 15, init_imagenet: bool = False):
    """
    Minimal opt object needed by HeatMap_Network / Encoder_Block.
    """
    return SimpleNamespace(
        num_heatmap=num_heatmap,
        init_ImageNet=init_imagenet,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders_txt",
        type=str,
        default="pose_estimation/utils/train.txt",
        help="Path to folders.txt (one sequence directory per line).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="log/egopwsceneego_heatmap_shared_B16/heatmap_best.ckpt",
        help="Path to trained HeatMap_Network checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        help="Backbone used when training the checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for frame-by-frame heatmap inference.",
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
        default="custom_pred_heatmaps",
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
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Input resize for the RGB images (should match training crop_size).",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=200,
        help="Print a progress message every N saved frames.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + loader: one image per sample, no temporal dimension.
    ds = EgocentricMonocularFromFoldersTxt(
        folders_txt=args.folders_txt,
        image_size=args.image_size,
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

    # HeatMap model (monocular) using your checkpoint.
    opt = _make_opt(num_heatmap=15, init_imagenet=False)
    net = HeatMap_Network(opt, model_name=args.backbone).to(device)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Heatmap weights not found at '{args.weights}'")

    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        # In case the checkpoint was saved as {"state_dict": ...}
        state = state["state_dict"]
    net.load_state_dict(state)
    net.eval()

    # Inference + save
    n_saved = 0
    n_seen = 0
    total = len(ds)
    print(f"Start heatmap generation: total_frames={total}, batch_size={args.batch_size}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            images = batch["image"].to(device)  # [B,3,H,W]
            seq_dirs = batch["sequence_dir"]
            image_stems = batch["image_stem"]

            # Frame-by-frame heatmaps (no temporal dimension).
            pred = net(images)                   # [B, 15, 64, 64] logits
            pred = torch.sigmoid(pred)          # convert to probabilities in [0,1]
            heatmaps = pred.detach().cpu().numpy().astype(np.float32)

            for i in range(heatmaps.shape[0]):
                out_dir = os.path.join(seq_dirs[i], args.output_subdir)
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, f"{image_stems[i]}_hm15x64x64.npy")

                if heatmaps[i].shape != (15, 64, 64):
                    raise ValueError(
                        f"Expected heatmap shape (15, 64, 64), got {heatmaps[i].shape}"
                    )
                np.save(out_path, heatmaps[i])
                n_saved += 1
                n_seen += 1

                if args.print_every > 0 and (n_seen % args.print_every == 0):
                    print(
                        f"[{n_seen}/{total}] saved heatmaps "
                        f"(batch={batch_idx}, seq_dir='{seq_dirs[i]}', image='{image_stems[i]}')"
                    )

    print(f"Done. Saved {n_saved} heatmaps into subdir '{args.output_subdir}'.")


if __name__ == "__main__":
    main()

