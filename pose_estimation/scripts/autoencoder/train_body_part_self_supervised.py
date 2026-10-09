"""
Self-supervised training of the body part branch (conv_body_part) in EgoGlass heatmap nets.

Freezes backbone + heatmap head; trains only conv_body_part to predict pseudo-limb masks
derived from the predicted heatmaps via heatmap_to_pseudo_limb_mask().
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloader.egocentric_monocular_dataset import EgocentricMonocularFromFoldersTxt
from model.network import HeatMap_EgoGlass
from utils.body_part import heatmap_to_pseudo_limb_mask


def _make_opt(num_heatmap: int = 15, init_imagenet: bool = False):
    return SimpleNamespace(
        num_heatmap=num_heatmap,
        init_ImageNet=init_imagenet,
    )


def _freeze_except_body_part(net: HeatMap_EgoGlass) -> None:
    for name, param in net.named_parameters():
        if "conv_body_part" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="pose_estimation/utils/train.txt")
    parser.add_argument("--weights_left", type=str, default="log/egoglass_B16/best_net_HeatMap_left.pth")
    parser.add_argument("--weights_right", type=str, default="log/egoglass_B16/best_net_HeatMap_right.pth")
    parser.add_argument("--log_dir", type=str, default="log/egoglass_body_part_self_supervised")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit_per_sequence", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = EgocentricMonocularFromFoldersTxt(
        folders_txt=args.train_list,
        image_size=256,
        only_prefix="color_frame",
        limit_per_sequence=args.limit_per_sequence,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
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

    _freeze_except_body_part(net_left)
    _freeze_except_body_part(net_right)

    params_left = [p for p in net_left.parameters() if p.requires_grad]
    params_right = [p for p in net_right.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_left + params_right, lr=args.lr)
    loss_fn = nn.MSELoss()

    net_left.train()
    net_right.train()

    num_batches = len(dl)
    print(f"Total batches per epoch: {num_batches}")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n = 0
        print(f"Starting epoch {epoch}/{args.epochs}...")

        for batch_idx, batch in enumerate(dl):
            images = batch["image"].to(device)

            # Forward (both left and right; for monocular same image)
            out_left = net_left(images)
            out_right = net_right(images)

            hm_left = out_left[0] if isinstance(out_left, tuple) else out_left
            bp_left = out_left[1] if isinstance(out_left, tuple) and len(out_left) > 1 else None
            hm_right = out_right[0] if isinstance(out_right, tuple) else out_right
            bp_right = out_right[1] if isinstance(out_right, tuple) and len(out_right) > 1 else None

            if bp_left is None or bp_right is None:
                raise RuntimeError("HeatMap_EgoGlass must return (heatmap, body_part)")

            # Pseudo-GT from predicted heatmaps (detach)
            gt_bp_left = heatmap_to_pseudo_limb_mask(hm_left.detach()).to(device)
            gt_bp_right = heatmap_to_pseudo_limb_mask(hm_right.detach()).to(device)

            loss_left = loss_fn(bp_left, gt_bp_left)
            loss_right = loss_fn(bp_right, gt_bp_right)
            loss = loss_left + loss_right

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            n += images.shape[0]

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                print(f"  epoch {epoch} batch {batch_idx + 1}/{num_batches} loss={loss.item():.6f}")

        mean_loss = total_loss / max(n, 1)
        print(f"epoch {epoch}/{args.epochs} loss={mean_loss:.6f}")

        # Save checkpoints
        torch.save(net_left.state_dict(), os.path.join(args.log_dir, f"epoch_{epoch}_net_HeatMap_left.pth"))
        torch.save(net_right.state_dict(), os.path.join(args.log_dir, f"epoch_{epoch}_net_HeatMap_right.pth"))
        print(f"epoch {epoch} has completed")

    print("Done. Use these weights for generate_egoglass_heatmaps_from_folders.py with --save_body_part.")


if __name__ == "__main__":
    main()
