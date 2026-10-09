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

from dataloader.egocentric_pose_dataset import EgocentricPoseFromListFile  # noqa: E402
from model.network import AutoEncoder  # noqa: E402
from utils.loss import LossFuncMPJPE, LossFuncCosSim  # noqa: E402


def _make_opt(num_heatmap: int = 15, ae_hidden_size: int = 20, num_joints: int = 15):
    # Minimal opt object needed by AutoEncoder
    return SimpleNamespace(
        num_heatmap=num_heatmap,
        ae_hidden_size=ae_hidden_size,
        num_joints=num_joints,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="pose_estimation/utils/train.txt", help="List file of sequence dirs.")
    parser.add_argument(
        "--val_list",
        type=str,
        default=None,
        help="Optional list file of sequence dirs for validation (MPJPE only).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ae_hidden_size", type=int, default=20)
    parser.add_argument(
        "--num_joints",
        type=int,
        default=15,
        choices=[15, 16],
        help="Pose output joints. Use 15 for your preferred convention, 16 for UnrealEgo convention.",
    )
    parser.add_argument("--limit_per_sequence", type=int, default=None)
    # Loss weights mirroring UnrealEgo's TrainOptions defaults
    parser.add_argument("--lambda_mpjpe", type=float, default=1.0)
    parser.add_argument("--lambda_cos_sim", type=float, default=-1e-2)
    parser.add_argument("--lambda_heatmap_rec", type=float, default=1e-3)
    parser.add_argument("--log_dir", type=str, default="log/egocentric_autoencoder_from_heatmaps")
    parser.add_argument(
        "--heatmap_subdir",
        type=str,
        default="unrealego_pred_heatmaps",
        help="Subdir under sequence_dir where heatmaps are stored (unrealego_pred_heatmaps or egoglass_pred_heatmaps).",
    )
    parser.add_argument(
        "--pretrained_ae",
        type=str,
        default=None,
        help="Optional path to pretrained AutoEncoder weights (e.g. EgoGlass best_net_AutoEncoder.pth).",
    )
    parser.add_argument(
        "--use_body_part",
        action="store_true",
        help="Load body part masks and use 38ch AE (heatmaps + body part fusion).",
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset / loaders
    train_ds = EgocentricPoseFromListFile(
        list_file=args.train_list,
        heatmap_subdir=args.heatmap_subdir,
        limit_per_sequence=args.limit_per_sequence,
        use_body_part=args.use_body_part,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_dl = None
    if args.val_list is not None:
        if not os.path.exists(args.val_list):
            raise FileNotFoundError(f"--val_list file not found: '{args.val_list}'")
        try:
            val_ds = EgocentricPoseFromListFile(
                list_file=args.val_list,
                heatmap_subdir=args.heatmap_subdir,
                limit_per_sequence=args.limit_per_sequence,
                use_body_part=args.use_body_part,
            )
            val_dl = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
        except ValueError as e:
            # If the val split has no usable frames, proceed with training only.
            print(f"[warn] Skipping validation: {e}")
            val_dl = None

    # Model: AutoEncoder (monocular 15ch, optionally +4ch body part = 19ch)
    extra_ch = 4 if args.use_body_part else 0
    opt = _make_opt(num_heatmap=15, ae_hidden_size=args.ae_hidden_size, num_joints=args.num_joints)
    # Monocular AE: input_channel_scale=1 → encoder sees 15ch heatmaps (+optional 4ch body part).
    net = AutoEncoder(opt, input_channel_scale=1, extra_input_channels=extra_ch).to(device)

    if args.pretrained_ae and os.path.exists(args.pretrained_ae):
        net.load_state_dict(torch.load(args.pretrained_ae, map_location="cpu"), strict=False)
        print(f"Loaded pretrained AutoEncoder from {args.pretrained_ae}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mpjpe = LossFuncMPJPE()
    loss_cos = LossFuncCosSim()
    loss_mse = nn.MSELoss()

    def _to_ae_input(batch):
        # Monocular input: use 15ch heatmaps directly (no duplication)
        hm15 = batch["heatmap15"].to(device)  # [B,15,64,64]
        if "body_part4" in batch and batch["body_part4"] is not None:
            bp4 = batch["body_part4"].to(device)  # [B,4,64,64]
            return torch.cat([hm15, bp4], dim=1)  # [B,19,64,64]
        return hm15

    def _get_gt_pose(batch):
        key = "gt_pose15" if args.num_joints == 15 else "gt_pose16"
        return batch[key].to(device)

    def _run_eval(dl):
        net.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in dl:
                ae_input = _to_ae_input(batch)
                gt = _get_gt_pose(batch)
                pred_pose = net.predict_pose(ae_input)
                val = loss_mpjpe(pred_pose, gt).item()
                total += val * ae_input.shape[0]
                n += ae_input.shape[0]
        net.train()
        return total / max(n, 1)

    # Train loop
    net.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        seen = 0

        for step, batch in enumerate(train_dl, start=1):
            ae_input = _to_ae_input(batch)
            gt = _get_gt_pose(batch)

            # Forward AutoEncoder: pose + reconstructed heatmaps (decoder outputs 15ch heatmaps)
            pred_pose, rec_hm = net(ae_input)

            # === Losses mirroring UnrealEgoAutoEncoderModel ===
            loss_pose = loss_mpjpe(pred_pose, gt)
            loss_cos_sim = loss_cos(pred_pose, gt)
            # Heatmap reconstruction: compare to the 15ch heatmaps (first 15 ch of input)
            hm15 = ae_input[:, :15]
            loss_hm_rec = loss_mse(rec_hm, hm15.detach())

            # Total loss: same structure as UnrealEgo (pose + cos_sim + hm_rec)
            loss = (
                args.lambda_mpjpe * loss_pose
                + args.lambda_cos_sim * args.lambda_mpjpe * loss_cos_sim
                + args.lambda_heatmap_rec * loss_hm_rec
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * ae_input.shape[0]
            seen += ae_input.shape[0]

            if step % 200 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss={loss.item():.6f} "
                    f"pose={loss_pose.item():.6f} "
                    f"cos={loss_cos_sim.item():.6f} "
                    f"hm_rec={loss_hm_rec.item():.6f}"
                )

        train_loss = running / max(seen, 1)
        msg = f"epoch {epoch}/{args.epochs} train_loss={train_loss:.6f}"

        if val_dl is not None:
            val_mpjpe = _run_eval(val_dl)
            msg += f" val_mpjpe={val_mpjpe:.6f}"

        print(msg)

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.log_dir, f"epoch_{epoch}_net_AutoEncoder.pth")
        torch.save(net.state_dict(), ckpt_path)

    print("Training done.")


if __name__ == "__main__":
    main()

