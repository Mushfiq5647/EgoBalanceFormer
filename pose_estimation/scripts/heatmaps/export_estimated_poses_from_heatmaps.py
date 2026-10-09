import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloader.egocentric_pose_dataset import EgocentricPoseFromListFile  # noqa: E402
from model.network import AutoEncoder  # noqa: E402


PREFERRED_JOINT_NAMES_15 = [
    "Neck",
    "Right_shoulder",
    "Right_elbow",
    "Right_wrist",
    "Left_shoulder",
    "Left_elbow",
    "Left_wrist",
    "Right_hip",
    "Right_knee",
    "Right_ankle",
    "Right_foot",
    "Left_hip",
    "Left_knee",
    "Left_ankle",
    "Left_foot",
]


def _make_opt(num_heatmap: int = 15, ae_hidden_size: int = 20, num_joints: int = 15):
    return SimpleNamespace(
        num_heatmap=num_heatmap,
        ae_hidden_size=ae_hidden_size,
        num_joints=num_joints,
    )


def _frame_number(image_stem: str) -> int:
    parts = str(image_stem).split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export frame-wise estimated poses to estimated_poses.json per sequence."
    )
    parser.add_argument(
        "--list_file",
        type=str,
        required=True,
        help="Text file with one sequence directory per line.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="log/egocentric_ae_15j/epoch_20_net_AutoEncoder.pth",
        help="Path to AutoEncoder checkpoint.",
    )
    parser.add_argument(
        "--heatmap_subdir",
        type=str,
        default="unrealego_pred_heatmaps",
        help="Subdirectory containing *_hm15x64x64.npy heatmaps.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="estimated_poses.json",
        help="Output filename saved inside each sequence directory.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ae_hidden_size", type=int, default=20)
    parser.add_argument("--limit_per_sequence", type=int, default=None)
    parser.add_argument(
        "--use_body_part",
        action="store_true",
        help="Use 38-channel AE input (heatmaps + body-part masks).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing estimated_poses.json.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = EgocentricPoseFromListFile(
        list_file=args.list_file,
        heatmap_subdir=args.heatmap_subdir,
        limit_per_sequence=args.limit_per_sequence,
        use_body_part=args.use_body_part,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    extra_ch = 8 if args.use_body_part else 0
    opt = _make_opt(num_heatmap=15, ae_hidden_size=args.ae_hidden_size, num_joints=15)
    net = AutoEncoder(opt, input_channel_scale=2, extra_input_channels=extra_ch).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: '{args.checkpoint}'")
    state = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    by_sequence = defaultdict(list)

    with torch.no_grad():
        for batch in dl:
            hm15 = batch["heatmap15"].to(device)
            hm30 = torch.cat([hm15, hm15], dim=1)

            if args.use_body_part:
                if "body_part4" not in batch:
                    raise ValueError("--use_body_part is set but body_part4 is missing from dataset batch.")
                bp4 = batch["body_part4"].to(device)
                bp8 = torch.cat([bp4, bp4], dim=1)
                ae_input = torch.cat([hm30, bp8], dim=1)
            else:
                ae_input = hm30

            pred_pose15 = net.predict_pose(ae_input).detach().cpu().numpy()  # (B, 15, 3)

            seq_dirs = batch["sequence_dir"]
            stems = batch["image_stem"]
            for i in range(pred_pose15.shape[0]):
                by_sequence[seq_dirs[i]].append(
                    {
                        "image_name": str(stems[i]),
                        "joints": {
                            "joint_names": PREFERRED_JOINT_NAMES_15,
                            "translation": pred_pose15[i].tolist(),
                        },
                    }
                )

    written = 0
    skipped = 0
    for seq_dir, entries in by_sequence.items():
        out_path = os.path.join(seq_dir, args.output_filename)
        if (not args.overwrite) and os.path.exists(out_path):
            skipped += 1
            continue

        entries.sort(key=lambda e: _frame_number(e.get("image_name", "")))
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2)
        written += 1

    print(
        f"Export complete. sequences_written={written} sequences_skipped={skipped} "
        f"output_filename='{args.output_filename}'"
    )


if __name__ == "__main__":
    main()
