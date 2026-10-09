"""
Find unreadable/corrupted image files in sequence folders listed in a folders.txt file.

Usage (from repo root):
  python pose_estimation/scripts/tools/find_bad_images.py --folders_txt pose_estimation/utils/train.txt

It will try to open images with PIL and print any paths that raise UnidentifiedImageError / OSError.
"""

import argparse
import glob
import os

from PIL import Image, UnidentifiedImageError


def read_folders_list(path_to_folders_txt: str):
    with open(path_to_folders_txt, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders_txt",
        type=str,
        default="pose_estimation/utils/train.txt",
        help="Text file with one sequence directory per line.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="color_frame*.png",
        help="Glob pattern for images inside each sequence dir (default: color_frame*.png).",
    )
    args = parser.parse_args()

    seq_dirs = read_folders_list(args.folders_txt)
    bad_images = []
    total = 0
    per_dir_total = {}
    per_dir_bad = {}

    for seq_dir in seq_dirs:
        img_paths = sorted(glob.glob(os.path.join(seq_dir, args.pattern)))
        per_dir_total[seq_dir] = len(img_paths)
        per_dir_bad.setdefault(seq_dir, 0)
        for p in img_paths:
            total += 1
            try:
                with Image.open(p) as img:
                    img.verify()  # quick check
            except (UnidentifiedImageError, OSError) as e:
                print(f"[BAD] {p}  ({type(e).__name__}: {e})")
                bad_images.append(p)
                per_dir_bad[seq_dir] += 1

    print(f"\nScanned {total} images across {len(seq_dirs)} sequence dirs.")
    print(f"Found {len(bad_images)} unreadable images.\n")

    print("Per-directory bad image counts:")
    for seq_dir in seq_dirs:
        t = per_dir_total.get(seq_dir, 0)
        b = per_dir_bad.get(seq_dir, 0)
        if t == 0 and b == 0:
            continue
        print(f"  {seq_dir}: {b} bad / {t} total")


if __name__ == "__main__":
    main()

