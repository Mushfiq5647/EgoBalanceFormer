"""
Temporal alignment diagnostic for a single sequence directory.

Checks whether the pose windows from ground_truth.json are correctly
aligned with the CoP samples from balance.json using timestamps.

Usage:
    python analysis/check_temporal_alignment.py \
        /data/My_Backup/Dataset/gemini-data/10-amanda-trial-1-cl
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def parse_ts(ts: str) -> float:
    """Parse 'YYYY-MM-DD-HH-MM-SS-mmm' → seconds as float."""
    parts = ts.split("-")
    # format: 2025-07-30-10-54-45-130  → 7 parts
    if len(parts) == 7:
        dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                      int(parts[3]), int(parts[4]), int(parts[5]))
        return dt.timestamp() + int(parts[6]) / 1000.0
    raise ValueError(f"Unexpected timestamp format: {ts!r}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_temporal_alignment.py <seq_dir>")
        sys.exit(1)

    seq_dir = Path(sys.argv[1])
    gt_path  = seq_dir / "ground_truth.json"
    bal_path = seq_dir / "balance.json"
    meta_paths = list(seq_dir.glob("metadata_*.txt"))

    print("=" * 70)
    print(f"TEMPORAL ALIGNMENT DIAGNOSTIC: {seq_dir.name}")
    print("=" * 70)

    # ── ground_truth.json ──────────────────────────────────────────────────
    with open(gt_path) as f:
        gt = json.load(f)

    frames = [(e["image_name"], int(e["image_name"].split("_")[-1]))
              for e in gt if "image_name" in e]
    frames.sort(key=lambda x: x[1])
    frame_ids = [f[1] for f in frames]

    print(f"\nground_truth.json:")
    print(f"  Total frames : {len(frames)}")
    print(f"  Frame range  : {frame_ids[0]} .. {frame_ids[-1]}")
    print(f"  At 22 fps    : {len(frames)/22:.2f} s")
    print(f"  11-frame windows: {len(frames) // 11}")

    # ── balance.json ───────────────────────────────────────────────────────
    with open(bal_path) as f:
        bal = json.load(f)

    print(f"\nbalance.json:")
    print(f"  Total CoP samples: {len(bal)}")
    print(f"  At 2 fps          : {len(bal)/2:.2f} s")
    print(f"  First timestamp   : {bal[0].get('timestamp', 'N/A')}")
    print(f"  Last  timestamp   : {bal[-1].get('timestamp', 'N/A')}")

    # ── metadata file ──────────────────────────────────────────────────────
    if meta_paths:
        import csv
        meta_file = meta_paths[0]
        ts_map = {}
        with open(meta_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_map[int(row["FrameIndex"])] = row["ColorTimestamp"]

        print(f"\nmetadata file: {meta_file.name}")
        print(f"  Total frame timestamps: {len(ts_map)}")

        # Timestamp of first and last ground_truth frame
        first_gt_ts = ts_map.get(frame_ids[0], "NOT FOUND")
        last_gt_ts  = ts_map.get(frame_ids[-1], "NOT FOUND")
        print(f"  First GT frame ({frame_ids[0]}) timestamp: {first_gt_ts}")
        print(f"  Last  GT frame ({frame_ids[-1]}) timestamp: {last_gt_ts}")

        # Compare with CoP timestamps
        cop_ts_first = bal[0].get("timestamp", "N/A")
        cop_ts_last  = bal[-1].get("timestamp", "N/A")

        print(f"\nTimestamp comparison:")
        print(f"  GT  starts: {first_gt_ts}")
        print(f"  CoP starts: {cop_ts_first}")
        print(f"  GT  ends  : {last_gt_ts}")
        print(f"  CoP ends  : {cop_ts_last}")

        try:
            t_gt_start  = parse_ts(first_gt_ts)
            t_gt_end    = parse_ts(last_gt_ts)
            t_cop_start = parse_ts(cop_ts_first)
            t_cop_end   = parse_ts(cop_ts_last)

            offset_start = t_gt_start - t_cop_start
            offset_end   = t_gt_end   - t_cop_end

            print(f"\n  Offset at start : GT - CoP = {offset_start:+.3f} s")
            print(f"  Offset at end   : GT - CoP = {offset_end:+.3f} s")

            if abs(offset_start) < 0.1:
                print("\n  ✓ Good alignment at start (< 100ms offset)")
            elif abs(offset_start) < 0.5:
                print(f"\n  ⚠ Moderate offset at start ({offset_start*1000:.0f} ms) "
                      f"≈ {abs(offset_start) * 22:.1f} pose frames")
            else:
                print(f"\n  ✗ Large offset at start ({offset_start:.3f} s) "
                      f"≈ {abs(offset_start) * 22:.1f} pose frames — check alignment!")

            # How many pose frames correspond to the CoP start offset
            frames_offset = int(round(offset_start * 22))
            print(f"\n  Suggested pose start frame offset: {frames_offset} frames")
            print(f"  (i.e., dataset should skip first {frames_offset} frames "
                  f"before aligning with CoP)")

        except ValueError as e:
            print(f"\n  Could not parse timestamps: {e}")

    else:
        print("\n  No metadata file found — cannot compare timestamps.")

    # ── Window vs CoP count ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WINDOW / CoP ALIGNMENT SUMMARY")
    print("=" * 70)

    n_pose_windows = len(frames) // 11
    n_cop = len(bal)
    n_usable = min(n_pose_windows, n_cop)

    print(f"\n  Pose 11-frame windows : {n_pose_windows}")
    print(f"  CoP samples           : {n_cop}")
    print(f"  Usable (min)          : {n_usable}")
    print(f"  Discarded pose windows: {n_pose_windows - n_usable}")
    print(f"  Discarded CoP samples : {n_cop - n_usable}")

    if n_pose_windows != n_cop:
        diff = n_pose_windows - n_cop
        print(f"\n  ⚠ Count mismatch: {abs(diff)} extra "
              f"{'pose windows' if diff > 0 else 'CoP samples'}.")
        print(f"    This means {abs(diff) / 2:.1f} seconds of data is unaligned.")
    else:
        print(f"\n  ✓ Counts match exactly — no trimming needed.")


if __name__ == "__main__":
    main()
