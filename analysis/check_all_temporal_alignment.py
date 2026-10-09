"""
Check temporal alignment for all directories in train.txt.
"""

import json
import csv
from datetime import datetime
from pathlib import Path


def parse_ts(ts: str) -> float:
    """Parse 'YYYY-MM-DD-HH-MM-SS-mmm' → seconds as float."""
    parts = ts.split("-")
    if len(parts) == 7:
        dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                      int(parts[3]), int(parts[4]), int(parts[5]))
        return dt.timestamp() + int(parts[6]) / 1000.0
    raise ValueError(f"Unexpected timestamp format: {ts!r}")


def check_sequence(seq_dir: Path):
    """
    Check temporal alignment for a single sequence.
    Returns dict with alignment info or None if error.
    """
    gt_path  = seq_dir / "ground_truth.json"
    bal_path = seq_dir / "balance.json"
    
    if not gt_path.exists() or not bal_path.exists():
        return None
    
    try:
        # Load ground_truth and use timestamps directly from JSON
        with open(gt_path) as f:
            gt = json.load(f)
        frames = [e for e in gt if isinstance(e, dict) and "image_name" in e]
        n_pose_frames = len(frames)
        n_pose_windows = n_pose_frames // 11

        # Use the first frame that has a valid 'timestamp' key
        first_gt_ts = None
        for e in frames:
            ts = e.get("timestamp")
            if isinstance(ts, str):
                first_gt_ts = ts
                break

        # Load balance
        with open(bal_path) as f:
            bal = json.load(f)
        n_cop_samples = len(bal)
        cop_ts_first = bal[0].get("timestamp") if n_cop_samples > 0 else None

        if not (first_gt_ts and cop_ts_first):
            return {
                "subject": seq_dir.name,
                "n_pose_frames": n_pose_frames,
                "n_pose_windows": n_pose_windows,
                "n_cop_samples": n_cop_samples,
                "has_metadata": False,
                "offset_start_s": None,
                "offset_frames": None,
            }

        t_gt_start  = parse_ts(first_gt_ts)
        t_cop_start = parse_ts(cop_ts_first)
        offset_start = t_gt_start - t_cop_start
        frames_offset = int(round(offset_start * 22))
        
        return {
            "subject": seq_dir.name,
            "n_pose_frames": n_pose_frames,
            "n_pose_windows": n_pose_windows,
            "n_cop_samples": n_cop_samples,
            "has_metadata": True,
            "offset_start_s": offset_start,
            "offset_frames": frames_offset,
            "first_frame_id": 0,
        }
        
    except Exception as e:
        return None


def main():
    train_txt = Path("pose_estimation/utils/train.txt")
    
    with open(train_txt) as f:
        directories = [line.strip() for line in f if line.strip()]
    
    print("=" * 80)
    print(f"TEMPORAL ALIGNMENT CHECK FOR ALL {len(directories)} SEQUENCES")
    print("=" * 80)
    
    results = []
    failed = 0
    
    for i, dir_str in enumerate(directories, 1):
        seq_dir = Path(dir_str)
        result = check_sequence(seq_dir)
        
        if result is None:
            failed += 1
            continue
        
        results.append(result)
        
        if i % 10 == 0 or i == len(directories):
            print(f"Processed {i}/{len(directories)} sequences... "
                  f"(✓ {len(results)}, ✗ {failed})")
    
    if not results:
        print("\n✗ No valid sequences found!")
        return
    
    # Analyze results
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    with_metadata = [r for r in results if r["has_metadata"]]
    offsets = [r["offset_start_s"] for r in with_metadata]
    frame_offsets = [r["offset_frames"] for r in with_metadata]
    
    print(f"\nSequences with metadata: {len(with_metadata)}/{len(results)}")
    
    if with_metadata:
        print(f"\nTemporal offset at start (GT - CoP):")
        print(f"  Mean:   {sum(offsets)/len(offsets):+.3f} s  "
              f"(≈ {sum(frame_offsets)/len(frame_offsets):.1f} frames @ 22fps)")
        print(f"  Median: {sorted(offsets)[len(offsets)//2]:+.3f} s")
        print(f"  Min:    {min(offsets):+.3f} s  ({min(frame_offsets):+d} frames)")
        print(f"  Max:    {max(offsets):+.3f} s  ({max(frame_offsets):+d} frames)")
        print(f"  Std:    {(sum((o - sum(offsets)/len(offsets))**2 for o in offsets)/len(offsets))**0.5:.3f} s")
    
    # Count mismatch analysis
    mismatches = [(r["n_pose_windows"] - r["n_cop_samples"]) for r in results]
    print(f"\nWindow/CoP count mismatch:")
    print(f"  Mean:   {sum(mismatches)/len(mismatches):+.1f} extra pose windows")
    print(f"  Median: {sorted(mismatches)[len(mismatches)//2]:+d}")
    print(f"  Min:    {min(mismatches):+d}")
    print(f"  Max:    {max(mismatches):+d}")
    
    # Detailed breakdown
    print("\n" + "=" * 80)
    print("PER-SEQUENCE DETAILS (first 10 and problematic ones)")
    print("=" * 80)
    print(f"\n{'Subject':<40} {'Frames':>7} {'Windows':>8} {'CoP':>5} {'Offset(s)':>10} {'OffsetFr':>9}")
    print("-" * 80)
    
    for r in results[:10]:
        offset_str = f"{r['offset_start_s']:+.3f}" if r['has_metadata'] else "N/A"
        offset_fr  = f"{r['offset_frames']:+d}" if r['has_metadata'] else "N/A"
        print(f"{r['subject']:<40} {r['n_pose_frames']:7d} {r['n_pose_windows']:8d} "
              f"{r['n_cop_samples']:5d} {offset_str:>10} {offset_fr:>9}")
    
    # Show problematic ones (large offset or large mismatch)
    problematic = [r for r in results if 
                   (r['has_metadata'] and abs(r['offset_start_s']) > 1.0) or
                   abs(r['n_pose_windows'] - r['n_cop_samples']) > 20]
    
    if problematic and len(problematic) < len(results):
        print("\n... (showing problematic sequences) ...")
        for r in problematic[:10]:
            offset_str = f"{r['offset_start_s']:+.3f}" if r['has_metadata'] else "N/A"
            offset_fr  = f"{r['offset_frames']:+d}" if r['has_metadata'] else "N/A"
            print(f"{r['subject']:<40} {r['n_pose_frames']:7d} {r['n_pose_windows']:8d} "
                  f"{r['n_cop_samples']:5d} {offset_str:>10} {offset_fr:>9}")
    
    # Save detailed CSV
    output_csv = Path("balance_prediction/temporal_alignment_report.csv")
    with open(output_csv, "w", newline="") as f:
        import csv as csvlib
        writer = csvlib.writer(f)
        writer.writerow(["subject", "n_pose_frames", "n_pose_windows", "n_cop_samples",
                         "window_cop_diff", "has_metadata", "offset_start_s", "offset_frames",
                         "first_frame_id"])
        for r in results:
            writer.writerow([
                r["subject"],
                r["n_pose_frames"],
                r["n_pose_windows"],
                r["n_cop_samples"],
                r["n_pose_windows"] - r["n_cop_samples"],
                r["has_metadata"],
                r.get("offset_start_s", ""),
                r.get("offset_frames", ""),
                r.get("first_frame_id", ""),
            ])
    
    print(f"\n✓ Detailed report saved to: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
