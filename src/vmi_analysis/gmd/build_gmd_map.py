#!/usr/bin/env python3
"""
Build frame-to-GMD correspondence map using timestamp matching.

Frame timestamp  = raw/meta col3 + col4 / 1e9
GMD timestamp    = current-time-both.txt column 2 (machine-B timestamp)

Output: gmd_map_first100.txt  (frame_idx, h5_file, frame_in_file, gmd_idx, gmd_val1, gmd_val2, time_diff_ms)
"""

import argparse
import h5py
import numpy as np
import glob
import os
import time

# --- default config (override via CLI) ---
DEFAULT_H5_DIR = ""                  # required: folder of HDF5 files
DEFAULT_H5_PATTERN = "RAW-*.h5"      # glob inside DEFAULT_H5_DIR
DEFAULT_GMD_PATH = ""                # required: current-time-both.txt
DEFAULT_OUTPUT_PATH = "gmd_map_all.txt"
DEFAULT_N_FILES = 0  # 0 = all files
# -----------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build frame-to-GMD correspondence map via timestamp matching."
    )
    parser.add_argument("--h5-dir", default=DEFAULT_H5_DIR,
                        help="Directory containing the HDF5 files.")
    parser.add_argument("--h5-pattern", default=DEFAULT_H5_PATTERN,
                        help="Glob used inside --h5-dir (default: RAW-*.h5). "
                             "Use *RAW-*.h5 for files with a prefix before RAW-.")
    parser.add_argument("--gmd-path", default=DEFAULT_GMD_PATH,
                        help="GMD timestamp file (current-time-both.txt).")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH,
                        help="Output map file path.")
    parser.add_argument("--n-files", type=int, default=DEFAULT_N_FILES,
                        help="Process only the first N files (0 = all).")
    args = parser.parse_args()
    if not args.h5_dir or not args.gmd_path:
        parser.error("--h5-dir and --gmd-path are required "
                     "(no dataset is bundled with this repository).")
    return args


def main(h5_dir, gmd_path, output_path, n_files, h5_pattern=DEFAULT_H5_PATTERN):
    t0 = time.time()

    # --- Load GMD data ---
    print("Loading GMD data...")
    gmd = np.loadtxt(gmd_path)
    gmd_timestamps = gmd[:, 1]      # machine-B timestamp
    gmd_vals_1 = gmd[:, 2]          # GMD method 1
    gmd_vals_2 = gmd[:, 3]          # GMD method 2
    print(f"  {len(gmd)} GMD entries loaded")
    print(f"  Time range: {gmd_timestamps[0]:.4f} -> {gmd_timestamps[-1]:.4f}")

    # --- Scan H5 files ---
    pattern = h5_pattern or DEFAULT_H5_PATTERN
    files = sorted(glob.glob(os.path.join(h5_dir, pattern)))
    if n_files > 0:
        files = files[:n_files]
    print(f"\nScanning {len(files)} HDF5 files matching {pattern!r}...")
    if not files:
        raise SystemExit(f"No HDF5 files matching {pattern!r} in {h5_dir}")

    # Collect (global_frame_idx, h5_basename, frame_in_file, frame_timestamp)
    frame_list = []  # list of [global_idx, h5_name, in_file_idx, timestamp]
    global_idx = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        with h5py.File(fpath, 'r') as hf:
            meta = hf['raw/meta'][:]
        for i in range(len(meta)):
            ft = meta[i, 3] + meta[i, 4] / 1e9
            frame_list.append([global_idx, fname, i, ft])
            global_idx += 1

    n_frames = len(frame_list)
    print(f"  {n_frames} frames total")
    if n_frames == 0:
        raise SystemExit("No frames found in the matched HDF5 files")

    # --- Match each frame to nearest GMD ---
    print("\nMatching frames to GMD entries...")
    frame_times = np.array([f[3] for f in frame_list])

    # For each frame time, find nearest GMD timestamp
    # Using searchsorted for speed (both arrays are sorted)
    gmd_timestamps = gmd_timestamps.astype(np.float64)
    frame_times = frame_times.astype(np.float64)

    insert_positions = np.searchsorted(gmd_timestamps, frame_times)
    # Clip to valid range
    insert_positions = np.clip(insert_positions, 0, len(gmd_timestamps) - 1)

    # Check left and right neighbor for each frame
    left_idx = np.maximum(insert_positions - 1, 0)
    right_idx = insert_positions

    left_diff = np.abs(gmd_timestamps[left_idx] - frame_times)
    right_diff = np.abs(gmd_timestamps[right_idx] - frame_times)

    nearest_idx = np.where(left_diff <= right_diff, left_idx, right_idx)
    time_diffs = frame_times - gmd_timestamps[nearest_idx]  # ms (positive = frame after GMD)

    # --- Write output ---
    print(f"\nWriting map to {output_path}")
    header = (
        "# frame_idx  h5_file              frame_in_file  gmd_idx  "
        "gmd_val1       gmd_val2       time_diff_ms\n"
        "#----------  -------------------  -------------  -------  "
        "--------------  -------------  -------------\n"
    )

    with open(output_path, 'w') as f:
        f.write(header)
        for fi in range(n_frames):
            gi, fname, fi_file, _ = frame_list[fi]
            gmd_i = nearest_idx[fi]
            line = (
                f"{gi:>10d}  {fname:<20s}  {fi_file:>13d}  "
                f"{gmd_i:>7d}  {gmd_vals_1[gmd_i]:.6e}  "
                f"{gmd_vals_2[gmd_i]:.6e}  {time_diffs[fi]*1000:+.3f}\n"
            )
            f.write(line)

    # --- Summary stats ---
    dt_ms = time_diffs * 1000
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Frames: {n_frames}")
    print(f"Unique GMD entries matched: {len(np.unique(nearest_idx))}")
    print(f"Time difference stats:")
    print(f"  mean = {dt_ms.mean():+.2f} ms")
    print(f"  std  = {dt_ms.std():.2f} ms")
    print(f"  min  = {dt_ms.min():+.2f} ms")
    print(f"  max  = {dt_ms.max():+.2f} ms")
    print(f"  |dT| < 20 ms: {(np.abs(dt_ms) < 20).mean()*100:.1f}% of frames")
    print(f"  |dT| < 40 ms: {(np.abs(dt_ms) < 40).mean()*100:.1f}% of frames")

    # Check sequential consistency
    gmd_indices = nearest_idx
    diffs = np.diff(gmd_indices)
    print(f"\nSequential check:")
    print(f"  delta = +1: {(diffs == 1).mean()*100:.1f}% of transitions")
    print(f"  delta =  0: {(diffs == 0).mean()*100:.1f}%")
    print(f"  delta >  1: {(diffs > 1).mean()*100:.1f}%")
    print(f"  delta <  0: {(diffs < 0).mean()*100:.1f}% (backwards!)")


if __name__ == "__main__":
    args = parse_args()
    main(args.h5_dir, args.gmd_path, args.output_path, args.n_files,
         args.h5_pattern)
