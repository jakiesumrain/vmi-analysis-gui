#!/usr/bin/env python3
"""
Compute a dark frame from dark HDF5 files and save to disk.

Defaults come from centroiding/config.toml (same as the main analysis
script); every value can be overridden on the command line.

Two methods (--method), both honour --frames (0 = all frames):
  mean:   streaming running sum — low memory
  median: parallel subsample — higher memory (all selected frames in RAM)

Output: <output-dir>/dark_{method}.npy (float64) plus a small
dark_{method}_info.txt with basic statistics. Point the main analysis
script's dark_file setting at the .npy to skip recomputing the dark
frame every run.

Usage:
  uv run python centroiding/compute_dark.py --dark-dir /path/to/dark_h5
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import h5py
import numpy as np

DEFAULT_PATTERN = "RAW-*.h5"
DEFAULT_DATASET = "raw/data"
DEFAULT_METHOD = "median"
DEFAULT_FRAMES = 2000


def _config_defaults(config_path: str) -> dict:
    """Read dark-related defaults from config.toml (missing file is OK)."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    vmi = cfg.get("vmi_picasso", {})
    proc = vmi.get("processing", {})
    return {
        "dark_dir": vmi.get("dark_data_dir", ""),
        "pattern": vmi.get("h5_pattern", DEFAULT_PATTERN),
        "dataset": vmi.get("h5_dataset", DEFAULT_DATASET),
        "method": proc.get("dark_method", DEFAULT_METHOD),
        "frames": proc.get("dark_frame_count", DEFAULT_FRAMES),
    }


def _sum_h5_file(fpath: str, dataset: str):
    """Worker: return (sum_of_frames, n_frames) for one H5 file."""
    with h5py.File(fpath, "r") as hf:
        data = hf[dataset][:]
    return data.sum(axis=0, dtype=np.float64), data.shape[0]


def _read_h5_data(fpath: str, dataset: str):
    """Worker: return full data array from one H5 file."""
    with h5py.File(fpath, "r") as hf:
        return hf[dataset][:]


def compute_mean(dark_files, dataset: str, frame_count: int):
    """Compute mean dark frame (streaming, low memory).

    Uses the first frame_count frames across the selected files;
    frame_count <= 0 means all frames of all files.
    """
    if frame_count > 0:
        with h5py.File(dark_files[0], "r") as hf:
            frames_per_file = hf[dataset].shape[0]
        n_needed = int(np.ceil(frame_count / frames_per_file))
        dark_files = dark_files[:min(n_needed, len(dark_files))]
    n_files = len(dark_files)
    print(f"[Mean] Processing {n_files} files"
          + (f" (target {frame_count} frames)..." if frame_count > 0
             else " (all frames)..."))
    running_sum = None
    n_total = 0
    t0 = time.time()
    for idx, fpath in enumerate(dark_files, 1):
        try:
            with h5py.File(fpath, "r") as hf:
                data = hf[dataset][:]
            if 0 < frame_count - n_total < data.shape[0]:
                data = data[:frame_count - n_total]
            if running_sum is None:
                running_sum = data.sum(axis=0, dtype=np.float64)
            else:
                running_sum += data.sum(axis=0, dtype=np.float64)
            n_total += data.shape[0]
            if idx % 50 == 0 or (frame_count > 0 and n_total >= frame_count):
                elapsed = time.time() - t0
                print(f"  [{idx}/{n_files}] {n_total} frames, "
                      f"{n_total/elapsed:.0f} frames/s", flush=True)
            if frame_count > 0 and n_total >= frame_count:
                break
        except Exception as e:
            print(f"  WARNING: {os.path.basename(fpath)}: {e}")
    dark = running_sum / n_total
    print(f"[Mean] Done: {n_total} frames in {time.time()-t0:.1f}s")
    return dark, n_total


def compute_median(dark_files, dataset: str, frame_count: int):
    """Compute median of subsampled dark frames (parallel read).

    Uses the first frame_count frames; frame_count <= 0 means all
    frames of all files (memory-hungry).
    """
    if frame_count > 0:
        with h5py.File(dark_files[0], "r") as hf:
            frames_per_file = hf[dataset].shape[0]
        n_needed = int(np.ceil(frame_count / frames_per_file))
        selected = dark_files[:min(n_needed, len(dark_files))]
    else:
        selected = dark_files
    n_workers = min(len(selected), cpu_count())
    print(f"[Median] Reading {len(selected)} files ({n_workers} workers)...")
    t0 = time.time()
    all_frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_read_h5_data, f, dataset): f
                   for f in selected}
        for fut in as_completed(futures):
            try:
                all_frames.append(fut.result())
            except Exception as e:
                print(f"  WARNING: {e}")
    if not all_frames:
        raise RuntimeError("No dark frames could be read!")
    stack = np.concatenate(all_frames, axis=0)
    if frame_count > 0:
        stack = stack[:frame_count]
    dark = np.median(stack, axis=0).astype(np.float64)
    print(f"[Median] Done: {stack.shape[0]} frames in {time.time()-t0:.1f}s")
    return dark, stack.shape[0]


def main(argv=None) -> None:
    here = os.path.dirname(os.path.abspath(__file__))

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=os.path.join(here, "config.toml"))
    known, _ = pre.parse_known_args(argv)
    defaults = _config_defaults(known.config)

    p = argparse.ArgumentParser(
        description="Compute a mean/median dark frame from dark HDF5 files.")
    p.add_argument("--config", default=known.config,
                   help="config.toml for default values")
    p.add_argument("--dark-dir", default=defaults.get("dark_dir", ""),
                   help="directory containing the dark HDF5 files")
    p.add_argument("--pattern", default=defaults.get("pattern", DEFAULT_PATTERN),
                   help="glob pattern inside --dark-dir")
    p.add_argument("--dataset", default=defaults.get("dataset", DEFAULT_DATASET),
                   help="frame dataset inside each H5 file")
    p.add_argument("--method", choices=["mean", "median"],
                   default=defaults.get("method", DEFAULT_METHOD))
    p.add_argument("--frames", type=int,
                   default=defaults.get("frames", DEFAULT_FRAMES),
                   help="number of frames to use; 0 = all frames of all files")
    p.add_argument("--output-dir", default=here,
                   help="where dark_{method}.npy is written "
                        "(default: this script's directory)")
    args = p.parse_args(argv)

    if not args.dark_dir:
        print("[ERROR] no dark directory: pass --dark-dir or set "
              "dark_data_dir in config.toml")
        sys.exit(1)

    t0 = time.time()
    pattern = os.path.join(args.dark_dir, args.pattern)
    print(f"Searching: {pattern}")
    dark_files = sorted(glob.glob(pattern))
    if not dark_files:
        print(f"ERROR: No HDF5 files found in {args.dark_dir}")
        sys.exit(1)
    print(f"Found {len(dark_files)} dark HDF5 files (method={args.method})")

    if args.method == "median":
        dark, n_used = compute_median(dark_files, args.dataset, args.frames)
    else:
        dark, n_used = compute_mean(dark_files, args.dataset, args.frames)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"dark_{args.method}.npy")
    np.save(out_path, dark)
    print(f"Saved: {out_path}")
    print(f"  shape={dark.shape}  dtype={dark.dtype}  "
          f"mean={dark.mean():.1f}  std={dark.std():.1f}")

    info_path = os.path.join(args.output_dir, f"dark_{args.method}_info.txt")
    with open(info_path, "w") as f:
        f.write(f"method={args.method}\n")
        f.write(f"n_frames={n_used}\n")
        f.write(f"mean={dark.mean():.6f}\n")
        f.write(f"std={dark.std():.6f}\n")
        f.write(f"min={dark.min():.0f}\n")
        f.write(f"max={dark.max():.0f}\n")
        f.write(f"computed_seconds={time.time()-t0:.1f}\n")
    print(f"Saved: {info_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
