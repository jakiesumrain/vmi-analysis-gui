#!/usr/bin/env python3
"""
Accumulate camera frames with threshold, morphological denoising and optional GMD scaling.

GMD scaling: each frame's contribution is weighted by mean_gmd / gmd_of_that_frame,
so frames from weaker FEL pulses get proportionally higher weight.

This uses the pre-computed gmd_map_all.txt (frame_idx → GMD value).
"""

import argparse
import h5py
import numpy as np
from numpy.lib.npyio import DataSource
import cv2
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- default config (override via CLI) ---
h5_dir       = ""    # required: folder containing the RAW-*.h5 files
output_path  = "accumulated_result.txt"
im_threshold = 5
do_denoise   = True
kernel       = np.ones((2, 2), dtype=np.uint8)
max_workers  = 6
frame_limit  = 0

# Simple sum mode: sum all frames as-is — no threshold, no denoising.
# Dark subtraction and GMD weighting still apply if configured below.
simple_sum   = False

# GMD scaling (set to False to skip)
gmd_map_path = ""
enable_gmd   = False

# GMD weight range: only include frames with weight in [gmd_w_min, gmd_w_max]
# Frames outside this range are discarded (weight = 0).
# Set both to 0 to disable filtering (use all frames).
gmd_w_min = 0.0
gmd_w_max = 4.0

# Dark frame (.npy file from centroiding/compute_dark.py)
# Set to empty string to skip dark subtraction
dark_path    = r""

# Headless / live-preview options (for web/GUI use)
no_show      = False   # skip the interactive plot window
live_path    = r""     # optional .npy path for partial images; empty to skip
live_every   = 25      # write the partial image every N completed files
# ------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Accumulate camera frames with threshold, denoising and optional GMD scaling.")
    p.add_argument("--h5-dir", default=h5_dir)
    p.add_argument("--output-path", default=output_path)
    p.add_argument("--im-threshold", type=float, default=im_threshold)
    p.add_argument("--do-denoise", action=argparse.BooleanOptionalAction,
                   default=do_denoise)
    p.add_argument("--max-workers", type=int, default=max_workers)
    p.add_argument("--frame-limit", type=int, default=frame_limit,
                   help="Process the first N frames in file order (0 = all).")
    p.add_argument("--simple-sum", action="store_true", default=simple_sum)
    p.add_argument("--gmd-map-path", default=gmd_map_path)
    p.add_argument("--enable-gmd", action="store_true", default=enable_gmd)
    p.add_argument("--gmd-w-min", type=float, default=gmd_w_min)
    p.add_argument("--gmd-w-max", type=float, default=gmd_w_max)
    p.add_argument("--dark-path", default=dark_path)
    p.add_argument("--no-show", action="store_true", default=no_show)
    p.add_argument("--live-path", default=live_path,
                   help="Write the partial accumulated image (.npy) here during the run.")
    p.add_argument("--live-every", type=int, default=live_every)
    args = p.parse_args()
    if not args.h5_dir:
        p.error("--h5-dir is required "
                "(no dataset is bundled with this repository).")
    if args.frame_limit < 0:
        p.error("--frame-limit must be zero or a positive integer.")
    if args.enable_gmd and not args.gmd_map_path:
        p.error("--enable-gmd requires --gmd-map-path.")
    return args


# worker-process globals — set once via initializer, never pickled per-task
_dark_frame  = None
_gmd_weights = None
_params      = None

def _worker_init(dark_frame, gmd_weights, params):
    global _dark_frame, _gmd_weights, _params
    _dark_frame  = dark_frame
    _gmd_weights = gmd_weights
    _params      = params


def load_gmd_weights(path, n_total_frames):
    """Load GMD map and return weight per global frame index.

    weight = mean_gmd / gmd_value  (linear normalization)
    Frames not in the map get weight = 1.0.
    """
    print(f"Loading GMD map: {path}")
    map_data = np.loadtxt(path, comments='#', usecols=(0, 4))
    frame_idx = map_data[:, 0].astype(int)
    gmd_vals = map_data[:, 1]

    mean_gmd = gmd_vals.mean()
    # Clamp near-zero GMD values (measurement noise floor ~ ±1e-11)
    # 57 frames in method 1, 626 in method 2 have slightly negative values
    gmd_safe = np.maximum(gmd_vals, 1e-12)
    weights = np.ones(n_total_frames, dtype=np.float64)
    weights[frame_idx] = mean_gmd / gmd_safe
    # Filter by weight range: zero out frames outside [gmd_w_min, gmd_w_max]
    if gmd_w_min > 0 or gmd_w_max > 0:
        if gmd_w_max > 0:
            weights[weights > gmd_w_max] = 0.0
        if gmd_w_min > 0:
            weights[weights < gmd_w_min] = 0.0
        n_discarded = int((weights == 0).sum())
        n_kept = n_total_frames - n_discarded
        print(f"  Weight filter [{gmd_w_min}, {gmd_w_max}]: "
              f"kept {n_kept}/{n_total_frames} frames ({n_discarded} discarded)")
    n_nan = np.isnan(weights).sum()
    if n_nan:
        weights[np.isnan(weights)] = 1.0
    print(f"  {len(frame_idx)} frames mapped, mean_gmd={mean_gmd:.3e}, "
          f"weight range [{weights[frame_idx].min():.3f}, {weights[frame_idx].max():.3f}]")
    return weights


def process_h5(args):
    """Accumulate a prefix of one HDF5 file with the configured corrections."""
    fpath, base_frame_idx, stop = args

    with h5py.File(fpath, 'r') as hf:
        data = hf['raw/data'][:stop]               # (N, H, W) uint16

    n = data.shape[0]
    if n == 0:
        return np.zeros(data.shape[1:], dtype=np.float64), 0

    # --- dark subtraction + ensure float64 for in-place ops ---
    if _dark_frame is not None:
        data = data.astype(np.float64) - _dark_frame
    else:
        data = data.astype(np.float64)

    # --- threshold + denoise (skipped in simple_sum mode) ---
    if not _params["simple_sum"]:
        # threshold (vectorized) — works on float64, negatives are < 1
        mask = data >= _params["im_threshold"]           # (N, H, W) bool

        # morphological denoise
        if _params["do_denoise"]:
            mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
            for i in range(n):
                mask_u8[i] = cv2.morphologyEx(mask_u8[i], cv2.MORPH_OPEN,
                                              _params["kernel"], iterations=1)
            mask = mask_u8.astype(bool)

        # zero below-threshold pixels in place (no large temporary array)
        data[~mask] = 0.0

    if _gmd_weights is not None:
        w = _gmd_weights[base_frame_idx:base_frame_idx + n].reshape(-1, 1, 1)
        data *= w  # in-place multiply, no extra full-array allocation
    acc = data.sum(axis=0, dtype=np.float64)
    return acc, n


def plot_accumulated(total_acc, title_suffix="", show=True):
    """Plot the accumulated image with a log color scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(7, 7))
    vmax = np.percentile(total_acc, 99.999999)  # total_acc.max()
    #plt.imshow(total_acc, origin='lower',
     #          norm=LogNorm(vmin=vmax * 0.1, vmax=vmax))
    plt.pcolormesh(total_acc, cmap='viridis',
                           norm=LogNorm(vmin=vmax * 0.01, vmax=vmax),
                           shading='auto')
    plt.axis('equal')
    plt.colorbar(label='weighted counts (log)')
    plt.title(f'Accumulated{title_suffix} (log scale)')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        out = os.path.splitext(output_path)[0] + ".png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {out}")
    plt.close()


def main():
    # Skip straight to plotting if the accumulated result already exists
    if os.path.exists(output_path):
        with DataSource().open(output_path, mode="rt", encoding="utf-8") as saved:
            first_line = saved.readline()
        prefix = "# frame_limit="
        saved_limit = int(first_line[len(prefix):]) if first_line.startswith(prefix) else 0
        if saved_limit != frame_limit:
            raise SystemExit(f"[Error] {output_path} was created with frame limit {saved_limit}. "
                             "Choose a different output path or clear the existing result.")
        print(f"[Resume] {output_path} exists — skipping accumulation, plotting directly")
        total_acc = np.loadtxt(output_path)
        plot_accumulated(total_acc, " (loaded from file)", show=not no_show)
        return

    files = sorted([
        os.path.join(h5_dir, f)
        for f in os.listdir(h5_dir) if f.endswith('.h5')
    ])
    # Keep the original acquisition offsets for GMD weighting.
    batch_args = []
    n_total = 0
    for f in files:
        with h5py.File(f, 'r') as hf:
            count = hf['raw/data'].shape[0]
        if not frame_limit or n_total < frame_limit:
            stop = min(count, frame_limit - n_total) if frame_limit else None
            if stop != 0:
                batch_args.append((f, n_total, stop))
        n_total += count
    n_files = len(batch_args)
    workers = min(max_workers, n_files) if n_files else 1
    print(f"Found {len(files)} h5 files, processing {n_files} with {workers} workers (processes)...")
    if frame_limit:
        print(f"Processing the first {min(frame_limit, n_total)} of {n_total} frames "
              f"(requested {frame_limit})")
    if simple_sum:
        print("Simple sum mode: ON (no threshold, no denoising)")

    # Load dark frame
    dark_frame = np.load(dark_path) if dark_path and os.path.exists(dark_path) else None
    if dark_frame is not None:
        print(f"Dark frame loaded: mean={dark_frame.mean():.1f}, std={dark_frame.std():.1f}")
    else:
        print(f"Dark subtraction DISABLED (dark_path={dark_path})")

    # Load GMD weights once
    gmd_weights = load_gmd_weights(gmd_map_path, n_total) if enable_gmd else None
    if gmd_weights is not None:
        print(f"GMD weighting ENABLED")
    else:
        print(f"GMD weighting DISABLED")

    t_start = time.time()
    total_frames = 0
    total_acc = None
    n_done = 0

    params = {
        "im_threshold": im_threshold,
        "do_denoise": do_denoise,
        "kernel": kernel,
        "simple_sum": simple_sum,
    }

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init,
                             initargs=(dark_frame, gmd_weights, params)) as executor:
        futures = {executor.submit(process_h5, args): args[0] for args in batch_args}
        with tqdm(total=n_files, desc="Processing H5 files") as pbar:
            for fut in as_completed(futures):
                try:
                    acc, n_frames = fut.result()
                    total_frames += n_frames
                    if total_acc is None:
                        total_acc = acc
                    else:
                        total_acc += acc
                except Exception as e:
                    if frame_limit:
                        raise
                    print(f"\n[Error] {e}")
                n_done += 1
                if live_path and total_acc is not None and n_done % live_every == 0:
                    np.save(live_path, total_acc)
                pbar.update(1)

    t_end = time.time()
    if total_acc is None:
        print("No files processed successfully; nothing to save.")
        return
    print(f"\n{total_frames} frames processed in {t_end - t_start:.1f}s")
    print(f"Sum of all counts: {np.sum(total_acc):.0f}")

    if frame_limit:
        # Publish the limited result only after the complete matrix is written.
        output = Path(output_path)
        with NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            np.savetxt(temporary, total_acc, header=f"frame_limit={frame_limit}")
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        np.savetxt(output_path, total_acc)
    print(f"Saved to {output_path}")

    gmd_label = "ON" if gmd_weights is not None else "OFF"
    plot_accumulated(total_acc, f" ({total_frames} frames, GMD={gmd_label})",
                     show=not no_show)


if __name__ == '__main__':
    _args = parse_args()
    h5_dir = _args.h5_dir
    output_path = _args.output_path
    im_threshold = _args.im_threshold
    do_denoise = _args.do_denoise
    max_workers = _args.max_workers
    frame_limit = _args.frame_limit
    simple_sum = _args.simple_sum
    gmd_map_path = _args.gmd_map_path
    enable_gmd = _args.enable_gmd
    gmd_w_min = _args.gmd_w_min
    gmd_w_max = _args.gmd_w_max
    dark_path = _args.dark_path
    no_show = _args.no_show
    live_path = _args.live_path
    live_every = _args.live_every
    main()
