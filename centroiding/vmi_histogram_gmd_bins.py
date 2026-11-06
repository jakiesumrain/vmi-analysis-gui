#!/usr/bin/env python3
"""
Build 2D VMI histograms of electron events partitioned into N categories
by user-specified GMD value ranges.

Each event in the checkpoint has a global `frame` index. The GMD map
(gmd_map_all_*.txt) provides one GMD value per frame. Events are sorted
into bins defined by GMD threshold values, and a separate 2D histogram
is accumulated for each bin.

GMD bin thresholds are defined by the user below (GMD_THRESHOLDS).
Example: thresholds = [3e-8, 5e-8, 7e-8]  ->  4 bins:
    bin 0: GMD < 3e-8
    bin 1: 3e-8 <= GMD < 5e-8
    bin 2: 5e-8 <= GMD < 7e-8
    bin 3: GMD >= 7e-8

Usage:
    uv run python centroiding/vmi_histogram_gmd_bins.py
"""

from __future__ import annotations
import argparse
import os
import sys
import tomllib
import numpy as np
import pandas as pd
import matplotlib
# Headless runs set MPLBACKEND (e.g. Agg); default stays the interactive TkAgg.
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

pd.options.mode.chained_assignment = None

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION (defaults — override via CLI)
# ─────────────────────────────────────────────────────────────────────────────

# GMD thresholds (3 values → 4 bins). Adjust to your needs.
GMD_THRESHOLDS = [3.75e-8, 5.5e-8, 7.25e-8]

# GMD column in the map file: 4 = gmd_val1 (method 1), 5 = gmd_val2 (method 2)
GMD_COL = 4

# Weight each event by its frame's GMD when accumulating the histogram.
#   "none"     : every event contributes 1        (default)
#   "gmd"      : weight = gmd                     (raw GMD scaling)
#   "gmd_norm" : weight = mean_gmd / gmd          (linear pulse-energy normalisation)
#
# NOTE: this is a per-frame scalar. It rescales how much each frame contributes,
# so it corrects the electron YIELD between bins, but it does NOT change the
# spectral SHAPE within a bin. Measured effect on the radial profile shape is
# <0.5% (see gmd_weight_test.py), so it will not make the four bins overlap.
WEIGHT_MODE = "gmd_norm"

# Exponent for "gmd_norm": weight = (mean_gmd / gmd) ** WEIGHT_POWER.
#
# WEIGHT_POWER = 1.0 is the linear pulse-energy normalisation assumed by
# config.toml, but that OVERCORRECTS for this dataset: dividing by GMD inverts
# the yield ordering (measured yield/frame 1.000/1.213/1.325/1.421 becomes
# 1.000/0.795/0.653/0.554) because the true scaling is strongly sub-linear.
# Fitting log(yield)/log(GMD) on the four bins gives an effective exponent of
# ~0.42, so 0.42 is the value that actually flattens the yield.
WEIGHT_POWER = 1.0

# Floor for non-positive GMD values (same value the V5 pipeline uses).
GMD_FLOOR = 1e-12

NO_SHOW = False  # skip the interactive plot window (for web/GUI use)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build 2D VMI histograms partitioned into GMD bins.")
    p.add_argument("--config", help="Optional legacy TOML defaults; CLI values take precedence")
    p.add_argument("--checkpoint-path", help="Input localization checkpoint HDF5")
    p.add_argument("--output-dir", help="Histogram output directory")
    p.add_argument("--gmd-map-path", help="Frame-to-GMD map file")
    p.add_argument("--resolution-factor", type=int)
    p.add_argument("--gmd-weight-min", type=float)
    p.add_argument("--gmd-weight-max", type=float)
    p.add_argument("--thresholds", default=",".join(f"{t:g}" for t in GMD_THRESHOLDS),
                   help="Comma-separated GMD thresholds (N values -> N+1 bins).")
    p.add_argument("--gmd-col", type=int, default=GMD_COL,
                   help="GMD column in the map file: 4 = method 1, 5 = method 2.")
    p.add_argument("--weight-mode", default=WEIGHT_MODE,
                   choices=["none", "gmd", "gmd_norm"])
    p.add_argument("--weight-power", type=float, default=WEIGHT_POWER)
    p.add_argument("--gmd-floor", type=float, default=GMD_FLOOR)
    p.add_argument("--no-show", action="store_true", default=NO_SHOW)
    args = p.parse_args()
    vmi = {}
    if args.config:
        try:
            with open(args.config, "rb") as handle:
                vmi = tomllib.load(handle)["vmi_picasso"]
        except (OSError, ValueError, KeyError) as exc:
            p.error(str(exc))
    if args.output_dir is None:
        args.output_dir = vmi.get("output_dir", "vmi_output")
    if args.checkpoint_path is None and vmi:
        args.checkpoint_path = os.path.join(args.output_dir, vmi.get("checkpoint_file", "vmi_results_raw.h5"))
    if args.gmd_map_path is None:
        args.gmd_map_path = vmi.get("gmd_map_path", "")
    if args.resolution_factor is None:
        args.resolution_factor = vmi.get("histogram", {}).get("resolution_factor", 1)
    for key in ("gmd_weight_min", "gmd_weight_max"):
        if getattr(args, key) is None:
            setattr(args, key, vmi.get("gmd", {}).get(key, 0.0))
    if not args.checkpoint_path or not args.gmd_map_path:
        p.error("Provide --checkpoint-path and --gmd-map-path (or --config with those inputs)")
    if args.resolution_factor < 1:
        p.error("--resolution-factor must be positive")
    return args


CLI = parse_args()
GMD_THRESHOLDS = [float(x) for x in CLI.thresholds.split(",")]
GMD_COL = CLI.gmd_col
WEIGHT_MODE = CLI.weight_mode
WEIGHT_POWER = CLI.weight_power
GMD_FLOOR = CLI.gmd_floor
NO_SHOW = CLI.no_show

# ─────────────────────────────────────────────────────────────────────────────
# RESOLVED INPUTS (optional TOML plus explicit CLI overrides)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = CLI.output_dir
CHECKPOINT_PATH = CLI.checkpoint_path
GMD_MAP_PATH = CLI.gmd_map_path
RES_FACTOR = CLI.resolution_factor

# GMD weight clamp — mirrors [vmi_picasso.gmd] in config.toml, the same guard
# the main V5 pipeline applies. WITHOUT this, 'gmd_norm' is dominated by the
# handful of frames whose GMD is near zero (weight -> mean/0 -> huge): in bin0
# the top 10 frames then carry ~13% of the total weight and the spectrum is
# destroyed by noise. Out-of-range weights are ZEROED (matching V5).
GMD_WEIGHT_MIN = CLI.gmd_weight_min
GMD_WEIGHT_MAX = CLI.gmd_weight_max

N_BINS = len(GMD_THRESHOLDS) + 1


def load_gmd_per_frame(map_path, col):
    """Load GMD map → dense array of one GMD value per frame index."""
    print(f"[GMD] Loading map: {map_path}")
    map_data = np.loadtxt(map_path, comments="#", usecols=(0, col))
    if map_data.ndim == 1:
        map_data = map_data[np.newaxis, :]

    frame_idx = map_data[:, 0].astype(int)
    gmd_vals = map_data[:, 1]

    # Clamp non-positive GMD. Mirrors the V5 pipeline's load_gmd_map(): GMD
    # measurement noise and dropouts produce negative or zero values, which
    # would give a NEGATIVE (or infinite) weight and corrupt the histogram.
    n_bad = int((gmd_vals <= 0).sum())
    if n_bad:
        print(f"[GMD] Clamping {n_bad} non-positive GMD values to {GMD_FLOOR:.0e}")
        gmd_vals = np.maximum(gmd_vals, GMD_FLOOR)

    n_total = int(frame_idx.max()) + 1
    gmd_arr = np.full(n_total, np.nan)
    gmd_arr[frame_idx] = gmd_vals
    print(f"[GMD] Frames in map: {n_total}")
    print(f"[GMD] GMD range: [{np.nanmin(gmd_arr):.3e}, {np.nanmax(gmd_arr):.3e}]")
    return gmd_arr


def bin_index(gmd_vals, thresholds):
    """Return bin index for each GMD value using the thresholds."""
    idx = np.searchsorted(thresholds, gmd_vals)  # 0..len(thresholds)
    return idx


def main():
    print("=" * 80)
    print("VMI HISTOGRAM BY GMD BINS")
    print("=" * 80)
    print(f"[Config] Checkpoint:  {CHECKPOINT_PATH}")
    print(f"[Config] GMD map:     {GMD_MAP_PATH}")
    print(f"[Config] Output dir:  {OUTPUT_DIR}")
    print(f"[Config] GMD thresholds: {GMD_THRESHOLDS}")
    print(f"[Config] Number of bins:  {N_BINS}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    if not GMD_MAP_PATH or not os.path.exists(GMD_MAP_PATH):
        print(f"[ERROR] GMD map not found: {GMD_MAP_PATH}")
        sys.exit(1)

    # ── 1. Load GMD per frame ─────────────────────────────────────────────
    gmd_arr = load_gmd_per_frame(GMD_MAP_PATH, GMD_COL)
    thresholds = np.array(GMD_THRESHOLDS)
    gmd_bin = bin_index(gmd_arr, thresholds)
    gmd_mean = float(np.nanmean(gmd_arr))
    print(f"[GMD] Mean GMD (for 'gmd_norm' weighting): {gmd_mean:.3e}")
    print(f"[GMD] Weight mode: {WEIGHT_MODE}")

    # ── 2. Histogram size (fixed camera frame) ───────────────────────────
    # Camera frame is 1024×1024 pixels. Resolution factor upscales.
    img_size = (1024, 1024)  # (height, width)
    hist_ny = img_size[0] * RES_FACTOR
    hist_nx = img_size[1] * RES_FACTOR
    print(f"\n[Hist] Image size: {hist_nx}×{hist_ny} "
          f"({img_size[0]}×{img_size[1]} × res factor {RES_FACTOR})")

    # ── 3. Accumulate histograms ─────────────────────────────────────────
    hists = [np.zeros((hist_ny, hist_nx), dtype=np.float64) for _ in range(N_BINS)]
    counts_per_bin = np.zeros(N_BINS, dtype=np.int64)
    n_events_total = 0
    n_no_gmd = 0  # events in frames without GMD data

    print(f"\n[Hist] Reading checkpoint in chunks...")
    for chunk in pd.read_hdf(CHECKPOINT_PATH, "localizations",
                             columns=["x", "y", "frame"], chunksize=500_000):
        x = chunk["x"].values * RES_FACTOR
        y = chunk["y"].values * RES_FACTOR
        frames = chunk["frame"].values.astype(int)

        # GMD bin for each event (NaN frames → bin = -1, discarded)
        bins = np.full(len(chunk), -1, dtype=int)
        valid_frames = frames < len(gmd_bin)
        bins[valid_frames] = gmd_bin[frames[valid_frames]]

        # Per-event GMD value, used when WEIGHT_MODE != "none"
        gmd_ev = np.full(len(chunk), np.nan)
        gmd_ev[valid_frames] = gmd_arr[frames[valid_frames]]

        for b in range(N_BINS):
            m = (bins == b) & np.isfinite(x) & np.isfinite(y)
            if not m.any():
                continue
            n_b = int(m.sum())
            counts_per_bin[b] += n_b

            # Per-frame scalar weight (see WEIGHT_MODE notes at the top).
            # This rescales each frame's contribution; it cannot change the
            # shape of an individual frame's distribution.
            if WEIGHT_MODE == "gmd":
                w = gmd_ev[m]
            elif WEIGHT_MODE == "gmd_norm":
                w = (gmd_mean / gmd_ev[m]) ** WEIGHT_POWER
                # Clamp pathological weights (near-zero GMD) — zeroed, as in V5
                if GMD_WEIGHT_MAX > 0:
                    w = np.where(w > GMD_WEIGHT_MAX, 0.0, w)
                if GMD_WEIGHT_MIN > 0:
                    w = np.where(w < GMD_WEIGHT_MIN, 0.0, w)
            else:
                w = None

            h, _, _ = np.histogram2d(
                y[m], x[m],
                bins=[hist_ny, hist_nx],
                range=[[0, hist_ny], [0, hist_nx]],
                weights=w,
            )
            hists[b] += h

        n_events_total += len(chunk)
        n_no_gmd += int((bins == -1).sum())

    print(f"[Hist] Total events:       {n_events_total:,}")
    print(f"[Hist] Events w/o GMD:     {n_no_gmd:,}  "
          f"({n_no_gmd / max(n_events_total, 1) * 100:.2f}%)")
    print(f"[Hist] Events per bin:")
    for b in range(N_BINS):
        lo = "-inf" if b == 0 else f"{thresholds[b-1]:.3e}"
        hi = "+inf" if b == N_BINS - 1 else f"{thresholds[b]:.3e}"
        print(f"  bin {b}  [{lo}, {hi}):  {counts_per_bin[b]:,}  "
              f"({counts_per_bin[b] / max(n_events_total, 1) * 100:.2f}%)")

    # ── 4. Overall histogram = sum of the bin histograms ─────────────────
    hist_all = np.zeros((hist_ny, hist_nx), dtype=np.float64)
    for h in hists:
        hist_all += h
    counts_all = counts_per_bin.sum()
    print(f"[Hist] Overall events:      {counts_all:,}")

    # ── 5. Save histograms ───────────────────────────────────────────────
    # A suffix keeps weighted output from overwriting the default (unweighted)
    # bins that the downstream circularization/CPBASEX pipeline consumes.
    if WEIGHT_MODE == "none":
        suffix = ""
    elif WEIGHT_MODE == "gmd_norm" and WEIGHT_POWER != 1.0:
        suffix = f"_w-gmd_norm_p{WEIGHT_POWER}"
    else:
        suffix = f"_w-{WEIGHT_MODE}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for b in range(N_BINS):
        lo = "minusinf" if b == 0 else f"{thresholds[b-1]:.2e}"
        hi = "plusinf" if b == N_BINS - 1 else f"{thresholds[b]:.2e}"
        fname = os.path.join(OUTPUT_DIR,
                             f"vmi_2d_gmd_bin{b}_{lo}_{hi}{suffix}.npy")
        np.save(fname, hists[b])
        print(f"[Save] {fname}")

    fname_all = os.path.join(OUTPUT_DIR, f"vmi_2d_gmd_all{suffix}.npy")
    np.save(fname_all, hist_all)
    print(f"[Save] {fname_all}")

    # ── 6. Plot 2×3 grid (4 bins + overall + summary) ────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="white", layout = 'constrained')
    axes = axes.ravel()

    for b in range(N_BINS):
        ax = axes[b]
        lo = "-inf" if b == 0 else f"{thresholds[b-1]:.2e}"
        hi = "+inf" if b == N_BINS - 1 else f"{thresholds[b]:.2e}"
        vmax = hists[b].max()
        im = ax.imshow(hists[b], origin="lower", cmap="hot",
                       norm=LogNorm(vmin=1, vmax=vmax),
                       interpolation="nearest")
        ax.set_title(f"Bin {b}: GMD [{lo}, {hi})\n"
                     f"{counts_per_bin[b]:,} events", fontsize=11)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel: overall histogram
    ax = axes[N_BINS]
    vmax_all = hist_all.max()
    im = ax.imshow(hist_all, origin="lower", cmap="hot",
                   norm=LogNorm(vmin=1, vmax=vmax_all),
                   interpolation="nearest")
    ax.set_title(f"Overall (sum of bins)\n{counts_all:,} events", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel: summary text
    ax = axes[N_BINS + 1]
    ax.axis("off")
    txt = f"VMI BY GMD BINS\n{'─'*28}\n"
    txt += f"Total events:  {n_events_total:,}\n"
    txt += f"Binned events: {counts_all:,}\n"
    txt += f"No GMD:        {n_no_gmd:,}\n\n"
    txt += f"GMD thresholds:\n"
    for b in range(N_BINS):
        lo = "-inf" if b == 0 else f"{thresholds[b-1]:.2e}"
        hi = "+inf" if b == N_BINS - 1 else f"{thresholds[b]:.2e}"
        txt += f"  bin{b} [{lo}, {hi}): {counts_per_bin[b]:,}\n"
    txt += f"\nImage: {hist_nx}×{hist_ny}\n"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            fontsize=11, va="top", fontfamily="monospace")

    plt.suptitle(f"VMI Histograms by GMD Bins — {counts_all:,} events",
                 fontsize=14, fontweight="bold")
    # Note: layout='constrained' in subplots handles spacing; do NOT call
    # plt.tight_layout() here — it conflicts with constrained layout + colorbars.
    out_png = os.path.join(OUTPUT_DIR, f"vmi_gmd_bins{suffix}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {out_png}")
    if NO_SHOW:
        print("[Plot] --no-show: skipping live window.")
    else:
        print("[Plot] Showing live window — close it to finish...")
        plt.show()
    plt.close()

    print("\nDONE")


if __name__ == "__main__":
    main()
