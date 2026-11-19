"""LUNA input for the Ion Pipeline."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h5py
import numpy as np

from vmi_analysis.ion_luna import (ACQUISITION_PREFIX, apply_timewalk, frame_order, input_files,
                                    luna_command, preprocess_acquisition, process_arguments, publish_result)
from vmi_analysis.ion_pipeline import PositionData, delay_from_files
from vmi_analysis.ion_acquisition import TDC_PERIOD_TICKS, acquisition_splits, read_timing
from vmi_analysis.ion_tof import ARRIVALS, DEFAULT_REFERENCE, _table, _tof_blocks, _trigger_times


NAME = "LUNA"
DEFAULT_SHIFT = 1
SHIFT_HELP = "GMD2 row j selects frame j + shift within the same acquisition."
def _decode_segment(paths, raw_tdc, target, options, log):
    windows = sys.platform == "win32"
    source = options["source"]
    with TemporaryDirectory(dir=target.parent, prefix=".luna-segment-") as temporary:
        scratch = Path(temporary)
        metadata = scratch / "metadata.h5"
        framed = scratch / "framed"
        preprocess_acquisition(paths, framed, metadata, numbered=True, combined=True)
        config = {
            "layout": "single", "tof-tdc-reference": DEFAULT_REFERENCE,
            "clustering-algorithm": options["clustering_algorithm"], "eps-t": options["eps_t"],
            "eps-s": options["eps_s"], "kuttruff-buffer-size": options["kuttruff_buffer_size"],
            "disable-clustering": options["disable_clustering"], "ctot-cut": options["ctot_cut"],
            "calculate-timewalk-statistics": options["timewalk"], "log-level": "warn",
        }
        pending = scratch / "decoded.h5"
        raw_output = scratch / "uncorrected.h5" if options["timewalk"] else target
        command = luna_command(framed / "acquisition.tpx3", pending, None, process_arguments(config), options["wsl_distribution"],
                               options["wsl_executable"], windows=windows)
        publish_result(command, pending, raw_output, windows=windows, log=log)
        if options["timewalk"]:
            apply_timewalk(raw_output, target, SimpleNamespace(
                ctot_cut=options["ctot_cut"], log_level="warn", wsl_distribution=options["wsl_distribution"],
                wsl_executable=options["wsl_executable"],
            ), windows=windows, log=log)
        with h5py.File(metadata, "r") as file:
            gmd2 = file["frames"].fields("gmd2")[:] * 1e9
        if options["keep_dumps"]:
            metadata.replace(target.with_name(target.stem + "-metadata.h5"))
    with h5py.File(target, "r") as dump:
        triggers = _trigger_times(dump, DEFAULT_REFERENCE)
        if len(triggers) != len(paths):
            raise ValueError(f"LUNA returned {len(triggers):,} triggers for {len(paths):,} frames.")
        residual = (triggers.astype(float) - raw_tdc + TDC_PERIOD_TICKS / 2) % TDC_PERIOD_TICKS
        if np.any(np.abs(residual - TDC_PERIOD_TICKS / 2) > 50):
            raise ValueError("The LUNA trigger sequence does not match the acquisition frames.")
        dataset, field = ARRIVALS[source]
        chunks = list(_tof_blocks(_table(dump, dataset, field), field, triggers))
    return (np.concatenate([values for values, _, _ in chunks]) if chunks else np.empty(0),
            np.concatenate([indices for _, indices, _ in chunks]) if chunks else np.empty(0, dtype=np.int64),
            gmd2)


def decode(folder, delay, output, options):
    inputs = frame_order(input_files(folder))
    if inputs is None:
        raise ValueError("LUNA verification requires frame numbers in the input filenames.")
    timing = [read_timing(path) for path in inputs]
    wall = np.array([row[0] for row in timing], dtype=np.int64)
    tdc = np.array([row[1] for row in timing])
    boundaries = [0, *acquisition_splits(wall, tdc), len(inputs)]
    paths = output / "dumps" / folder.name if options["keep_dumps"] else None
    if paths is not None:
        paths.mkdir(parents=True, exist_ok=True)
    log_folder = output / "logs"
    log_folder.mkdir(exist_ok=True)
    log_file = log_folder / (folder.name + ".log")
    tofs, frame_indices, gmd = [], [], []
    print(f"  Decoding {len(inputs):,} frames in {len(boundaries) - 1} acquisition segment(s).", flush=True)
    try:
        with TemporaryDirectory(dir=output, prefix=".ion-position-") as temporary, \
                log_file.open("w", encoding="utf-8") as log:
            for i, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
                target = (paths or Path(temporary)) / f"segment-{i:03d}.h5"
                tof, frames, gmd2 = _decode_segment(inputs[start:end], tdc[start:end], target, options, log)
                tofs.append(tof)
                frame_indices.append(frames + start)
                gmd.append(gmd2)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{exc} See {log_file.name} in the run's logs folder.") from exc
    return PositionData(delay_from_files(inputs, delay), np.concatenate(tofs), np.concatenate(frame_indices),
                        np.concatenate(gmd), np.diff(boundaries).tolist())


def select_frames(data, options):
    selected = np.zeros(len(data.gmd2_na), dtype=bool)
    shift = options["gmd_shift"]
    start = 0
    for count in data.segment_lengths:
        stop = start + count
        gmd = data.gmd2_na[start:stop]
        matches = np.flatnonzero((gmd >= options["gmd_min"]) & (gmd <= options["gmd_max"])) + shift
        selected[start + matches[(matches >= 0) & (matches < count)]] = True
        start = stop
    return selected


def fit_peak(values, low, high, center, bins):
    from scipy.optimize import curve_fit

    def gaussian(x, height, mu, sigma):
        return height * np.exp(-0.5 * ((x - mu) / sigma)**2)

    sample = values[(values >= low) & (values <= high)]
    if len(sample) < 200:
        return None
    counts, edges = np.histogram(sample, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    try:
        params, _ = curve_fit(gaussian, centers, counts, p0=[counts.max(), center, 70.0], maxfev=40_000)
    except (ValueError, RuntimeError):
        return None
    _, mu, sigma = params
    if not np.isfinite(params).all() or not sigma or not low <= mu <= high:
        return None
    return {"mu": float(mu), "sigma": abs(float(sigma))}


def y_label(options):
    noun = "Pixel hits" if options["source"] == "pixels" else "Clustered events"
    return f"{noun} per selected frame"
