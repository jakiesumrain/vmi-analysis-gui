"""Correct localization coordinates and accumulate their unweighted counts."""

import json
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd

from vmi_analysis.centroiding.frame_roster import successful_shots
from vmi_analysis.paths import validate_distinct_output

from .circularization import eval_reassignment_model, source_radius


STATUS = {"counted": 0, "invalid_input": 1, "outside_calibration": 2, "outside_histogram": 3}


def checkpoint_targets(checkpoint, extracted, reassign, corrected, counts, *, cwd=None):
    inputs = (("source checkpoint", checkpoint), ("trig-fit calibration", extracted),
              ("radial reassignment file", reassign))
    corrected = validate_distinct_output(corrected, inputs, cwd=cwd, output_label="corrected checkpoint")
    counts = validate_distinct_output(counts, (*inputs, ("corrected checkpoint", corrected)),
                                      cwd=cwd, output_label="count histogram")
    if corrected.suffix.lower() not in (".h5", ".hdf5") or counts.suffix.lower() != ".npy":
        raise ValueError("Use .h5 or .hdf5 for the corrected checkpoint and .npy for the count histogram.")
    return corrected, counts


def _slope_bound(model, radius):
    """Bound the absolute derivative over [0, radius], including spline knots."""
    if model["kind"] == "poly":
        slope = np.poly1d([*model["coeffs"], 0.0]).deriv()
        roots = slope.deriv().r
        points = np.real(roots[np.isreal(roots)])
    else:
        radius = min(radius, model["r_max"])
        slope = model["dinterp"]
        points = np.r_[slope.x, slope.derivative().roots(extrapolate=False)]
    points = points[(points > 0) & (points < radius)]
    return np.max(np.abs(slope(np.r_[0.0, points, radius])))


class EventWarp:
    """Invert the image sampling map up to the edges of the histogram grid."""

    def __init__(self, fit_params, max_trig_n, reassignment, polar_dr, shape, center):
        if not np.isfinite(polar_dr) or polar_dr <= 0:
            raise ValueError("Polar radial step must be finite and positive.")
        self.map_args = fit_params, max_trig_n, reassignment, polar_dr
        self.shape, self.center = shape, center
        ny, nx = shape
        xc, yc = center
        # Include the half-pixel edges of the histogram, not just pixel centers.
        self.left, self.right = xc + 0.5, nx - xc - 0.5
        self.top, self.bottom = yc + 0.5, ny - yc - 0.5
        self.radius = float(np.hypot(max(self.left, self.right), max(self.top, self.bottom)))
        outer_target, _ = eval_reassignment_model(reassignment, self.radius / polar_dr)

        # Reassignment is strictly increasing by construction. Bound the remaining
        # derivative over the full grid, including the continued radial model.
        bound = sum(_slope_bound(model, outer_target) for model in fit_params.values())
        if not np.isfinite(bound) or bound >= 1:
            raise ValueError("Cannot verify a strictly increasing radial map for all angles. "
                             "Check the fitted trig terms and radial model.")
        self.identity = bound == 0 and reassignment["kind"] == "identity"
        self.iterations = max(1, int(np.ceil(np.log2(self.radius) - np.log2(1e-8))))

    def correct(self, x, y, resolution):
        # Centroiding bins detector coordinates from edge zero. Image index 0
        # denotes the center of its first bin on the calibration grid.
        dx = x * resolution - 0.5 - self.center[0]
        dy = y * resolution - 0.5 - self.center[1]
        radius = np.hypot(dx, dy)
        status = np.full(len(x), STATUS["invalid_input"], dtype=np.uint8)
        finite = np.isfinite(radius)
        if self.identity:
            status[finite] = STATUS["counted"]
            return x, y, status

        status[finite] = STATUS["outside_histogram"]
        corrected_x, corrected_y = np.full_like(x, np.nan), np.full_like(y, np.nan)
        measured_radius = radius[finite]
        dx, dy = dx[finite], dy[finite]
        theta = np.arctan2(dy, dx)
        # Intersect each ray with the output rectangle to bracket the inverse.
        x_edge = np.where(dx < 0, self.left, self.right)
        y_edge = np.where(dy < 0, self.top, self.bottom)
        x_limit = np.divide(x_edge * measured_radius, np.abs(dx),
                            out=np.full_like(dx, self.radius), where=dx != 0)
        y_limit = np.divide(y_edge * measured_radius, np.abs(dy),
                            out=np.full_like(dy, self.radius), where=dy != 0)
        limit = np.minimum(x_limit, y_limit)
        upper, _ = source_radius(limit, theta, *self.map_args)
        keep = measured_radius <= upper
        valid = np.flatnonzero(finite)[keep]
        status[valid] = STATUS["counted"]

        theta, radius = theta[keep], measured_radius[keep]
        lo, hi = np.zeros_like(radius), limit[keep]
        for _ in range(self.iterations):
            mid = (lo + hi) / 2
            measured, _ = source_radius(mid, theta, *self.map_args)
            below = measured < radius
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        corrected = (lo + hi) / 2
        scale = np.divide(corrected, radius, out=np.ones_like(radius), where=radius > 0)
        # The bracket guarantees these coordinates are in the grid. Clip only
        # floating-point roundoff at the rectangular boundaries.
        corrected_x[valid] = np.clip(x[valid] + dx[keep] * (scale - 1) / resolution, 0, self.shape[1] / resolution)
        corrected_y[valid] = np.clip(y[valid] + dy[keep] * (scale - 1) / resolution, 0, self.shape[0] / resolution)
        return corrected_x, corrected_y, status


def _bin_events(histogram, x, y, x_edges, y_edges, include=True):
    ny, nx = histogram.shape
    inside = include & (x >= x_edges[0]) & (x <= x_edges[-1]) & (y >= y_edges[0]) & (y <= y_edges[-1])
    # Interior edges enter the bin on their right. The final edge is inclusive.
    cols = np.minimum(np.searchsorted(x_edges, x[inside], side="right") - 1, nx - 1)
    rows = np.minimum(np.searchsorted(y_edges, y[inside], side="right") - 1, ny - 1)
    np.add.at(histogram, (rows, cols), 1)
    return inside


def correct_checkpoint(checkpoint, corrected, counts, *, shape, center, resolution, warp,
                       metadata, polar_preview=None, chunksize=250_000):
    """Export retained electrons and the complete successful-shot roster."""
    with h5py.File(checkpoint, "r") as source:
        shots = successful_shots(source)
    if not len(shots):
        raise ValueError("The raw checkpoint has no successfully processed shots. Rerun Centroiding.")
    ny, nx = shape
    x_edges = np.arange(nx + 1, dtype=np.float64) / resolution
    y_edges = np.arange(ny + 1, dtype=np.float64) / resolution
    histogram = np.zeros(shape, dtype=np.int64)
    raw_histogram = np.zeros_like(histogram) if polar_preview is not None and warp is not None else None
    totals = np.zeros(len(STATUS), dtype=np.int64)
    for path in (corrected, counts):
        path.parent.mkdir(parents=True, exist_ok=True)
    with (TemporaryDirectory(dir=corrected.parent, prefix=".circularize-") as checkpoint_dir,
          TemporaryDirectory(dir=counts.parent, prefix=".circularize-") as counts_dir):
        staged_checkpoint = Path(checkpoint_dir) / "checkpoint.h5"
        staged_counts = Path(counts_dir) / "counts.npy"
        with (pd.HDFStore(checkpoint, mode="r") as source,
              h5py.File(staged_checkpoint, "w") as destination):
            if "/localizations" not in source or not source.get_storer("localizations").is_table:
                raise ValueError("The source checkpoint needs a localizations table.")
            columns = source.select("localizations", start=0, stop=0).columns
            if not {"x", "y", "frame"}.issubset(columns):
                raise ValueError("The localizations table needs x, y, and frame columns.")
            events = destination.create_group("events")
            for name, dtype in (("shot_id", "i8"), ("x_corrected", "f8"), ("y_corrected", "f8")):
                events.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                      chunks=(min(chunksize, 65_536),), compression="gzip", compression_opts=1)
            destination.create_group("shots").create_dataset("id", data=shots, dtype="i8")
            geometry = destination.create_group("geometry")
            geometry.create_dataset("x_edges", data=x_edges)
            geometry.create_dataset("y_edges", data=y_edges)
            geometry.attrs.update(x_direction="right", y_direction="down", coordinate_units="detector_pixel",
                                  center_xy=(np.asarray(center) + 0.5) / resolution,
                                  resolution_factor=resolution)
            for chunk in source.select("localizations", chunksize=chunksize):
                frames = chunk["frame"].to_numpy()
                if (frames.dtype.kind not in "iu"
                        or (frames.dtype.kind == "u" and np.any(frames > np.iinfo(np.int64).max))):
                    raise ValueError("Localizations need integer shot IDs in the Int64 range. Rerun Centroiding.")
                frames = frames.astype(np.int64, copy=False)
                if not np.isin(frames, shots).all():
                    raise ValueError("Localizations reference invalid or unsuccessful shots. Rerun Centroiding.")
                raw_x = chunk["x"].to_numpy(dtype=float)
                raw_y = chunk["y"].to_numpy(dtype=float)
                if warp is None:
                    x, y = raw_x, raw_y
                    status = np.where(np.isfinite(x) & np.isfinite(y), STATUS["counted"], STATUS["invalid_input"])
                else:
                    x, y, status = warp.correct(raw_x, raw_y, resolution)
                if raw_histogram is not None:
                    _bin_events(raw_histogram, raw_x, raw_y, x_edges, y_edges)
                valid = status == STATUS["counted"]
                inside = _bin_events(histogram, x, y, x_edges, y_edges, valid)
                status[valid & ~inside] = STATUS["outside_histogram"]
                start = int(totals[STATUS["counted"]])
                totals += np.bincount(status, minlength=len(STATUS))
                stop = int(totals[STATUS["counted"]])
                if stop > start:
                    for name, values in (("shot_id", frames), ("x_corrected", x), ("y_corrected", y)):
                        events[name].resize((stop,))
                        events[name][start:stop] = values[inside]

            summary = {name: int(totals[code]) for name, code in STATUS.items()}
            destination.attrs.update(schema_version=1, source_checkpoint=str(Path(checkpoint).resolve()),
                                     coordinate_correction="none" if warp is None else "circularize")
            provenance = dict(metadata)
            if warp is not None:
                destination.attrs["calibration_id"] = provenance.pop("calibration_id")
            destination.attrs["circularization"] = json.dumps({
                **provenance, "max_corrected_radius_grid_pixels": warp.radius if warp is not None else None,
                "event_totals": summary,
            })
        with staged_counts.open("wb") as stream:
            np.save(stream, histogram, allow_pickle=False)
        prior_counts = Path(counts_dir) / "previous-counts.npy"
        if counts.exists():
            copy2(counts, prior_counts)
        staged_counts.replace(counts)
        try:
            staged_checkpoint.replace(corrected)
        except OSError:
            if prior_counts.exists():
                prior_counts.replace(counts)
            else:
                counts.unlink()
            raise
    if polar_preview is not None:
        from .polar_fit import save_polar_comparison

        try:
            if raw_histogram is None:
                save_polar_comparison(polar_preview, histogram, None, center)
            else:
                save_polar_comparison(polar_preview, raw_histogram, histogram, center)
        except (OSError, ValueError) as exc:
            print(f"Warning: could not save polar previews: {exc}")
    return summary
