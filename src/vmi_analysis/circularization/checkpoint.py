"""Correct localization coordinates and accumulate their unweighted counts."""

import json
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import brentq

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
    """Invert the image sampling map over the supported radial interval."""

    def __init__(self, fit_params, max_trig_n, reassignment, polar_dr, outer_ring):
        if not np.isfinite([polar_dr, outer_ring]).all() or min(polar_dr, outer_ring) <= 0:
            raise ValueError("Polar radial step and outer fitted radius must be finite and positive.")
        self.map_args = fit_params, max_trig_n, reassignment, polar_dr
        outer_fixed = outer_ring
        supported = outer_ring
        if reassignment["kind"] != "identity":
            outer_fixed = reassignment["r_max"]

            def target(r):
                return eval_reassignment_model(reassignment, r)[0]

            supported = min(supported, float(target(outer_fixed)))
            if target(outer_fixed) > supported:
                outer_fixed = brentq(lambda r: target(r) - supported, 0, outer_fixed)

        # Reassignment is strictly increasing by construction. Bound the remaining
        # derivative for every angle, including between detector pixel centers.
        bound = sum(_slope_bound(model, supported) for model in fit_params.values())
        if not np.isfinite(bound) or bound >= 1:
            raise ValueError("Cannot verify a strictly increasing radial map for all angles. "
                             "Check the fitted trig terms and radial model.")
        self.radius = polar_dr * outer_fixed
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("The supported corrected radius must be finite and positive.")
        self.identity = bound == 0 and reassignment["kind"] == "identity"
        self.iterations = max(1, int(np.ceil(np.log2(self.radius) - np.log2(1e-8))))

    def correct(self, x, y, center, resolution):
        # Centroiding bins detector coordinates from edge zero. Image index 0
        # denotes the center of its first bin on the calibration grid.
        dx = x * resolution - 0.5 - center[0]
        dy = y * resolution - 0.5 - center[1]
        radius = np.hypot(dx, dy)
        status = np.full(len(x), STATUS["invalid_input"], dtype=np.uint8)
        finite = np.isfinite(radius)
        status[finite] = STATUS["outside_calibration"]
        corrected_x, corrected_y = np.full_like(x, np.nan), np.full_like(y, np.nan)
        theta = np.arctan2(dy[finite], dx[finite])
        if self.identity:
            upper = self.radius
        else:
            upper, _ = source_radius(self.radius, theta, *self.map_args)
        valid = np.flatnonzero(finite)[radius[finite] <= upper]
        status[valid] = STATUS["counted"]
        if self.identity:
            corrected_x[valid], corrected_y[valid] = x[valid], y[valid]
            return corrected_x, corrected_y, status

        theta = np.arctan2(dy[valid], dx[valid])
        radius = radius[valid]
        lo, hi = np.zeros_like(radius), np.full_like(radius, self.radius)
        for _ in range(self.iterations):
            mid = (lo + hi) / 2
            measured, _ = source_radius(mid, theta, *self.map_args)
            below = measured < radius
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        corrected = (lo + hi) / 2
        scale = np.divide(corrected, radius, out=np.ones_like(radius), where=radius > 0)
        corrected_x[valid] = x[valid] + dx[valid] * (scale - 1) / resolution
        corrected_y[valid] = y[valid] + dy[valid] * (scale - 1) / resolution
        return corrected_x, corrected_y, status


def correct_checkpoint(checkpoint, corrected, counts, *, shape, center, resolution, warp,
                       metadata, chunksize=250_000):
    """Stream rows into a new HDF5 checkpoint and publish the integer image."""
    ny, nx = shape
    histogram = np.zeros(shape, dtype=np.int64)
    totals = np.zeros(len(STATUS), dtype=np.int64)
    for path in (corrected, counts):
        path.parent.mkdir(parents=True, exist_ok=True)
    with (TemporaryDirectory(dir=corrected.parent, prefix=".circularize-") as checkpoint_dir,
          TemporaryDirectory(dir=counts.parent, prefix=".circularize-") as counts_dir):
        staged_checkpoint = Path(checkpoint_dir) / "checkpoint.h5"
        staged_counts = Path(counts_dir) / "counts.npy"
        with (pd.HDFStore(checkpoint, mode="r") as source,
              pd.HDFStore(staged_checkpoint, mode="w", complevel=1) as destination):
            if "/localizations" not in source or not source.get_storer("localizations").is_table:
                raise ValueError("The source checkpoint needs a localizations table.")
            columns = source.select("localizations", start=0, stop=0).columns
            if not {"x", "y", "frame"}.issubset(columns):
                raise ValueError("The localizations table needs x, y, and frame columns.")
            for chunk in source.select("localizations", chunksize=chunksize):
                x, y, status = warp.correct(chunk["x"].to_numpy(dtype=float),
                                            chunk["y"].to_numpy(dtype=float), center, resolution)
                px, py = x * resolution, y * resolution
                valid = status == STATUS["counted"]
                inside = valid & (px >= 0) & (px <= nx) & (py >= 0) & (py <= ny)
                status[valid & ~inside] = STATUS["outside_histogram"]
                # Match the legacy histogram's inclusive rightmost edge.
                cols = np.minimum(px[inside].astype(np.intp), nx - 1)
                rows = np.minimum(py[inside].astype(np.intp), ny - 1)
                np.add.at(histogram, (rows, cols), 1)
                totals += np.bincount(status, minlength=len(STATUS))
                destination.append("localizations", chunk.assign(
                    x_corrected=x, y_corrected=y, correction_status=status), index=False)
            if not totals.sum():
                raise ValueError("The source checkpoint has no localizations.")

        summary = {name: int(totals[code]) for name, code in STATUS.items()}
        with h5py.File(checkpoint, "r") as source, h5py.File(staged_checkpoint, "r+") as destination:
            for name in source:
                if name != "localizations":
                    source.copy(name, destination)
            for key, value in source.attrs.items():
                if key not in destination.attrs:
                    destination.attrs[key] = value
            destination.attrs["circularization"] = json.dumps({
                **metadata, "version": 1, "source_checkpoint": str(Path(checkpoint).resolve()),
                "shape_yx": list(shape), "center_xy": list(center), "resolution_factor": resolution,
                "grid_center_offset": -0.5, "max_corrected_radius": warp.radius,
                "correction_status": STATUS, "event_totals": summary,
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
    return summary
