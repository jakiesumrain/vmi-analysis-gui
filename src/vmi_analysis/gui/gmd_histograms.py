"""GMD value and weight distributions for the Binning panel previews."""

from __future__ import annotations

import numpy as np

GMD_COLUMNS = (4, 5)


def load_gmd_map(path, column=4):
    """Load (frame index, raw GMD value) pairs from a GMD map."""
    if column not in GMD_COLUMNS:
        raise ValueError("GMD column must be 4 (value 1) or 5 (value 2).")
    map_data = np.loadtxt(path, comments="#", usecols=(0, column), ndmin=2)
    if map_data.size == 0:
        raise ValueError("GMD map contains no rows")
    frame_index = map_data[:, 0].astype(int)
    gmd_values = map_data[:, 1].astype(float)
    if not np.isfinite(gmd_values).all():
        raise ValueError("GMD map contains non-finite values")
    return frame_index, gmd_values


def gmd_weights(gmd_values, mode="gmd_norm", power=1.0):
    """Compute raw per-frame weights for the histogram previews.

    This intentionally applies no GMD floor and no weight limits; those
    settings belong to the binning job, not to these diagnostic histograms.
    """
    values = np.asarray(gmd_values, dtype=float)
    if values.size == 0:
        raise ValueError("GMD values are empty")
    if mode == "none":
        return np.ones_like(values)
    if mode == "gmd":
        return values.copy()
    if mode != "gmd_norm":
        raise ValueError(f"Unknown GMD weight mode: {mode!r}")
    with np.errstate(divide="ignore", invalid="ignore"):
        return (float(np.mean(values)) / values) ** float(power)


def gmd_histogram_samples(path, column=4, mode="gmd_norm", power=1.0):
    """Return (GMD values, weights, mean GMD) for histogram previews."""
    _, gmd_values = load_gmd_map(path, column)
    weights = gmd_weights(gmd_values, mode=mode, power=power)
    return gmd_values, weights, float(np.mean(gmd_values))
