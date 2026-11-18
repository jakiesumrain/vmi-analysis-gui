"""Histogram edges for the Ion TOF panels."""

from math import ceil, isclose, isfinite

import numpy as np


DEFAULT_BIN_SIZE_NS = 6500 / 1200
MAX_BINS = 20_000


def tof_bin_edges(minimum, maximum, bins, *, bin_size_ns=None):
    """Keep the requested TOF limits and shorten the last bin when needed."""
    if bin_size_ns is None:
        if not 1 <= bins <= MAX_BINS:
            raise ValueError(f"Choose between 1 and {MAX_BINS:,} bins.")
        edges = np.linspace(minimum, maximum, bins + 1)
    else:
        if not isfinite(bin_size_ns) or bin_size_ns <= 0:
            raise ValueError("TOF bin size must be finite and greater than zero.")
        steps = (maximum - minimum) / bin_size_ns
        if not isfinite(steps):
            raise ValueError(f"Increase the TOF bin size to use at most {MAX_BINS:,} bins.")
        nearest = round(steps)
        count = nearest if nearest and isclose(steps, nearest, rel_tol=1e-12) else max(1, ceil(steps))
        if count > MAX_BINS:
            raise ValueError(f"Increase the TOF bin size to use at most {MAX_BINS:,} bins.")
        edges = minimum + bin_size_ns * np.arange(count + 1)
        edges[-1] = maximum
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError("The TOF limits are too close for the selected binning.")
    return edges
