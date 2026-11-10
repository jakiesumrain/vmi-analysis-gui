"""Locate the near-zero-energy reference pixel in a VMI image."""

import numpy as np


def brightest_pixel(image):
    """Return the zero-based (row, column) of the first finite maximum."""
    image = np.asarray(image)
    if image.ndim != 2 or not image.size:
        raise ValueError("Expected a nonempty 2D VMI image.")
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("The VMI image has no finite pixels.")
    peak = np.argmax(image if finite.all() else np.where(finite, image, -np.inf))
    row, col = np.unravel_index(peak, image.shape)
    return int(row), int(col)
