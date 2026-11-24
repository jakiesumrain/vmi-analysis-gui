"""Threshold validation and artifact names shared by Binning and its CLI."""

from math import isfinite


DEFAULT_THRESHOLDS = "3.75e-8,5.5e-8,7.25e-8"
DEFAULT_GMD_COLUMN = 4
DEFAULT_WEIGHT_MODE = "gmd_norm"
DEFAULT_WEIGHT_POWER = 1.0
DEFAULT_GMD_FLOOR = 1e-12
WEIGHT_MODES = ("none", "gmd", "gmd_norm")


def parse_thresholds(text):
    try:
        values = tuple(float(value) for value in text.split(","))
    except (AttributeError, ValueError) as exc:
        raise ValueError("Enter comma-separated numeric GMD thresholds.") from exc
    if not all(map(isfinite, values)) or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("GMD thresholds must be finite and strictly increasing.")
    return values


def weight_suffix(mode, power):
    if mode == "none":
        return ""
    if mode == "gmd_norm" and power != 1.0:
        return f"_w-gmd_norm_p{power}"
    return f"_w-{mode}"


def histogram_filenames(thresholds, suffix):
    limits = ("minusinf", *(f"{value:.2e}" for value in thresholds), "plusinf")
    return tuple(f"vmi_2d_gmd_bin{index}_{lo}_{hi}{suffix}.npy"
                 for index, (lo, hi) in enumerate(zip(limits, limits[1:])))
