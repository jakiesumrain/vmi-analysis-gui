"""Clean-room extraction of the circularisation trig-fit coefficients.

Pipeline (rows are the polar angle, columns the radius, exactly like the
Gascooke software's "Theta vs r" image):

    cartesian image
      -> polar remap (rows theta, radius columns, includes r = 0)
      -> optional inverse Abel          (Hansen-Law, per constant-theta row)
      -> per-ring Gaussian ridge fit
      -> canonical extracted trig-fit block (one block per accepted ring)

Enhancement changes only the display. The fit uses point-sampled detector
intensities with no polar-bin area correction.

No Gascooke executable is involved; the model is the published one:

    rg(theta) = r0 + sum_n [ A_n sin(n theta) + B_n cos(n theta) ]
    I(r,theta) = I0 (1 + beta P2(cos theta)) exp(-(r-rg)^2/(2 sigma^2)) + c

with P2(x) = (3x^2 - 1)/2. The Abel transform is applied along the radial
axis, one constant-angle line at a time, as in the paper.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares

from .circularization import parse_extracted_text

TWO_PI = 2.0 * np.pi
EXTRACTED_HEADER = '"Parameter" "Value" "+-" "Uncertainty (1 std dev)"'

DEFAULT_ENHANCE_PARAMS = {
    "smooth_theta": 1.0,
    "smooth_r": 0.8,
    "low_r": 2.0,
    "high_r": 20.0,
    "background_theta": 8.0,
    "background_r": 25.0,
    "ridge_sigma": 1.2,
    "blur_theta": 6.0,
    "blur_r": 18.0,
    "clip_limit": 0.02,
    "kernel_theta": 24,
    "kernel_r": 96,
}
ENHANCE_MODES = ("none", "bandpass", "ridge", "unsharp", "clahe")
ENHANCE_PLACEMENTS = ("off", "raw", "pre_abel", "post_abel")


# ---------------------------------------------------------------------------
# Image loading and polar remap
# ---------------------------------------------------------------------------


def load_image(path):
    """Load a 2D NumPy detector image."""
    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Choose a NumPy detector image (.npy).")
    image = np.load(path, allow_pickle=False)
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    return image


def polar_grid(image_shape, center, n_r=512, n_theta=720, r_max=None, *, radial_bin_size=None):
    """Return (theta, r, dr) for a polar remap that includes r = 0."""
    ny, nx = image_shape
    cx, cy = (float(center[0]), float(center[1]))
    if not (0.0 <= cx <= nx - 1 and 0.0 <= cy <= ny - 1):
        raise ValueError(f"Centre {(cx, cy)} lies outside a {nx}x{ny} image")
    if r_max is None:
        r_max = min(cx, nx - 1 - cx, cy, ny - 1 - cy)
    r_max = float(r_max)
    if not np.isfinite(r_max) or r_max <= 0:
        raise ValueError("The maximum radius must be finite and positive.")
    if radial_bin_size is None:
        n_r = int(n_r)
    else:
        dr = float(radial_bin_size)
        if not np.isfinite(dr) or dr <= 0:
            raise ValueError("The radial bin size must be finite and positive.")
        # Allow one rounding step when the maximum radius is an exact multiple.
        steps = np.nextafter(r_max / dr, np.inf)
        if not np.isfinite(steps):
            raise ValueError("The radial bin size is too small for this maximum radius.")
        n_r = int(steps) + 1
    n_theta = int(n_theta)
    if n_r < 8 or n_theta < 8:
        raise ValueError("The polar grid needs at least 8 radial samples and 8 angular bins.")
    theta = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
    if radial_bin_size is None:
        r = np.linspace(0.0, r_max, n_r)
        dr = float(r[1] - r[0])
    else:
        r = np.arange(n_r, dtype=float) * dr
        r[-1] = min(r[-1], r_max)
    return theta, r, dr


def remap_polar(image, center, n_r=512, n_theta=720, r_max=None, order=1, *, radial_bin_size=None):
    """Remap a cartesian image to polar coordinates.

    Returns ``(polar, theta, r, dr, valid)`` where ``polar`` has shape
    ``(n_theta, n_r)``, row index = angle, column index = radius. Samples
    outside the image are NaN and flagged in ``valid``.
    """
    image = np.asarray(image, dtype=float)
    theta, r, dr = polar_grid(image.shape, center, n_r=n_r, n_theta=n_theta, r_max=r_max,
                              radial_bin_size=radial_bin_size)
    cx, cy = float(center[0]), float(center[1])
    ny, nx = image.shape

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = cx + np.outer(cos_t, r)
    y = cy + np.outer(sin_t, r)
    valid = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)

    coords = np.vstack([y.ravel(), x.ravel()])
    values = map_coordinates(image, coords, order=order, mode="constant",
                             cval=np.nan).reshape(x.shape)
    values[~valid] = np.nan
    return values, theta, r, dr, valid


def inverse_abel(polar, dr=1.0, method="hansenlaw"):
    """Inverse Abel transform of every constant-angle row along the radius."""
    polar = np.asarray(polar, dtype=float)
    valid = np.isfinite(polar)
    data = np.where(valid, polar, 0.0)
    method = str(method).lower()
    if method == "hansenlaw":
        from abel.hansenlaw import hansenlaw_transform

        out = np.asarray(hansenlaw_transform(data, dr=float(dr),
                                             direction="inverse"), dtype=float)
    elif method in ("dasch_three_point", "dasch_3", "dasch_two_point", "onion_peeling"):
        from abel.dasch import (onion_peeling_transform, three_point_transform,
                                two_point_transform)

        funcs = {
            "dasch_three_point": three_point_transform,
            "dasch_3": three_point_transform,
            "dasch_two_point": two_point_transform,
            "onion_peeling": onion_peeling_transform,
        }
        out = np.asarray(funcs[method](data, dr=float(dr)), dtype=float)
    else:
        raise ValueError(f"Unsupported inverse Abel method: {method!r}")
    out[~valid] = np.nan
    return out


# ---------------------------------------------------------------------------
# Radial enhancement filters (ported from circularization/enhance_polar_fits.py)
# ---------------------------------------------------------------------------


def robust_normalize(image, low_pct=1.0, high_pct=99.5):
    """Normalise finite values to 0..1 using robust percentiles."""
    image = np.asarray(image, dtype=float)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low = float(np.percentile(finite, low_pct))
    high = float(np.percentile(finite, high_pct))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def enhance_ridge(image, *, smooth_theta=1.0, smooth_r=0.8,
                  background_theta=8.0, background_r=25.0, ridge_sigma=1.2):
    """Suppress the broad background and highlight narrow radial ridges."""
    smoothed = gaussian_filter(image, sigma=(smooth_theta, smooth_r), mode="nearest")
    background = gaussian_filter(smoothed, sigma=(background_theta, background_r),
                                 mode="nearest")
    highpass = smoothed - background
    ridge_base = gaussian_filter1d(highpass, sigma=ridge_sigma, axis=1, mode="nearest")
    d2r = gaussian_filter1d(ridge_base, sigma=ridge_sigma, axis=1, order=2,
                            mode="nearest")
    ridge = np.clip(-d2r, 0.0, None)
    highpass_pos = np.clip(highpass, 0.0, None)
    return 0.75 * robust_normalize(ridge) + 0.25 * robust_normalize(highpass_pos)


def enhance_unsharp(image, *, smooth_theta=1.0, smooth_r=1.0,
                    blur_theta=6.0, blur_r=18.0):
    base = gaussian_filter(image, sigma=(smooth_theta, smooth_r), mode="nearest")
    blurred = gaussian_filter(base, sigma=(blur_theta, blur_r), mode="nearest")
    detail = base - blurred
    return np.clip(0.35 * robust_normalize(base) + 0.65 * robust_normalize(detail),
                   0.0, 1.0).astype(np.float32)


def enhance_clahe(image, *, smooth_theta=1.0, smooth_r=1.0, clip_limit=0.02,
                  kernel_theta=24, kernel_r=96):
    try:
        from skimage import exposure
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency is installed
        raise RuntimeError("scikit-image is required for CLAHE mode") from exc
    base = gaussian_filter(image, sigma=(smooth_theta, smooth_r), mode="nearest")
    norm = robust_normalize(base)
    enhanced = exposure.equalize_adapthist(norm, kernel_size=(int(kernel_theta),
                                                             int(kernel_r)),
                                           clip_limit=clip_limit)
    return np.asarray(enhanced, dtype=np.float32)


def enhance_bandpass(image, *, smooth_theta=1.0, smooth_r=0.8, low_r=2.0, high_r=20.0):
    """Radial difference-of-Gaussians: keep thin ridges, remove the background."""
    if not np.isfinite([low_r, high_r]).all() or not 0 <= low_r < high_r:
        raise ValueError("Bandpass widths must satisfy 0 ≤ low r < high r.")
    base = gaussian_filter(image, sigma=(smooth_theta, smooth_r), mode="nearest")
    lowpass_small = gaussian_filter1d(base, sigma=low_r, axis=1, mode="nearest") if low_r else base
    lowpass_large = gaussian_filter1d(base, sigma=high_r, axis=1, mode="nearest")
    band = np.clip(lowpass_small - lowpass_large, 0.0, None)
    return robust_normalize(band).astype(np.float32)


def enhance_polar(image, mode="bandpass", params=None):
    """Apply one radial enhancement mode to a polar image, preserving NaNs."""
    image = np.asarray(image, dtype=float)
    if mode in (None, "none", "off"):
        return image.copy()
    if mode not in ENHANCE_MODES:
        raise ValueError(f"Unknown enhancement mode: {mode!r}")
    merged = dict(DEFAULT_ENHANCE_PARAMS)
    if params:
        merged.update({k: v for k, v in params.items() if k in merged})
    valid = np.isfinite(image)
    data = np.where(valid, image, 0.0)
    keys = {
        "bandpass": ("smooth_theta", "smooth_r", "low_r", "high_r"),
        "ridge": ("smooth_theta", "smooth_r", "background_theta",
                  "background_r", "ridge_sigma"),
        "unsharp": ("smooth_theta", "smooth_r", "blur_theta", "blur_r"),
        "clahe": ("smooth_theta", "smooth_r", "clip_limit", "kernel_theta",
                  "kernel_r"),
    }[mode]
    kwargs = {key: merged[key] for key in keys}
    if mode == "bandpass":
        out = enhance_bandpass(data, **kwargs)
    elif mode == "ridge":
        out = enhance_ridge(data, **kwargs)
    elif mode == "unsharp":
        out = enhance_unsharp(data, **kwargs)
    else:
        out = enhance_clahe(data, **kwargs)
    out = np.asarray(out, dtype=float)
    out[~valid] = np.nan
    return out


def process_options(abel=True, abel_method="hansenlaw", enhancement=None, fit_filtered=False):
    """Build a normalised processing-options dictionary."""
    enhancement = dict(enhancement or {})
    placement = enhancement.get("placement", "off")
    if placement not in ENHANCE_PLACEMENTS:
        raise ValueError(f"Unknown enhancement placement: {placement!r}")
    mode = enhancement.get("mode", "bandpass")
    if mode not in ENHANCE_MODES:
        raise ValueError(f"Unknown enhancement mode: {mode!r}")
    params = dict(DEFAULT_ENHANCE_PARAMS)
    params.update(enhancement.get("params") or {})
    return {
        "abel": bool(abel),
        "abel_method": str(abel_method),
        "fit_filtered": bool(fit_filtered),
        "enhancement": {"placement": placement, "mode": mode, "params": params},
    }


def describe_processing(options):
    """One-line human-readable description used in provenance comments."""
    options = process_options(**{k: options[k] for k in ("abel", "abel_method", "enhancement")},
                              fit_filtered=options.get("fit_filtered", False))
    enh = options["enhancement"]
    parts = ["fit=point-sampled",
             f"abel={'none' if not options['abel'] else options['abel_method']}"]
    filter_name = "fit_filter" if options["fit_filtered"] else "display_filter"
    if enh["placement"] == "off" or enh["mode"] == "none":
        parts.append(f"{filter_name}=none")
    else:
        p = enh["params"]
        if enh["mode"] == "bandpass":
            detail = f"low_r={p['low_r']},high_r={p['high_r']}"
        elif enh["mode"] == "ridge":
            detail = f"background={p['background_theta']},{p['background_r']},sigma={p['ridge_sigma']}"
        elif enh["mode"] == "unsharp":
            detail = f"blur={p['blur_theta']},{p['blur_r']}"
        else:
            detail = f"clip={p['clip_limit']},kernel={p['kernel_theta']},{p['kernel_r']}"
        parts.append(f"{filter_name}={enh['mode']}({enh['placement']}:{detail})")
    return " ".join(parts)


def process_polar(polar, dr, *, abel=True, abel_method="hansenlaw", enhancement=None,
                  fit_filtered=False):
    """Return the fit image, display image and processing options."""
    options = process_options(abel=abel, abel_method=abel_method, enhancement=enhancement,
                              fit_filtered=fit_filtered)
    enh = options["enhancement"]
    raw = np.asarray(polar, dtype=float)
    fit = inverse_abel(raw, dr=dr, method=abel_method) if abel else raw
    placement = enh["placement"]
    if placement == "off" or enh["mode"] == "none":
        return fit, fit, options
    source = fit if placement == "post_abel" else raw
    display = enhance_polar(source, enh["mode"], enh["params"])
    if placement == "pre_abel" and abel:
        display = inverse_abel(display, dr=dr, method=abel_method)
    return (display if fit_filtered else fit), display, options


# ---------------------------------------------------------------------------
# Selection rectangles and masks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    """Rectangular region in polar data coordinates (radians, pixels)."""

    r0: float
    r1: float
    theta0: float = 0.0
    theta1: float = TWO_PI

    def normalised(self):
        r_lo, r_hi = sorted((float(self.r0), float(self.r1)))
        return Rect(r_lo, r_hi, float(self.theta0), float(self.theta1))

    def contains(self, r, theta):
        r_lo, r_hi = sorted((self.r0, self.r1))
        return (_angular_mask(np.asarray(theta), self.theta0, self.theta1)
                & (np.asarray(r) >= r_lo) & (np.asarray(r) <= r_hi))


def _angular_mask(theta, t0, t1):
    theta = np.asarray(theta, dtype=float)
    if (t1 - t0) >= TWO_PI - 1e-9:
        return np.ones_like(theta, dtype=bool)
    t0 %= TWO_PI
    t1 %= TWO_PI
    if t0 <= t1:
        return (theta >= t0) & (theta <= t1)
    return (theta >= t0) | (theta <= t1)


def selection_mask(theta, r, band, masks=()):
    """Boolean mask of pixels inside ``band`` and outside every mask."""
    band = band.normalised()
    theta = np.asarray(theta, dtype=float)
    r = np.asarray(r, dtype=float)
    tmask = _angular_mask(theta, band.theta0, band.theta1)
    rmask = (r >= band.r0) & (r <= band.r1)
    mask = np.outer(tmask, rmask)
    for rect in masks:
        rect = rect.normalised()
        mask &= ~np.outer(_angular_mask(theta, rect.theta0, rect.theta1),
                          (r >= rect.r0) & (r <= rect.r1))
    return mask


# ---------------------------------------------------------------------------
# Ridge fit
# ---------------------------------------------------------------------------


def parameter_names(n_terms, use_beta=True):
    names = ["intensity"]
    if use_beta:
        names.append("beta")
    names += ["offset", "sigma", "r0"]
    for n in range(1, int(n_terms) + 1):
        names += [f"A{n}", f"B{n}"]
    return names


def _model_vector(vector, r, theta, names, n_terms, use_beta, *, jacobian=False):
    p = dict(zip(names, vector))
    basis = {"r0": np.ones_like(theta)}
    for n in range(1, int(n_terms) + 1):
        basis[f"A{n}"] = np.sin(n * theta)
        basis[f"B{n}"] = np.cos(n * theta)
    rg = np.full_like(r, p["r0"], dtype=float)
    for key, term in basis.items():
        if key != "r0":
            rg += p[key] * term
    p2 = (3.0 * np.cos(theta) ** 2 - 1.0) / 2.0 if use_beta else 0.0
    angular = 1.0 + p.get("beta", 0.0) * p2
    scaled = (r - rg) / p["sigma"]
    gaussian = np.exp(-0.5 * scaled ** 2)
    signal = p["intensity"] * angular * gaussian
    if not jacobian:
        return signal + p["offset"]

    ridge_derivative = signal * scaled / p["sigma"]
    derivatives = {key: ridge_derivative * term for key, term in basis.items()}
    derivatives.update(intensity=angular * gaussian, offset=np.ones_like(r),
                       sigma=ridge_derivative * scaled)
    if use_beta:
        derivatives["beta"] = p["intensity"] * p2 * gaussian
    return np.column_stack([derivatives[name] for name in names])


def auto_guess(polar, theta, r, band, masks=(), n_terms=3, use_beta=True):
    """Initial parameter guess from the selected band's radial profile."""
    band = band.normalised()
    mask = selection_mask(theta, r, band, masks) & np.isfinite(polar)
    if not mask.any():
        raise ValueError("The selected band is empty")
    r_indices = np.flatnonzero((r >= band.r0) & (r <= band.r1))
    if r_indices.size < 3:
        raise ValueError("The selected band is narrower than three radial bins")
    selected = np.where(mask, polar, np.nan)[:, r_indices]
    with np.errstate(invalid="ignore"):
        values = np.nanmean(selected, axis=0)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("The selected band contains no finite data")
    values = np.where(finite, values, np.nanmin(values[finite]))
    peak = int(np.argmax(values))
    r0 = float(r[r_indices[peak]])
    baseline = float(np.median(np.concatenate([values[:max(1, peak // 4)],
                                               values[-max(1, peak // 4):]])))
    amplitude = float(values[peak] - baseline)
    if amplitude <= 0:
        amplitude = float(max(abs(values[peak]), 1e-12))
    half = baseline + amplitude / 2.0
    above = values >= half
    left = peak
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak
    while right < len(values) - 1 and above[right + 1]:
        right += 1
    width = max(float(r[r_indices[right]] - r[r_indices[left]]), float(r[1] - r[0]))
    sigma = max(width / 2.3548, float(r[1] - r[0]))
    guess = {
        "intensity": amplitude,
        "offset": baseline,
        "sigma": sigma,
        "r0": r0,
    }
    if use_beta:
        guess["beta"] = 0.0
    for n in range(1, int(n_terms) + 1):
        guess[f"A{n}"] = 0.0
        guess[f"B{n}"] = 0.0
    return guess


def _bounds_for(names, r):
    lower = []
    upper = []
    for name in names:
        if name == "intensity":
            lower.append(0.0)
            upper.append(np.inf)
        elif name == "beta":
            lower.append(-1.0)
            upper.append(2.0)
        elif name == "sigma":
            lower.append(float(max((r[1] - r[0]) * 1e-3, 1e-9)))
            upper.append(np.inf)
        else:
            lower.append(-np.inf)
            upper.append(np.inf)
    return np.array(lower), np.array(upper)


@dataclass
class RingFit:
    """One fitted ridge in the polar image."""

    params: dict
    sigmas: dict
    n_terms: int
    use_beta: bool
    band: Rect
    masks: list = field(default_factory=list)
    n_pixels: int = 0
    rms: float = float("nan")
    chi2_red: float = float("nan")
    converged: bool = False
    message: str = ""
    processing: dict = field(default_factory=dict)
    created: str = ""
    unconstrained: list[str] | None = None
    source_block: str | None = None

    def ridge_centers(self, theta):
        centers = np.full_like(theta, self.params["r0"], dtype=float)
        for n in range(1, self.n_terms + 1):
            centers += self.params[f"A{n}"] * np.sin(n * theta)
            centers += self.params[f"B{n}"] * np.cos(n * theta)
        return centers

    def export_issue(self):
        names = parameter_names(self.n_terms, self.use_beta)
        if any(not np.isfinite(self.params.get(name, np.nan)) for name in names):
            return "The fit contains missing or nonfinite parameters."
        if self.unconstrained is None:
            return "Refit this ring to check which parameters the data constrain."
        if self.unconstrained:
            terms = ", ".join(self.unconstrained)
            return f"The data cannot determine {terms}. Widen the band or reduce the harmonic order."
        return None

    def to_dict(self):
        return {
            "params": {k: float(v) for k, v in self.params.items()},
            "sigmas": {k: float(v) for k, v in self.sigmas.items()},
            "n_terms": int(self.n_terms),
            "use_beta": bool(self.use_beta),
            "band": [float(self.band.r0), float(self.band.r1),
                     float(self.band.theta0), float(self.band.theta1)],
            "masks": [[float(m.r0), float(m.r1), float(m.theta0), float(m.theta1)]
                      for m in self.masks],
            "n_pixels": int(self.n_pixels),
            "rms": float(self.rms),
            "chi2_red": float(self.chi2_red),
            "converged": bool(self.converged),
            "message": self.message,
            "processing": self.processing,
            "created": self.created,
            "unconstrained": self.unconstrained,
            "source_block": self.source_block,
        }

    @classmethod
    def from_dict(cls, data):
        band = Rect(*data["band"])
        masks = [Rect(*m) for m in data.get("masks", [])]
        return cls(params=dict(data["params"]), sigmas=dict(data["sigmas"]),
                   n_terms=int(data["n_terms"]), use_beta=bool(data["use_beta"]),
                   band=band, masks=masks, n_pixels=int(data.get("n_pixels", 0)),
                   rms=float(data.get("rms", np.nan)),
                   chi2_red=float(data.get("chi2_red", np.nan)),
                   converged=bool(data.get("converged", False)),
                   message=str(data.get("message", "")),
                   processing=dict(data.get("processing", {})),
                   created=str(data.get("created", "")),
                   unconstrained=data.get("unconstrained"), source_block=data.get("source_block"))


def fit_comparison(polar, theta, r, fit):
    """Crop the band and mask both the fit input and model in the same way."""
    band = selection_mask(theta, r, fit.band)
    ti = np.flatnonzero(band.any(axis=1))
    if ti.size:
        ti = np.arange(ti[0], ti[-1] + 1)
    ri = np.flatnonzero(band.any(axis=0))
    selection = np.ix_(ti, ri)
    angles, radii = theta[ti], r[ri]
    data = polar[selection]
    mask = selection_mask(theta, r, fit.band, fit.masks)[selection] & np.isfinite(data)
    angle_grid, radius_grid = np.meshgrid(angles, radii, indexing="ij")
    names = parameter_names(fit.n_terms, fit.use_beta)
    model = _model_vector([fit.params[name] for name in names], radius_grid, angle_grid,
                          names, fit.n_terms, fit.use_beta)
    return angles, radii, np.where(mask, data, np.nan), np.where(mask, model, np.nan)


def fit_ring(polar, theta, r, band, masks=(), n_terms=3, use_beta=True,
             iterative=True, max_nfev=None, processing=None):
    """Fit one Gaussian ridge with a trigonometric centre to the selected band."""
    polar = np.asarray(polar, dtype=float)
    theta = np.asarray(theta, dtype=float)
    r = np.asarray(r, dtype=float)
    mask = selection_mask(theta, r, band, masks) & np.isfinite(polar)
    if not mask.any():
        raise ValueError("The selected band contains no finite pixels")
    theta_grid, r_grid = np.meshgrid(theta, r, indexing="ij")
    tv = theta_grid[mask]
    rv = r_grid[mask]
    values = polar[mask]
    guess = auto_guess(polar, theta, r, band, masks, n_terms=n_terms,
                       use_beta=use_beta)

    def run(names, initial):
        for name in names:
            initial.setdefault(name, 0.0)
        x0 = np.array([initial[name] for name in names], dtype=float)
        lower, upper = _bounds_for(names, r)
        x0 = np.clip(x0, lower + 1e-12 * np.isfinite(lower).astype(float),
                     upper - 1e-12 * np.isfinite(upper).astype(float))
        n_fitted = sum(1 for name in names if name.startswith("A"))
        return least_squares(
            lambda vec: _model_vector(vec, rv, tv, names, n_fitted, use_beta) - values,
            x0,
            jac=lambda vec: _model_vector(vec, rv, tv, names, n_fitted, use_beta, jacobian=True),
            bounds=(lower, upper), method="trf", x_scale="jac",
            max_nfev=max_nfev,
        )

    if iterative and n_terms > 0:
        current = dict(guess)
        for k in range(0, n_terms + 1):
            names_k = parameter_names(k, use_beta=use_beta)
            for n in range(k + 1, n_terms + 1):
                current.pop(f"A{n}", None)
                current.pop(f"B{n}", None)
            solution = run(names_k, current)
            current = {name: float(value)
                       for name, value in zip(names_k, solution.x)}
        names = parameter_names(n_terms, use_beta=use_beta)
        result = run(names, current)
    else:
        names = parameter_names(n_terms, use_beta=use_beta)
        result = run(names, guess)

    n_params = len(names)
    residual = result.fun
    chi2 = float(np.sum(residual ** 2))
    dof = max(int(values.size) - n_params, 1)
    rms = float(np.sqrt(chi2 / values.size))
    jacobian = np.asarray(result.jac, dtype=float)
    norms = np.linalg.norm(jacobian, axis=0)
    scales = np.where(norms > 0, norms, 1.0)
    _, singular, vectors = np.linalg.svd(jacobian / scales, full_matrices=values.size < n_params)
    tolerance = singular[0] * max(jacobian.shape) * np.finfo(float).eps
    rank = np.count_nonzero(singular > tolerance)
    if rank < n_params:
        involved = np.linalg.norm(vectors[rank:], axis=0) > np.sqrt(np.finfo(float).eps)
        unconstrained = [name for name, present in zip(names, involved) if present]
        errors = np.full(n_params, np.nan)
    else:
        unconstrained = []
        errors = np.linalg.norm(vectors.T / singular, axis=1) * np.sqrt(chi2 / dof) / scales
    params = {name: float(value) for name, value in zip(names, result.x)}
    sigmas = {name: float(value) for name, value in zip(names, errors)}
    return RingFit(
        params=params, sigmas=sigmas, n_terms=int(n_terms),
        use_beta=bool(use_beta), band=band.normalised(),
        masks=[m.normalised() for m in masks], n_pixels=int(values.size),
        rms=rms, chi2_red=float(chi2 / dof), converged=bool(result.success),
        message=str(result.message), processing=processing or {},
        created=time.strftime("%Y-%m-%dT%H:%M:%S"), unconstrained=unconstrained,
    )


# ---------------------------------------------------------------------------
# Canonical file output
# ---------------------------------------------------------------------------


def _number(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Cannot write a non-finite fit parameter")
    return f"{value:.10g}"


def ring_block_text(fit, provenance=True, extra_comments=()):
    """Render one accepted ring as a Gascooke-compatible text block."""
    if issue := fit.export_issue():
        raise ValueError(issue)
    lines = []
    if provenance:
        band = fit.band
        m0, m1 = np.degrees(band.theta0), np.degrees(band.theta1)
        lines.append(
            f"# ring r0={_number(fit.params['r0'])} "
            f"band={band.r0:.4f}-{band.r1:.4f}px "
            f"theta={m0:.2f}-{m1:.2f}deg terms={fit.n_terms} "
            f"masks={len(fit.masks)} rms={fit.rms:.4g}")
        if fit.processing:
            lines.append(f"# processing: {describe_processing(fit.processing)}")
        lines.extend(str(comment) for comment in extra_comments)
    lines.append(EXTRACTED_HEADER)
    lines.append(f'"Gaussian Centre" {_number(fit.params["r0"])} '
                 f'+- {_number(fit.sigmas.get("r0", 0.0))}')
    for n in range(1, fit.n_terms + 1):
        for trig, prefix in (("sin", "A"), ("cos", "B")):
            key = f"{prefix}{n}"
            lines.append(f'"Trig Term {trig}({n}*theta)" '
                         f'{_number(fit.params[key])} '
                         f'+- {_number(fit.sigmas.get(key, 0.0))}')
    return "\n".join(lines) + "\n"


def validate_block_text(text):
    """Parse one rendered ring block and return its radius (raises on failure)."""
    blocks = parse_extracted_text(text)
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly one fit block, got {len(blocks)}")
    return float(blocks[0]["radius"])


def append_ring_block(path, text):
    """Append one validated block to a trig-fit file, creating it if needed."""
    validate_block_text(text)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        prefix = "\n"
    else:
        prefix = ""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(prefix + text)
    return path


def existing_radii(path):
    """Radii of rings already present in a trig-fit file (empty if absent)."""
    path = Path(path)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if not any(line.strip() and not line.strip().startswith("#")
               for line in text.splitlines()):
        return []
    return [float(block["radius"]) for block in
            parse_extracted_text(text, source=str(path))]


def load_accepted_rings(path):
    """Load measurements and preserve the source blocks for a later rebuild."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not any(line.strip() and not line.lstrip().startswith("#") for line in lines):
        return []
    starts = []
    for index, line in enumerate(lines):
        if line.strip() == EXTRACTED_HEADER:
            start = index
            while start > 0 and lines[start - 1].lstrip().startswith("#"):
                start -= 1
            starts.append(start)
    if not starts:
        raise ValueError(f"No fit blocks found in {path}")
    starts[0] = 0
    rings = []
    for start, end in zip(starts, [*starts[1:], len(lines)]):
        text = "".join(lines[start:end])
        block, = parse_extracted_text(text, source=str(path))
        if "radius" not in block or not np.isfinite(block["radius"]):
            raise ValueError(f"Missing or invalid Gaussian Centre in {path}")
        terms = [key for key in block if re.fullmatch(r"[AB]\d+", key)]
        rms = re.search(r"^# ring .*\brms=([\d.eE+-]+)\s*$", text, re.MULTILINE)
        rings.append(RingFit(
            params={"r0": block["radius"], **{key: block[key] for key in terms}},
            sigmas={"r0": block["radius_sigma"], **{key: block[f"{key}_sigma"] for key in terms}},
            n_terms=max((int(key[1:]) for key in terms), default=0),
            use_beta=False, band=Rect(block["radius"], block["radius"]),
            rms=float(rms[1]) if rms else float("nan"), source_block=text,
        ))
    return rings


def write_ring_blocks(path, blocks):
    """Rewrite a trig-fit file from a sequence of rendered blocks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(block.rstrip("\n") + "\n" for block in blocks)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# Rings -> a_n, b_n reduction (uses the existing circularisation models)
# ---------------------------------------------------------------------------


def reduce_rings(fits, model="poly", order=2):
    """Reduce per-ring A_n(r), B_n(r) to a_n, b_n slopes through the origin."""
    from .circularization import (build_all_trig_interpolators, build_fit_dataset,
                                  fit_all_trig_polynomials)

    blocks = []
    for fit in fits:
        block = {"radius": fit.params["r0"],
                 "radius_sigma": fit.sigmas.get("r0", 0.0)}
        for n in range(1, fit.n_terms + 1):
            for prefix in ("A", "B"):
                key = f"{prefix}{n}"
                if key in fit.params:
                    block[key] = fit.params[key]
                    block[f"{key}_sigma"] = fit.sigmas.get(key, 0.0)
        blocks.append(block)
    dataset = build_fit_dataset(blocks)
    if model == "poly":
        fit_params, fit_meta = fit_all_trig_polynomials(dataset, order=int(order))
        slopes = {}
        for key, value in fit_params.items():
            coeffs = np.asarray(value["coeffs"], dtype=float)
            slopes[key] = float(coeffs[-1])
            cov = fit_meta.get(key, {}).get("cov")
            slopes[f"{key}_sigma"] = (float(np.sqrt(max(cov[-1, -1], 0.0)))
                                      if cov is not None else float("nan"))
        return {"dataset": dataset, "fit_params": fit_params,
                "fit_meta": fit_meta, "a": slopes, "model": "poly",
                "order": int(order)}
    if model == "interp":
        fit_params, fit_meta = build_all_trig_interpolators(dataset)
        slopes = {}
        for key, value in fit_params.items():
            slopes[key] = float(value["dinterp"](0.0))
            slopes[f"{key}_sigma"] = float("nan")
        return {"dataset": dataset, "fit_params": fit_params,
                "fit_meta": fit_meta, "a": slopes, "model": "interp",
                "order": None}
    raise ValueError("model must be 'poly' or 'interp'")


def radial_model_curves(reduced):
    """Sample the Circularize models from the origin to the outermost fitted ring."""
    from .circularization import eval_radial_model

    radii = np.linspace(0, float(np.max(reduced["dataset"]["radii"])), 500)
    curves = {key: eval_radial_model(model, radii)[0]
              for key, model in reduced["fit_params"].items()}
    return radii, curves


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

SESSION_SCHEMA = "vmi-trig-calibration/2"


def save_session(path, session):
    """Atomically write the panel session JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(session)
    payload.setdefault("schema", SESSION_SCHEMA)
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True),
                         encoding="utf-8")
    temporary.replace(path)
    return path


def load_session(path):
    """Load a panel session JSON and check its schema."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") == "vmi-trig-calibration/1":
        raise ValueError(f"{path} uses the previous fitting pipeline. Refit the rings "
                         "and export them to a new calibration file.")
    if payload.get("schema") != SESSION_SCHEMA:
        raise ValueError(f"Unsupported session schema in {path}: "
                         f"{payload.get('schema')!r}")
    return payload


def session_rings(session):
    """RingFit objects stored in a session payload."""
    return [RingFit.from_dict(item) for item in session.get("rings", [])]
