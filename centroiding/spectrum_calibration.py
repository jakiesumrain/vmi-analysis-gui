"""Xe 4d calibration from fitted peak centers in squared-radius coordinates."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks

# Gas-phase ionization potentials, not condensed-phase XPS binding energies.
# Elettra PES reference table, citing King et al., J. Phys. B 10 (1977) 2479:
# https://www.elettra.eu/lightsources/elettra/elettra-beamlines/gas-phase/useful-reference-datafor-pes.html
XE_4D_5_2_IP_EV = 67.548
XE_4D_3_2_IP_EV = 69.537


@dataclass(frozen=True)
class CalibrationSettings:
    alpha: float | None = None
    fit_xe_4d: bool = False
    fit_min_px2: float | None = None
    fit_max_px2: float | None = None

    @classmethod
    def from_config(cls, config):
        def optional_number(key):
            value = config.get(key)
            if value is None or value == "":
                return None
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            return value

        alpha = optional_number("alpha")
        lo, hi = optional_number("fit_min_px2"), optional_number("fit_max_px2")
        if alpha is not None and alpha <= 0:
            raise ValueError("alpha must be positive (eV/pixel²), or left empty")
        if any(value is not None and value < 0 for value in (lo, hi)):
            raise ValueError("Xe fit limits must be nonnegative pixel² values, or left empty")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("Xe fit minimum must be smaller than its maximum")
        enabled = config.get("fit_xe_4d", False)
        if not isinstance(enabled, bool):
            raise ValueError("fit_xe_4d must be true or false")
        return cls(alpha, enabled, lo, hi)


@dataclass(frozen=True)
class XeCalibrationFit:
    q_3_2: float
    q_5_2: float
    alpha: float
    photon_energy_ev: float
    alpha_std: float
    photon_energy_std_ev: float
    fit_q: np.ndarray
    fit_density_per_px2: np.ndarray


def scale_spectrum(q, density_per_px2, alpha=None):
    """Convert dN/d(r²) to dN/dE, preserving the integrated electron yield."""
    settings = CalibrationSettings.from_config({"alpha": alpha})
    factor = settings.alpha if settings.alpha is not None else 1.0
    return np.asarray(q) * factor, np.asarray(density_per_px2) / factor


def fit_xe_4d(q, density_per_px2, fit_min_px2=None, fit_max_px2=None):
    """Fit two Gaussians + linear background in q=r², then solve E=hv-IP.

    The two most prominent resolved peaks in the optional search window seed
    the fit. Their amplitudes and widths are independent. The lower-q peak
    is assigned to 4d3/2 and the higher-q peak to 4d5/2. This assignment must
    be appropriate for the experiment; use a window to exclude other lines.
    Reported errors propagate the formal least-squares center covariance;
    they exclude centering, model, reference-energy and detector systematics.
    """
    limits = CalibrationSettings.from_config(
        {"fit_min_px2": fit_min_px2, "fit_max_px2": fit_max_px2})
    q, y = np.asarray(q, dtype=float), np.asarray(density_per_px2, dtype=float)
    if q.ndim != 1 or y.shape != q.shape:
        raise ValueError("Expected matching one-dimensional spectrum arrays")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(y)):
        raise ValueError("Spectrum contains nonfinite values")
    if np.any(q < 0) or np.any(np.diff(q) <= 0):
        raise ValueError("Squared-radius coordinates must be nonnegative and increasing")
    selected = np.ones(q.size, dtype=bool)
    if limits.fit_min_px2 is not None:
        selected &= q >= limits.fit_min_px2
    if limits.fit_max_px2 is not None:
        selected &= q <= limits.fit_max_px2
    x, y = q[selected], y[selected]
    if x.size < 12 or np.ptp(y) <= 0:
        raise ValueError("Need at least 12 samples and two resolved peaks in the fit window")

    smooth = gaussian_filter1d(y, 0.6)
    peaks, info = find_peaks(smooth, prominence=0.08 * np.ptp(smooth), distance=3)
    if peaks.size < 2:
        raise ValueError("Could not find two prominent Xe peaks; adjust the pixel² fit window")
    seeds = np.sort(peaks[np.argsort(info["prominences"])[-2:]])
    c1, c2 = x[seeds]
    separation = c2 - c1
    # Fit a local neighborhood, retaining background on both sides rather
    # than fitting unrelated spectral structure elsewhere in the image.
    local = (x >= c1 - 0.8 * separation) & (x <= c2 + 0.8 * separation)
    x, y = x[local], y[local]
    if x.size < 12:
        raise ValueError("The doublet is undersampled; need at least 12 local samples")
    offset, scale = c1, separation
    t = (x - offset) / scale
    amplitude_scale = np.ptp(y)
    baseline = np.percentile(y, 15)
    z = (y - baseline) / amplitude_scale
    step = np.median(np.diff(t))
    if step >= 0.25:
        raise ValueError("The doublet is not sufficiently resolved for calibration")

    def model(t, a1, mu1, s1, a2, mu2, s2, b0, b1):
        return (a1 * np.exp(-0.5 * ((t - mu1) / s1)**2)
                + a2 * np.exp(-0.5 * ((t - mu2) / s2)**2) + b0 + b1 * t)

    width = max(0.08, step)
    lower = [0, -0.25, step / 3, 0, 0.75, step / 3, -np.inf, -np.inf]
    upper = [np.inf, 0.25, 0.4, np.inf, 1.25, 0.4, np.inf, np.inf]
    initial = [max(np.interp(0, t, z), 0.1), 0, width,
               max(np.interp(1, t, z), 0.1), 1, width, 0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            params, covariance = curve_fit(model, t, z, p0=initial,
                                           bounds=(lower, upper), max_nfev=10000)
        except (RuntimeError, OptimizeWarning) as exc:
            raise ValueError(f"Doublet fit did not converge: {exc}") from exc
    residual = z - model(t, *params)
    noise = np.sqrt(np.mean(residual**2))
    if (min(params[0], params[3]) < 5 * noise
            or params[4] - params[1] <= 2 * max(params[2], params[5])
            or not np.all(np.isfinite(covariance))
            or any(abs(params[i] - seed) >= 0.245 for i, seed in ((1, 0), (4, 1)))):
        raise ValueError("Doublet fit is ambiguous or poorly resolved; narrow the pixel² window")

    q1, q2 = offset + scale * params[[1, 4]]
    centers_cov = covariance[np.ix_([1, 4], [1, 4])] * scale**2
    if np.max(np.sqrt(np.maximum(np.diag(centers_cov), 0))) >= (q2 - q1) / 5:
        raise ValueError("Fitted peak centers are too uncertain for calibration")
    # alpha*q1 = hv-IP(4d3/2), alpha*q2 = hv-IP(4d5/2).
    delta_ip = XE_4D_3_2_IP_EV - XE_4D_5_2_IP_EV
    alpha = delta_ip / (q2 - q1)
    hv = alpha * q1 + XE_4D_3_2_IP_EV
    grad_alpha = np.array([1, -1]) * alpha / (q2 - q1)
    grad_hv = np.array([q2, -q1]) * alpha / (q2 - q1)
    alpha_std = np.sqrt(max(float(grad_alpha @ centers_cov @ grad_alpha), 0))
    hv_std = np.sqrt(max(float(grad_hv @ centers_cov @ grad_hv), 0))
    fit_q = np.linspace(x[0], x[-1], 600)
    fit_density = baseline + amplitude_scale * model((fit_q - offset) / scale, *params)
    return XeCalibrationFit(float(q1), float(q2), float(alpha), float(hv),
                            float(alpha_std), float(hv_std), fit_q, fit_density)
