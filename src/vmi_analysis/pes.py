"""Extract photoelectron spectra from VMI images and fit spectral peaks."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import abel
import numpy as np
from scipy.ndimage import gaussian_filter1d, rotate
from scipy.optimize import brentq, least_squares
from scipy.signal import find_peaks
from scipy.special import voigt_profile

from .centroiding.plot_data import save_plot_data
from .centroiding.spectrum_calibration import scale_spectrum
from .imaging.centering import brightest_pixel

_FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
ABEL_METHODS = {
    "hansenlaw": "Hansen–Law",
    "three_point": "Three-point",
    "two_point": "Two-point",
    "onion_peeling": "Onion peeling",
    "direct": "Direct",
    "basex": "BASEX",
    "daun": "Daun",
}
BETA_ORDERS = (0, 2, 4, 6, 8, 10)
BASELINE_PARAMETERS = {"None": 0, "Constant": 1, "Linear": 2}


@dataclass(frozen=True)
class Geometry:
    center: tuple[float, float] | None = None  # x, y in the image grid
    angle: float = 0.0
    resolution: float = 1.0  # image pixels per detector pixel

    def __post_init__(self):
        if not np.isfinite(self.angle) or not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("Enter a finite rotation and a positive resolution factor.")
        if self.center is not None and (len(self.center) != 2 or not np.isfinite(self.center).all()):
            raise ValueError("Enter finite Center X and Center Y values.")


def load_image(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".txt":
        image = np.loadtxt(path, ndmin=2)
    else:
        raise ValueError("Choose a two-dimensional .npy or .txt image.")
    if image.ndim != 2 or min(image.shape) < 5 or image.dtype.kind not in "iuf" or not np.isfinite(image).all():
        raise ValueError("Expected a finite, real 2D image with at least five pixels per side.")
    return image


def extract_spectrum(image, geometry=Geometry(), alpha=None, *, method="hansenlaw"):
    """Return detector r², calibrated x, reference intensity, and the chosen center."""
    return _analyze_image(image, geometry, alpha, method, 0)[:4]


def _analyze_image(image, geometry, alpha, method, beta_order):
    if geometry.center is None:
        y, x = brightest_pixel(image)
    else:
        x, y = geometry.center
    ny, nx = image.shape
    # Round down to the detector area covered after centering and rotation.
    radius_limit = min(x, y, nx - 1 - x, ny - 1 - y) - 1
    if radius_limit < 1:
        raise ValueError("The selected center leaves too little detector coverage for a spectrum.")
    # Center on a pixel in both axes, including after a quarter-turn.
    odd_image = image[:ny - (ny % 2 == 0), :nx - (nx % 2 == 0)]
    centered = abel.tools.center.center_image(odd_image.astype(float), (y, x))
    angle = geometry.angle % 360
    if angle % 90 == 0:
        centered = np.rot90(centered, int(angle // 90))
    else:
        centered = rotate(centered, angle, reshape=False, order=1)
    # Matrix-based methods may write basis data unless given an explicit directory policy.
    options = {"basis_dir": None} if method in ("three_point", "two_point", "onion_peeling", "basex", "daun") else {}
    inverted = abel.Transform(centered, method=method, direction="inverse", origin="none",
                              transform_options=options).transform
    radius, speed = abel.tools.vmi.angular_integration_3D(inverted)
    radius_limit = min(radius_limit, min(centered.shape) / 2 - 1)
    inside = (radius > 0) & (radius <= radius_limit)
    q_grid, intensity = abel.tools.vmi.toPES(radius[inside], speed[inside].copy(), 1.0)
    q_detector = q_grid / geometry.resolution**2
    energy, intensity = scale_spectrum(q_detector, intensity, alpha)
    if not energy.size or not np.isfinite(energy).all() or not np.isfinite(intensity).all():
        raise ValueError("Abel inversion did not produce a finite spectrum.")
    angular = _angular_moments(inverted, int(radius_limit), beta_order, geometry.resolution, alpha) if beta_order else None
    return q_detector, energy, intensity, (float(x), float(y)), angular


def _angular_moments(image, rmax, order, resolution, alpha):
    distribution = abel.tools.vmi.Distributions(rmax=rmax, order=order)(image)
    radius = distribution.r[1:]
    # Convert Legendre coefficients to differential moments in r². All orders
    # share the spherical Jacobian 4πr² and the conversion dr/d(r²) = 1/(2r).
    moments = 2 * np.pi * radius * distribution.harmonics()[:, 1:]
    moments[:, ~distribution.valid[1:]] = np.nan
    x, _ = scale_spectrum((radius / resolution) ** 2, moments[0], alpha)
    return x, moments


def anisotropy_values(moments):
    """Return β_L = I_L / I_0 where the isotropic moment is positive."""
    isotropic = moments[0]
    return np.divide(moments[1:], isotropic, out=np.full_like(moments[1:], np.nan),
                     where=np.isfinite(isotropic) & (isotropic > 0))


def average_anisotropy(x, moments, center, fwhm):
    """Integrate the angular moments over a complete, sampled FWHM window."""
    low, high = center - fwhm / 2, center + fwhm / 2
    missing = [None] * (moments.shape[0] - 1)
    if x.size < 2 or low < x[0] or high > x[-1] or low >= high:
        return missing
    if np.count_nonzero((x >= low) & (x <= high)) < 2:
        return missing
    start = np.searchsorted(x, low, side="right") - 1
    stop = np.searchsorted(x, high) + 1
    x, moments = x[start:stop], moments[:, start:stop]
    if not np.isfinite(moments).all():
        return missing
    window = np.r_[low, x[(x > low) & (x < high)], high]
    integrated = np.trapezoid([np.interp(window, x, moment) for moment in moments], window, axis=1)
    if integrated[0] <= 0 or not np.isfinite(integrated).all():
        return missing
    return (integrated[1:] / integrated[0]).tolist()


@dataclass(frozen=True)
class PeakGuess:
    profile: str
    center: float
    gaussian_fwhm: float
    lorentzian_fwhm: float = 0.0


def _estimate_peaks(x, y, baseline, profile):
    """Seed resolved positive peaks using prominence and half-prominence widths."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    if baseline == "Linear":
        y = y - np.interp(x, x[[0, -1]], y[[0, -1]])
    smooth = gaussian_filter1d(y, 0.6)
    curvature = np.diff(y, n=2)
    noise = np.median(np.abs(curvature - np.median(curvature))) / (0.67449 * np.sqrt(6))
    prominence = max(0.08 * np.ptp(smooth), 4 * noise, np.sqrt(np.finfo(float).eps))
    indices, info = find_peaks(smooth, prominence=prominence, distance=3, width=1)
    if not indices.size:
        raise ValueError("No prominent peaks found. Adjust the fit range or enter peak estimates.")

    # Keep the fit overdetermined and limit the number of automatic components.
    parameter_count = 4 if profile == "Voigt" else 3
    baseline_count = BASELINE_PARAMETERS[baseline]
    count = min(6, (np.unique(x).size - baseline_count - 1) // parameter_count)
    if count < 1:
        raise ValueError("Select a wider fit range with more spectrum samples.")
    strongest = np.sort(np.argsort(info["prominences"])[-count:])
    samples = np.arange(x.size)
    widths = (np.interp(info["right_ips"][strongest], samples, x)
              - np.interp(info["left_ips"][strongest], samples, x))
    if profile == "Voigt":
        return [PeakGuess(profile, x[i], 0.7 * width, 0.5 * width)
                for i, width in zip(indices[strongest], widths)]
    return [PeakGuess(profile, x[i], width) for i, width in zip(indices[strongest], widths)]


def fit_peaks(x, y, region, guesses=(), baseline="Constant", *, auto_profile="Gaussian"):
    """Fit peaks and a baseline to the native spectrum samples in the region."""
    low, high = region
    if not np.isfinite(region).all() or low >= high:
        raise ValueError("Enter an increasing, finite fit range.")
    if baseline not in BASELINE_PARAMETERS:
        raise ValueError("Choose None, Constant, or Linear for the baseline.")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Expected matching one-dimensional spectrum coordinates and intensities.")
    inside = np.isfinite(x) & np.isfinite(y) & (x >= low) & (x <= high)
    x, y = x[inside], y[inside]
    span, midpoint = high - low, low + (high - low) / 2
    yscale = float(np.max(np.abs(y))) if y.size else 0.0
    if not np.isfinite(span) or yscale == 0 or np.unique(x).size < 3:
        raise ValueError("The fit range contains too few distinct samples or no signal.")
    t, z = (x - midpoint) / span, y / yscale
    if not guesses:
        if auto_profile not in ("Gaussian", "Voigt"):
            raise ValueError("Choose a Gaussian or Voigt profile.")
        guesses = _estimate_peaks(x, z, baseline, auto_profile)
    scale = _FWHM_PER_SIGMA
    nbase = BASELINE_PARAMETERS[baseline]
    level = float(np.median(z)) if nbase else 0.0
    start = [level, 0.0][:nbase]
    lower, upper = [-np.inf] * nbase, [np.inf] * nbase
    slices = []
    for guess in guesses:
        if (guess.profile not in ("Gaussian", "Voigt") or not low <= guess.center <= high
                or not np.isfinite(guess.gaussian_fwhm) or not 0 < guess.gaussian_fwhm <= span):
            raise ValueError("Peak centers must lie inside the fit range and widths must be positive and within its span.")
        mu, sigma = (guess.center - midpoint) / span, guess.gaussian_fwhm / span / scale
        if sigma < 1e-8:
            raise ValueError("The Gaussian FWHM is too small for the selected fit range.")
        gamma = guess.lorentzian_fwhm / span / 2 if guess.profile == "Voigt" else 0.0
        if not np.isfinite(gamma) or not 0 <= gamma <= 0.5:
            raise ValueError("Voigt Lorentzian FWHM must be between zero and the fit-range span.")
        height = max(float(z[np.argmin(abs(t - mu))] - level), 0.01)
        offset = len(start)
        start.extend([height / float(voigt_profile(0, sigma, gamma)), mu, sigma])
        lower.extend([0.0, -0.5, min(1e-6, sigma / 2)])
        upper.extend([np.inf, 0.5, 1 / scale])
        if guess.profile == "Voigt":
            start.append(gamma)
            lower.append(0.0)
            upper.append(0.5)
        slices.append(slice(offset, len(start)))
    if np.unique(x).size <= len(start):
        raise ValueError(f"Select a region with more than {len(start)} distinct spectrum samples.")

    def curves(params):
        background = np.full(t.shape, params[0] if nbase else 0.0)
        if nbase == 2:
            background += params[1] * t
        peaks = [p[0] * voigt_profile(t - p[1], p[2], p[3] if len(p) == 4 else 0.0)
                 for p in (params[part] for part in slices)]
        return background, peaks

    def residual(params):
        background, peaks = curves(params)
        return background + np.sum(peaks, axis=0) - z

    fit = least_squares(residual, start, bounds=(lower, upper), max_nfev=1500)
    if not fit.success:
        raise ValueError(f"Peak fit did not converge. {fit.message}")
    background, components = curves(fit.x)
    total = (background + np.sum(components, axis=0)) * yscale
    rows = []
    for index, (guess, part) in enumerate(zip(guesses, slices), 1):
        area, mu, sigma, *wing = fit.x[part]
        gamma = wing[0] if wing else 0.0
        half = voigt_profile(0, sigma, gamma) / 2
        fwhm = 2 * brentq(lambda r: voigt_profile(r, sigma, gamma) - half, 0, 3, xtol=1e-14)
        rows.append(dict(peak=index, profile=guess.profile, center=midpoint + mu * span,
                         fwhm=fwhm * span, area=area * yscale * span,
                         gaussian_fwhm=sigma * scale * span, lorentzian_fwhm=2 * gamma * span,
                         bound=bool(np.any(fit.active_mask[part]))))
    return x, total, background * yscale, np.asarray(components) * yscale, y - total, rows


def evaluate_fit_for_plot(x, background, peaks):
    """Evaluate the fitted model on a dense grid, including around narrow peaks."""
    first, last = np.argmin(x), np.argmax(x)
    low, high = x[first], x[last]
    grids = [x, np.linspace(low, high, 1000)]
    for peak in peaks:
        center, radius = peak["center"], 3 * peak["fwhm"]
        left, right = np.clip([center - radius, center + radius], low, high)
        grids.append(np.linspace(left, right, 101))
    grid = np.unique(np.concatenate(grids))
    baseline = np.interp(grid, [low, high], background[[first, last]])
    components = np.asarray([
        peak["area"] * voigt_profile(grid - peak["center"],
                                    peak["gaussian_fwhm"] / _FWHM_PER_SIGMA,
                                    peak["lorentzian_fwhm"] / 2)
        for peak in peaks
    ])
    return grid, baseline + components.sum(axis=0), baseline, components


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate photoelectron spectra from 2D VMI images")
    parser.add_argument("--image", nargs=5, metavar=("PATH", "X", "Y", "ROTATION", "RESOLUTION"),
                        action="append", required=True, help="Repeat per image. Use auto auto for the brightest pixel.")
    parser.add_argument("--alpha", type=float, help="Energy calibration in eV per squared detector pixel")
    parser.add_argument("--abel-method", choices=ABEL_METHODS, default="hansenlaw")
    parser.add_argument("--beta-order", type=int, choices=BETA_ORDERS, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Output spectrum bundle (.npz)")
    args = parser.parse_args(argv)
    if args.alpha is not None and (not np.isfinite(args.alpha) or args.alpha <= 0):
        parser.error("alpha must be finite and positive")
    if args.output.suffix.lower() != ".npz":
        parser.error("output must end in .npz")
    records, arrays = [], {}
    for filename, x, y, angle, resolution in args.image:
        try:
            geometry = Geometry(None if x == y == "auto" else (float(x), float(y)), float(angle), float(resolution))
            q, energy, intensity, center, angular = _analyze_image(
                load_image(filename), geometry, args.alpha, args.abel_method, args.beta_order)
        except (OSError, ValueError, EOFError) as exc:
            print(f"[PES] {filename}: {exc}")
            continue
        i = len(records)
        records.append(dict(path=filename, center=center, rotation=geometry.angle, resolution=geometry.resolution))
        arrays.update({f"r2_{i}": q, f"x_{i}": energy, f"y_{i}": intensity})
        if angular is not None:
            arrays.update({f"beta_x_{i}": angular[0], f"moments_{i}": angular[1]})
        print(f"[PES] {filename} · center X={center[0]:g}, Y={center[1]:g} · {len(energy)} spectrum samples")
    if not records:
        raise SystemExit("[PES] No spectra generated.")
    save_plot_data(args.output, {"alpha": args.alpha, "images": records,
                                "abel_method": args.abel_method, "beta_order": args.beta_order}, **arrays)


if __name__ == "__main__":
    main()
