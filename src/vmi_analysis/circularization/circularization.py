import argparse
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def poly_func_no_const(r, *coeffs):
    """
    Evaluate a polynomial with no constant term.

    coeffs are ordered from highest power to lowest power:
    [a_k, a_{k-1}, ..., a_1], corresponding to
    a_k * r^k + ... + a_1 * r.
    """
    return np.polyval(list(coeffs) + [0.0], r)


def poly_derivative_no_const(r, *coeffs):
    """
    Evaluate the radial derivative of a polynomial with no constant term.

    If p(r) = a_k r^k + ... + a_1 r, returns dp/dr.
    """
    if len(coeffs) == 0:
        return np.zeros_like(r, dtype=float)

    powers = np.arange(len(coeffs), 0, -1, dtype=float)
    deriv_coeffs = np.asarray(coeffs, dtype=float) * powers

    if len(deriv_coeffs) == 1:
        return np.full_like(r, deriv_coeffs[0], dtype=float)

    return np.polyval(deriv_coeffs, r)


def eval_radial_model(model, r):
    """
    Evaluate a radial model and its derivative at radius r.

    Supported models:
    - {"kind": "poly", "coeffs": ...}
    - {"kind": "interp", "interp": ..., "dinterp": ..., "r_min": ..., "r_max": ...}
    """
    kind = model["kind"]

    if kind == "poly":
        coeffs = model["coeffs"]
        return (
            poly_func_no_const(r, *coeffs),
            poly_derivative_no_const(r, *coeffs),
        )

    if kind == "interp":
        interp = model["interp"]
        dinterp = model["dinterp"]
        r_min = model["r_min"]
        r_max = model["r_max"]
        r_eval = np.clip(r, r_min, r_max)
        return interp(r_eval), dinterp(r_eval)

    raise ValueError(f"Unknown radial model kind: {kind}")


def eval_reassignment_model(model, r_fixed):
    """
    Evaluate the inverse radial reassignment map and its derivative.

    Returns r_circ and dr_circ/dr_fixed.

    Supported models:
    - None: identity map
    - {"kind": "identity"}
    - {"kind": "interp", "interp": ..., "dinterp": ..., "r_min": ..., "r_max": ...}
    """
    if model is None:
        return r_fixed, np.ones_like(r_fixed, dtype=float)

    kind = model["kind"]
    if kind == "identity":
        return r_fixed, np.ones_like(r_fixed, dtype=float)

    if kind == "interp":
        interp = model["interp"]
        dinterp = model["dinterp"]
        r_min = model["r_min"]
        r_max = model["r_max"]
        r_fixed = np.asarray(r_fixed, dtype=float)
        r_circ = np.empty_like(r_fixed, dtype=float)
        dr_circ_dr_fixed = np.empty_like(r_fixed, dtype=float)

        inside_mask = (r_fixed >= r_min) & (r_fixed <= r_max)
        below_mask = r_fixed < r_min
        above_mask = r_fixed > r_max

        if np.any(inside_mask):
            r_eval = r_fixed[inside_mask]
            r_circ[inside_mask] = interp(r_eval)
            dr_circ_dr_fixed[inside_mask] = dinterp(r_eval)

        if np.any(below_mask):
            r_circ[below_mask] = r_fixed[below_mask]
            dr_circ_dr_fixed[below_mask] = 1.0

        if np.any(above_mask):
            r_circ[above_mask] = r_fixed[above_mask]
            dr_circ_dr_fixed[above_mask] = 1.0

        return r_circ, dr_circ_dr_fixed

    raise ValueError(f"Unknown reassignment model kind: {kind}")


def build_reassignment_interpolator(r_circ_points, r_fixed_points):
    """
    Build the inverse reassignment map r_fixed -> r_circ.

    Parameters
    ----------
    r_circ_points : array-like
        Radii after circularization.
    r_fixed_points : array-like
        Desired assigned radii for those same features.
    """
    r_circ_points = np.asarray(r_circ_points, dtype=float)
    r_fixed_points = np.asarray(r_fixed_points, dtype=float)

    if r_circ_points.ndim != 1 or r_fixed_points.ndim != 1:
        raise ValueError("r_circ_points and r_fixed_points must be 1D arrays")
    if len(r_circ_points) != len(r_fixed_points):
        raise ValueError("r_circ_points and r_fixed_points must have the same length")
    if len(r_circ_points) < 2:
        raise ValueError("Need at least two points to build a reassignment map")

    sort_idx = np.argsort(r_fixed_points)
    r_fixed_sorted = r_fixed_points[sort_idx]
    r_circ_sorted = r_circ_points[sort_idx]

    if r_fixed_sorted[0] > 0.0 or r_circ_sorted[0] > 0.0:
        r_fixed_sorted = np.concatenate(([0.0], r_fixed_sorted))
        r_circ_sorted = np.concatenate(([0.0], r_circ_sorted))

    if np.any(np.diff(r_fixed_sorted) <= 0):
        raise ValueError("r_fixed_points must be strictly increasing")

    interp = PchipInterpolator(r_fixed_sorted, r_circ_sorted, extrapolate=True)
    dinterp = interp.derivative()

    return {
        "kind": "interp",
        "interp": interp,
        "dinterp": dinterp,
        "r_min": float(np.min(r_fixed_sorted)),
        "r_max": float(np.max(r_fixed_sorted)),
    }


def parse_reassignment_file(path):
    """
    Read a two-column text file: r_circ  r_fixed
    One pair per line, comments allowed with '#'.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        rows.append((float(parts[0]), float(parts[1])))

    if len(rows) < 2:
        raise ValueError(f"Need at least two reassignment pairs in {path}")

    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def plot_reassignment_map(reassignment_model, r_circ_points, r_fixed_points, output_dir):
    """
    Plot the inverse reassignment map r_fixed -> r_circ and save it.
    """
    if plt is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r_fixed_min = float(np.min(r_fixed_points))
    r_fixed_max = float(np.max(r_fixed_points))
    r_fixed_plot = np.linspace(r_fixed_min, r_fixed_max, 500)
    r_circ_plot, _ = eval_reassignment_model(reassignment_model, r_fixed_plot)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_fixed_plot, r_circ_plot, label=r"$r_{\mathrm{fixed}} \to r_{\mathrm{circ}}$")
    ax.plot(r_fixed_points, r_circ_points, "o", label="Input pairs")
    ax.set_xlabel(r"$r_{\mathrm{fixed}}$ (polar-image radial units)")
    ax.set_ylabel(r"$r_{\mathrm{circ}}$ (polar-image radial units)")
    ax.set_title("Radial reassignment map")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "reassignment_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fit_distortion_polynomials(radii, measured_vals, order=2, sigma=None):
    """
    Fit A_n(r) or B_n(r) with a polynomial constrained to vanish at r = 0.

    Parameters
    ----------
    radii : array-like
        Radii r_i where coefficients were extracted from the Gascooke software.
    measured_vals : array-like
        Measured A_n(r_i) or B_n(r_i) values in pixels.
    order : int
        Polynomial order. order=2 means a2*r^2 + a1*r.
    sigma : array-like or None
        Optional 1-sigma uncertainties for weighted fitting.
    """
    radii = np.asarray(radii, dtype=float)
    measured_vals = np.asarray(measured_vals, dtype=float)

    if radii.ndim != 1 or measured_vals.ndim != 1:
        raise ValueError("radii and measured_vals must be 1D arrays")
    if len(radii) != len(measured_vals):
        raise ValueError("radii and measured_vals must have the same length")
    if len(radii) < order:
        raise ValueError("not enough data points for requested polynomial order")

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != measured_vals.shape:
            raise ValueError("sigma must have the same shape as measured_vals")
        absolute_sigma = True
    else:
        absolute_sigma = False

    p0 = np.zeros(order, dtype=float)
    popt, pcov = curve_fit(
        poly_func_no_const,
        radii,
        measured_vals,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        maxfev=10000,
    )
    return popt, pcov


def evaluate_delta_r(r, theta, fit_params, max_trig_n=None):
    """
    Evaluate Delta r(r, theta) = sum_n [A_n(r) sin(n theta) + B_n(r) cos(n theta)].

    fit_params is a dict like:
        {
            "A1": [a2, a1],
            "B1": [b2, b1],
            "A2": [...],
            ...
        }
    """
    if max_trig_n is None:
        trig_orders = sorted(
            int(key[1:]) for key in fit_params if key and key[0] in {"A", "B"}
        )
        max_trig_n = max(trig_orders, default=0)

    delta_r = np.zeros_like(r, dtype=float)
    d_delta_r_dr = np.zeros_like(r, dtype=float)

    for n in range(1, max_trig_n + 1):
        a_key = f"A{n}"
        b_key = f"B{n}"

        if a_key in fit_params:
            a_r, da_dr = eval_radial_model(fit_params[a_key], r)
            sin_term = np.sin(n * theta)
            delta_r += a_r * sin_term
            d_delta_r_dr += da_dr * sin_term
        if b_key in fit_params:
            b_r, db_dr = eval_radial_model(fit_params[b_key], r)
            cos_term = np.cos(n * theta)
            delta_r += b_r * cos_term
            d_delta_r_dr += db_dr * cos_term

    return delta_r, d_delta_r_dr


def circularize_vmi(
    raw_image,
    center,
    fit_params,
    max_trig_n=None,
    interp_order=3,
    apply_jacobian=True,
    jacobian_eps=1e-12,
    reassignment_model=None,
    polar_dr=1.0,
):
    """
    Circularize an image by inverse warping.

    Parameters
    ----------
    raw_image : 2D ndarray
        Distorted image.
    center : tuple(float, float)
        Fixed image center in Python indexing: (xc, yc).
    fit_params : dict
        Polynomial coefficients for A_n(r) and B_n(r).
    max_trig_n : int or None
        Highest trig order to include. If None, infer from fit_params.
    interp_order : int
        Interpolation order passed to scipy.ndimage.map_coordinates.

    Notes
    -----
    The angle convention matches the paper:
    - theta is measured anticlockwise from +x
    - image y increases downward, so we use y_math = yc - y_image
    """
    raw_image = np.asarray(raw_image, dtype=float)
    if raw_image.ndim != 2:
        raise ValueError("raw_image must be a 2D array")

    ny, nx = raw_image.shape
    xc, yc = center

    y_grid, x_grid = np.indices((ny, nx), dtype=float)

    dx = x_grid - xc
    dy_math = y_grid - yc

    r_fixed = np.sqrt(dx**2 + dy_math**2)
    theta = np.arctan2(dy_math, dx)

    if polar_dr <= 0:
        raise ValueError("polar_dr must be positive")

    r_fixed_model = r_fixed / polar_dr
    r_target_model, dr_targetmodel_dr_fixedmodel = eval_reassignment_model(
        reassignment_model, r_fixed_model
    )

    delta_r, d_delta_r_dr = evaluate_delta_r(
        r_target_model, theta, fit_params, max_trig_n=max_trig_n
    )

    r_source_model = r_target_model + delta_r
    r_source = polar_dr * r_source_model
    x_source = xc + r_source * np.cos(theta)
    y_source = yc + r_source * np.sin(theta)

    coords = np.vstack([y_source.ravel(), x_source.ravel()])
    corrected_flat = map_coordinates(
        raw_image,
        coords,
        order=interp_order,
        mode="constant",
        cval=0.0,
    )
    corrected = corrected_flat.reshape((ny, nx))

    if apply_jacobian:
        dr_sourcemodel_dr_targetmodel = 1.0 + d_delta_r_dr
        dr_source_dr_fixed = (
            dr_sourcemodel_dr_targetmodel * dr_targetmodel_dr_fixedmodel
        )

        r_safe = np.where(r_fixed > jacobian_eps, r_fixed, 1.0)
        jacobian = (r_source / r_safe) * dr_source_dr_fixed

        # At the exact center, use the limiting radial derivative.
        center_mask = r_fixed <= jacobian_eps
        if np.any(center_mask):
            jacobian = np.array(jacobian, copy=True)
            jacobian[center_mask] = dr_source_dr_fixed[center_mask] ** 2

        finite_mask = np.isfinite(jacobian)
        positive_mask = jacobian > 0.0
        valid_mask = finite_mask & positive_mask

        corrected = np.where(valid_mask, corrected * jacobian, 0.0)

    return corrected


def load_image_txt(path, shape=None):
    """
    Load a whitespace-separated text image.

    If shape is provided, reshape to that exact shape.
    Otherwise rely on np.loadtxt to infer 2D structure.
    """
    arr = np.loadtxt(path)
    if shape is not None:
        arr = np.asarray(arr, dtype=float).reshape(shape)
    return arr


def save_image_txt(path, image):
    np.savetxt(path, image, fmt="%.18e")


def parse_extracted_txt(path):
    """
    Parse the Gascooke software export in extracted.txt.

    Returns
    -------
    blocks : list of dict
        One dict per fitted ring. Each dict contains:
        - "radius", "radius_sigma"
        - "A1", "A1_sigma", "B1", "B1_sigma", ...
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]

    header = '"Parameter" "Value" "+-" "Uncertainty (1 std dev)"'
    blocks = []
    current = None

    trig_pattern = re.compile(r'^Trig Term (sin|cos)\((\d+)\*theta\)$')

    for line in lines:
        if not line:
            continue

        if line == header:
            if current:
                blocks.append(current)
            current = {}
            continue

        if current is None:
            continue

        parts = re.findall(r'"([^"]+)"|([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(\+\-)', line)
        tokens = []
        for quoted, number, pm in parts:
            if quoted:
                tokens.append(quoted)
            elif number:
                tokens.append(number)
            elif pm:
                tokens.append(pm)

        if len(tokens) < 4:
            continue

        name = tokens[0]
        value = float(tokens[1])
        sigma = float(tokens[3])

        if name == "Gaussian Centre":
            current["radius"] = value
            current["radius_sigma"] = sigma
            continue

        match = trig_pattern.match(name)
        if match:
            trig_type, order_str = match.groups()
            order = int(order_str)
            key = f"A{order}" if trig_type == "sin" else f"B{order}"
            current[key] = value
            current[f"{key}_sigma"] = sigma

    if current:
        blocks.append(current)

    if not blocks:
        raise ValueError(f"No fit blocks found in {path}")

    return blocks


def build_fit_dataset(blocks, max_trig_n=None):
    """
    Convert parsed blocks into arrays grouped by trig order.
    """
    radii = np.array([block["radius"] for block in blocks], dtype=float)
    dataset = {"radii": radii}

    if max_trig_n is None:
        max_trig_n = 0
        for block in blocks:
            for key in block:
                if key.startswith(("A", "B")) and not key.endswith("_sigma"):
                    max_trig_n = max(max_trig_n, int(key[1:]))

    for n in range(1, max_trig_n + 1):
        for prefix in ("A", "B"):
            key = f"{prefix}{n}"
            values = []
            sigmas = []
            for block in blocks:
                if key in block:
                    values.append(block[key])
                    sigmas.append(block[f"{key}_sigma"])
            if len(values) == len(blocks):
                dataset[key] = np.array(values, dtype=float)
                dataset[f"{key}_sigma"] = np.array(sigmas, dtype=float)

    return dataset


def fit_all_trig_polynomials(dataset, order=2, max_trig_n=None):
    """
    Fit all available A_n(r), B_n(r) arrays in the dataset.
    """
    radii = dataset["radii"]
    fit_params = {}
    fit_meta = {}

    if max_trig_n is None:
        max_trig_n = 0
        for key in dataset:
            if key.startswith(("A", "B")) and not key.endswith("_sigma"):
                max_trig_n = max(max_trig_n, int(key[1:]))

    for n in range(1, max_trig_n + 1):
        for prefix in ("A", "B"):
            key = f"{prefix}{n}"
            sigma_key = f"{key}_sigma"
            if key not in dataset:
                continue
            coeffs, cov = fit_distortion_polynomials(
                radii,
                dataset[key],
                order=order,
                sigma=dataset.get(sigma_key),
            )
            fit_params[key] = {
                "kind": "poly",
                "coeffs": np.asarray(coeffs, dtype=float),
            }
            fit_meta[key] = {
                "cov": cov,
                "order": order,
                "radii": radii.copy(),
                "values": dataset[key].copy(),
                "sigma": dataset.get(sigma_key),
                "model_kind": "poly",
            }

    return fit_params, fit_meta


def build_all_trig_interpolators(dataset, max_trig_n=None):
    """
    Build shape-preserving radial interpolants for A_n(r), B_n(r).

    The extracted radii are fit exactly at the sampled points and clipped
    outside the sampled radial range to avoid unstable extrapolation.
    """
    radii = dataset["radii"]
    fit_params = {}
    fit_meta = {}

    if max_trig_n is None:
        max_trig_n = 0
        for key in dataset:
            if key.startswith(("A", "B")) and not key.endswith("_sigma"):
                max_trig_n = max(max_trig_n, int(key[1:]))

    for n in range(1, max_trig_n + 1):
        for prefix in ("A", "B"):
            key = f"{prefix}{n}"
            if key not in dataset:
                continue

            radii_aug = np.concatenate(([0.0], radii))
            values_aug = np.concatenate(([0.0], dataset[key]))

            interp = PchipInterpolator(radii_aug, values_aug, extrapolate=True)
            dinterp = interp.derivative()
            fit_params[key] = {
                "kind": "interp",
                "interp": interp,
                "dinterp": dinterp,
                "r_min": 0.0,
                "r_max": float(np.max(radii)),
            }
            fit_meta[key] = {
                "order": None,
                "radii": radii_aug.copy(),
                "values": values_aug.copy(),
                "sigma": dataset.get(f"{key}_sigma"),
                "model_kind": "interp",
            }

    return fit_params, fit_meta


def print_dataset_summary(dataset):
    radii = dataset["radii"]
    print("Parsed fitted rings from extracted.txt")
    print(f"Number of rings: {len(radii)}")
    print("Radii:")
    for idx, radius in enumerate(radii, start=1):
        print(f"  ring {idx}: r = {radius:.6f}")
    print("Available trig series:")
    keys = sorted(
        key for key in dataset if key.startswith(("A", "B")) and not key.endswith("_sigma")
    )
    for key in keys:
        vals = ", ".join(f"{v:.6f}" for v in dataset[key])
        sigmas = ", ".join(f"{s:.6f}" for s in dataset[f"{key}_sigma"])
        print(f"  {key}: [{vals}]")
        print(f"  {key}_sigma: [{sigmas}]")


def plot_trig_fits(dataset, fit_params, fit_meta, output_dir, max_trig_n=None):
    """
    Plot measured A_n(r), B_n(r) values with uncertainties and fitted curves.
    """
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed, so plotting is unavailable in this environment"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    radii = dataset["radii"]
    if max_trig_n is None:
        max_trig_n = 0
        for key in fit_params:
            max_trig_n = max(max_trig_n, int(key[1:]))

    r_max = float(np.max(radii)) if len(radii) else 1.0
    r_plot = np.linspace(0.0, r_max * 1.1, 500)

    for n in range(1, max_trig_n + 1):
        keys = [f"A{n}", f"B{n}"]
        existing = [key for key in keys if key in fit_params and key in dataset]
        if not existing:
            continue

        fig, axes = plt.subplots(1, len(existing), figsize=(6 * len(existing), 4.5))
        if len(existing) == 1:
            axes = [axes]

        for ax, key in zip(axes, existing):
            sigma_key = f"{key}_sigma"
            y = dataset[key]
            yerr = dataset.get(sigma_key)
            model = fit_params[key]
            y_plot, _ = eval_radial_model(model, r_plot)

            ax.errorbar(
                radii,
                y,
                yerr=yerr,
                fmt="o",
                capsize=4,
                label=f"{key} data",
            )
            ax.plot(r_plot, y_plot, "-", label=f"{key} fit")
            ax.axhline(0.0, color="0.7", linewidth=1)
            ax.axvline(0.0, color="0.7", linewidth=1)
            ax.set_xlabel("Radius r / pixels")
            ax.set_ylabel(f"{key}(r) / pixels")
            ax.set_title(f"{key}(r)")
            ax.legend()

            if model["kind"] == "poly":
                coeffs = model["coeffs"]
                coeff_text = " + ".join(
                    f"{c:.3e} r^{p}"
                    for c, p in zip(coeffs, range(len(coeffs), 0, -1))
                )
            else:
                coeff_text = (
                    f"PCHIP on [{model['r_min']:.1f}, {model['r_max']:.1f}], "
                    "clipped outside"
                )
            ax.text(
                0.03,
                0.97,
                coeff_text if coeff_text else "0",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
            )

        fig.suptitle(f"Trig order n = {n}")
        fig.tight_layout()
        fig.savefig(output_dir / f"trig_order_{n}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def print_fit_summary(fit_params):
    print("Polynomial fits used for circularization:")
    for key in sorted(fit_params, key=lambda x: (x[0], int(x[1:]))):
        model = fit_params[key]
        if model["kind"] == "poly":
            coeffs = model["coeffs"]
            terms = [
                f"{coeff:.6e} * r^{power}"
                for coeff, power in zip(coeffs, range(len(coeffs), 0, -1))
            ]
            expr = " + ".join(terms) if terms else "0"
            print(f"  {key}(r) = {expr}")
        else:
            print(
                f"  {key}(r) = PCHIP interpolation on "
                f"[{model['r_min']:.6f}, {model['r_max']:.6f}], clipped outside"
            )


def summarize_delta_r_contributions(fit_params, max_radius, max_trig_n=None, num_theta=720):
    """
    Print the maximum absolute Delta r contribution from each trig term
    over 0 <= r <= max_radius and 0 <= theta < 2*pi.
    """
    if max_trig_n is None:
        max_trig_n = 0
        for key in fit_params:
            max_trig_n = max(max_trig_n, int(key[1:]))

    r_samples = np.linspace(0.0, float(max_radius), 400)
    theta_samples = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)

    print("Maximum absolute Delta r contribution by term:")

    total_max_abs = 0.0
    for n in range(1, max_trig_n + 1):
        for prefix, trig_name in (("A", "sin"), ("B", "cos")):
            key = f"{prefix}{n}"
            if key not in fit_params:
                continue

            radial_part, _ = eval_radial_model(fit_params[key], r_samples)
            trig_part = (
                np.sin(n * theta_samples)[None, :]
                if trig_name == "sin"
                else np.cos(n * theta_samples)[None, :]
            )
            contribution = radial_part[:, None] * trig_part
            max_abs = float(np.max(np.abs(contribution)))
            total_max_abs += max_abs
            print(f"  {key}: max |contribution| = {max_abs:.6f} pixels")

    print(f"  Sum of per-term maxima (upper bound): {total_max_abs:.6f} pixels")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Circularize a VMI image using fixed-center polynomial A_n(r), B_n(r) fits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("image.txt"),
        help="Input text image file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image_circularized.txt"),
        help="Output text image file.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(1023, 1023),
        metavar=("NY", "NX"),
        help="Image shape if reshaping is needed.",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        default=(511.0, 511.0),
        metavar=("XC", "YC"),
        help="Fixed center in Python indexing.",
    )
    parser.add_argument(
        "--interp-order",
        type=int,
        default=3,
        help="Interpolation order for remapping.",
    )
    parser.add_argument(
        "--extracted",
        type=Path,
        default=Path("extracted.txt"),
        help="Gascooke trig-fit export file.",
    )
    parser.add_argument(
        "--poly-order",
        type=int,
        default=2,
        help="Polynomial order for A_n(r), B_n(r) fits.",
    )
    parser.add_argument(
        "--radial-model",
        choices=("interp", "poly"),
        default="interp",
        help="Radial model for A_n(r), B_n(r). 'interp' passes through extracted points.",
    )
    parser.add_argument(
        "--max-trig-n",
        type=int,
        default=None,
        help="Highest trig order to use. Default: infer from extracted.txt.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only parse and print extracted.txt summary.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("trig_fit_plots"),
        help="Directory for plots of A_n(r), B_n(r) and fitted curves.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing diagnostic plots.",
    )
    parser.add_argument(
        "--no-jacobian",
        action="store_true",
        help="Disable Jacobian intensity correction during remapping.",
    )
    parser.add_argument(
        "--reassign-file",
        type=Path,
        default=None,
        help="Optional two-column file mapping circularized radii to desired fixed radii: r_circ r_fixed",
    )
    parser.add_argument(
        "--polar-dr",
        type=float,
        default=1.0,
        help="Radial step size used in the polar image. Extracted radii/A_n/B_n are interpreted in these units.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    blocks = parse_extracted_txt(args.extracted)
    dataset = build_fit_dataset(blocks, max_trig_n=args.max_trig_n)

    print_dataset_summary(dataset)

    if args.summary_only:
        return

    raw_image = load_image_txt(args.input, shape=tuple(args.shape))
    if args.radial_model == "poly":
        fit_params, fit_meta = fit_all_trig_polynomials(
            dataset,
            order=args.poly_order,
            max_trig_n=args.max_trig_n,
        )
    else:
        fit_params, fit_meta = build_all_trig_interpolators(
            dataset,
            max_trig_n=args.max_trig_n,
        )
    print_fit_summary(fit_params)
    summarize_delta_r_contributions(
        fit_params,
        max_radius=float(np.max(dataset["radii"])),
        max_trig_n=args.max_trig_n,
    )

    if not args.no_plots and plt is not None:
        plot_trig_fits(
            dataset,
            fit_params,
            fit_meta,
            output_dir=args.plot_dir,
            max_trig_n=args.max_trig_n,
        )
    elif not args.no_plots and plt is None:
        print("Plotting skipped: matplotlib is not installed in this environment.")

    reassignment_model = {"kind": "identity"}
    if args.reassign_file is not None:
        r_circ_points, r_fixed_points = parse_reassignment_file(args.reassign_file)
        reassignment_model = build_reassignment_interpolator(r_circ_points, r_fixed_points)
        print("Applied radial reassignment:")
        for rc, rf in zip(r_circ_points, r_fixed_points):
            print(f"  r_circ={rc:.6f} -> r_fixed={rf:.6f}")
        plot_reassignment_map(
            reassignment_model,
            r_circ_points,
            r_fixed_points,
            output_dir=args.plot_dir,
        )
    print(f"Using polar radial step size: polar_dr = {args.polar_dr:.6f} Cartesian pixels per polar radial unit")

    corrected = circularize_vmi(
        raw_image,
        center=tuple(args.center),
        fit_params=fit_params,
        max_trig_n=args.max_trig_n,
        interp_order=args.interp_order,
        apply_jacobian=not args.no_jacobian,
        reassignment_model=reassignment_model,
        polar_dr=args.polar_dr,
    )

    save_image_txt(args.output, corrected)


if __name__ == "__main__":
    main()
