"""
Process the four GMD-binned VMI images through the PES / Abel-inversion pipeline.

Pipeline (per bin), following PES-analysis-stretch-angular_..._80eV-202606.ipynb
with the two external gaps bridged the way circularization/background_subtraction.py
does it:

    bin_n.npy
      -> apply_circular_bg_patch   (feathered ring at the detector hole)
      -> rotate 90 deg
      -> abel.tools.center.center_image         (center_of_img_old)
      -> processed_img  --> circularization.py  --> IMcirc
      -> abel.tools.center.center_image         (511, 511)
      -> apply_soft_mcp_mask                    (edge apodization, 470 / 90)
      -> image_cleaned  --> MATLAB CPBASEX or PyAbel --> E, IE, betas

Data source: the four GMD-binned histograms produced by
centroiding/vmi_histogram_gmd_bins.py (dataset R0102, "Xe_j_100_sig").

!! IMPORTANT !!
The MATLAB step writes to a SHARED directory (CPBASEX/examples/image.txt) and
new_10.m hardcodes 'image.txt' as its input. The bins therefore MUST run
sequentially, and each bin's MATLAB outputs are copied out immediately after
its run. Do not parallelise this script.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' matplotlib style)
from scipy import ndimage
from skimage.transform import rotate
import abel

plt.style.use(['science'])
SMALL_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)

HERE = Path(__file__).parent
CIRCULARIZATION_DIR = HERE  # circularization.py lives alongside this script

# =============================================================================
# Configuration
# =============================================================================

# --- Abel inversion (configured here; no config.toml dependency) ---
INVERSION_BACKEND = "pyabel"  # "matlab" or "pyabel"
PYABEL_METHOD = "daun"       # PyAbel only; e.g. "daun", "hansenlaw", "three_point"
# PyAbel uses the method's defaults (daun: reg=0, no regularization).
if INVERSION_BACKEND not in ("matlab", "pyabel"):
    raise ValueError("INVERSION_BACKEND must be 'matlab' or 'pyabel'")
if not PYABEL_METHOD.isidentifier():
    raise ValueError("PYABEL_METHOD must be a PyAbel method name")

# --- Inputs: the four GMD bins ---
BIN_DIR = HERE.parent / "centroiding" / "vmi_analysis_output_xe_j_100_sig"

# Which bin set to process (vmi_histogram_gmd_bins.py WEIGHT_MODE suffix):
#   ""            -> unweighted bins (default)
#   "_w-gmd"      -> raw GMD scaling
#   "_w-gmd_norm" -> linear pulse-energy normalisation
WEIGHT_SUFFIX = "_w-gmd_norm"

# --- Output ---
# Weighted runs go to their own directory so their cached inversion outputs can
# never collide with (and silently be reused in place of) the unweighted ones.
OUTPUT_DIR = HERE / (f"gmd_bins_results{WEIGHT_SUFFIX}" if WEIGHT_SUFFIX
                     else "gmd_bins_results")
if INVERSION_BACKEND == "pyabel":
    OUTPUT_DIR = OUTPUT_DIR.with_name(OUTPUT_DIR.name + f"_pyabel_{PYABEL_METHOD}")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Circularization calibration (confirmed for this dataset) ---
EXTRACTED_FILE = CIRCULARIZATION_DIR / "extracted-FOC100eV-20260710.txt"
REASSIGN_FILE = CIRCULARIZATION_DIR / "reassignment-FOC100eV-20260710.txt"

# --- MATLAB / CPBASEX ---
MATLAB_CPBASEX = Path("E:/work/projects/cVMI-exp/dat/2025-02-17/CPBASEX")
MATLAB_EXAMPLES_DIR = MATLAB_CPBASEX / "examples"
MATLAB_SCRIPT = MATLAB_EXAMPLES_DIR / "new_10.m"
MATLAB_PBASE_PATH = MATLAB_CPBASEX / "pbasex-MATLAB"

# --- Image / geometry parameters (80 eV notebook, this dataset) ---
CENTER_OLD = np.array([473.47577550518895, 499.38892648532163])  # bright spot (row, col)
CENTER_NEW = np.array([511.0, 511.0])

# Shape passed to circularization.py --shape. NOTE: this is the shape of
# `processed_img` AFTER abel center_image(square=True), which crops the raw
# 1024x1024 bin to 1023x1023 (odd size -> single-pixel center).
RAW_SHAPE = (1024, 1024)
IMG_SHAPE = (1023, 1023)

# Detector-hole repair (notebook CELL 13)
HOLE_RADIUS = 13
DONOR_DX = -22          # donor_xc = center_old[1] + DONOR_DX
DONOR_DY = +22          # donor_yc = center_old[0] + DONOR_DY
HOLE_RING_WIDTH = 2
HOLE_SIGMA = 7

# Edge apodization before inversion (notebook CELL 60)
MCP_EDGE_CUTOFF = 470.0
TAPER_WIDTH = 90.0

# circularization.py options
MAX_TRIG_N = 6
INTERP_ORDER = 4
NO_JACOBIAN = True

# Background subtraction: the bins are already dark-subtracted by the Picasso
# pipeline; the notebook has background subtraction commented out.
NO_BKG = True

# Energy calibration (pixel^2 -> eV). Set to 1.0 for now; the real scaling
# factor is to be determined by hand later.
ALPHA = 1.0

# --- Beta-plot presentation ---
# The CPBASEX beta fit is ill-conditioned as intensity -> 0, producing
# unphysical spikes at the extreme ends of the energy axis. Restrict the
# plotted window and clip the y-axis so the physical region is readable.
BETA_ENERGY_FRACTION = 0.5      # plot E in [0, fraction * Emax]
BETA_YLIM = (-1.5, 2.5)         # physical beta range is roughly [-1, +2]
BETA_THRESHOLD = []             # optional vertical guide lines, in E units


# =============================================================================
# Image-processing helpers (from the notebook)
# =============================================================================

def apply_circular_bg_patch(image, hole_xc, hole_yc, hole_radius,
                            donor_xc, donor_yc, ring_width=10, sigma=3):
    """Blend ONLY a feathered ring around the detector-hole boundary.

    The image inside the hole is preserved; only the ring at r ~ hole_radius
    is cross-faded with a donor patch copied from a clean region.
    """
    image = image.astype(np.float32)

    hole_xc = int(round(hole_xc))
    hole_yc = int(round(hole_yc))
    donor_xc = int(round(donor_xc))
    donor_yc = int(round(donor_yc))
    hole_radius = int(round(hole_radius))

    H, W = image.shape

    # Build donor image by copying a square ROI
    pad = int(hole_radius + ring_width + 3 * sigma)

    hx0 = max(0, hole_xc - pad); hx1 = min(W, hole_xc + pad + 1)
    hy0 = max(0, hole_yc - pad); hy1 = min(H, hole_yc + pad + 1)
    dx0 = max(0, donor_xc - pad); dx1 = min(W, donor_xc + pad + 1)
    dy0 = max(0, donor_yc - pad); dy1 = min(H, donor_yc + pad + 1)

    w = min(hx1 - hx0, dx1 - dx0)
    h = min(hy1 - hy0, dy1 - dy0)

    hole_roi = (slice(hy0, hy0 + h), slice(hx0, hx0 + w))
    donor_roi = (slice(dy0, dy0 + h), slice(dx0, dx0 + w))

    replacement = image.copy()
    replacement[hole_roi] = image[donor_roi]

    # Ring-shaped feather mask
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - hole_xc) ** 2 + (yy - hole_yc) ** 2)

    ring = ((r >= hole_radius - ring_width / 2)
            & (r <= hole_radius + ring_width / 2))

    soft_mask = ndimage.gaussian_filter(ring.astype(np.float32), sigma=sigma)
    if soft_mask.max() > 0:
        soft_mask /= soft_mask.max()

    result = image * (1 - soft_mask) + replacement * soft_mask
    return np.clip(result, 0, None).astype(image.dtype)


def apply_soft_mcp_mask(image, center, r0, width=30):
    """cos^2 taper to zero beyond r0 (P'=0 at both ends) to stop Abel ringing."""
    ny, nx = image.shape
    xc, yc = center

    y_grid, x_grid = np.indices((ny, nx))
    r = np.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2)

    r_start = r0 - width
    mask = np.ones_like(r, dtype=float)

    taper = (r >= r_start) & (r <= r0)
    phi = (r[taper] - r_start) / width
    mask[taper] = np.cos(np.pi / 2 * phi) ** 2
    mask[r > r0] = 0.0

    return image * mask


# =============================================================================
# External steps
# =============================================================================

def run_circularization(input_path: Path, output_path: Path) -> None:
    """Shell out to circularization.py (must run with cwd = this directory).

    Paths are passed ABSOLUTE: the subprocess cwd is CIRCULARIZATION_DIR, so a
    bare filename would not resolve against the output subdirectory.
    """
    cmd = [
        "uv", "run", "circularization.py",
        "--input", str(input_path.resolve()),
        "--output", str(output_path.resolve()),
        "--center", "511", "511",
        "--shape", str(IMG_SHAPE[0]), str(IMG_SHAPE[1]),
        "--max-trig-n", str(MAX_TRIG_N),
        "--extracted", str(EXTRACTED_FILE.resolve()),
        "--reassign-file", str(REASSIGN_FILE.resolve()),
        "--interp-order", str(INTERP_ORDER),
        "--no-plots",
    ]
    if NO_JACOBIAN:
        cmd.append("--no-jacobian")

    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True,
                            cwd=str(CIRCULARIZATION_DIR))
    if result.returncode != 0:
        print(f"    [circularization.py FAILED, rc={result.returncode}]")
        if result.stdout:
            print("    --- stdout (tail) ---")
            print("    " + "\n    ".join(result.stdout.strip().splitlines()[-15:]))
        if result.stderr:
            print("    --- stderr (tail) ---")
            print("    " + "\n    ".join(result.stderr.strip().splitlines()[-25:]))
        raise RuntimeError(f"circularization.py failed (rc={result.returncode})")


def run_matlab_cpbasex() -> None:
    """Run the CPBASEX inversion in MATLAB (reads examples/image.txt)."""
    matlab_cmd = (
        f"matlab -batch \"addpath('{MATLAB_PBASE_PATH.as_posix()}'); "
        f"cd('{MATLAB_EXAMPLES_DIR.as_posix()}'); run('new_10.m');\""
    )
    subprocess.run(matlab_cmd, shell=True, check=True)


def load_matlab_results():
    """Read the CPBASEX outputs (raw channel units)."""
    Ek = np.loadtxt(MATLAB_EXAMPLES_DIR / "energy_l10.txt", delimiter=",")
    IEk = np.loadtxt(MATLAB_EXAMPLES_DIR / "energy_spec_l10.txt", delimiter=",")
    betas = np.loadtxt(MATLAB_EXAMPLES_DIR / "betas10.txt", delimiter=",")
    return Ek, IEk, betas


def run_pyabel(image):
    """Invert the centered image; return r², PES, beta2..beta10 and the slice."""
    inverted = abel.Transform(
        image, direction="inverse", method=PYABEL_METHOD, origin="none",
        symmetry_axis=(0, 1),
    ).transform
    radial, intensity, *betas = abel.tools.vmi.rIbeta(
        inverted, origin="cc", rmax=image.shape[1] // 2, order=10,
    )
    # Ibeta gives dN/dr; toPES applies dN/d(r²) = (dN/dr)/(2r).
    # Keep factor=1: saved axes remain r², as in the MATLAB pipeline.
    # toPES modifies its intensity argument, so pass a copy.
    Ek, IEk = abel.tools.vmi.toPES(
        radial, intensity.copy(), energy_cal_factor=1.0,
        per_energy_scaling=True,
    )
    return Ek, IEk, np.asarray(betas).T, inverted


# =============================================================================
# Per-bin processing
# =============================================================================

def process_bin(bin_path: Path, label: str, use_cache: bool = False):
    """Full chain for one GMD bin. Returns dict or raises.

    If use_cache and this bin's inversion outputs already exist in OUTPUT_DIR,
    circularization and inversion are skipped and the cached results reloaded.
    """
    cached_energy = OUTPUT_DIR / f"energy_l10_GMD_{label}.txt"
    cached_spec = OUTPUT_DIR / f"energy_spec_l10_GMD_{label}.txt"
    cached_betas = OUTPUT_DIR / f"betas10_GMD_{label}.txt"
    cached_img = OUTPUT_DIR / f"image_cleaned_GMD_{label}.txt"

    if use_cache and cached_energy.exists() and cached_spec.exists() \
            and cached_betas.exists() and cached_img.exists():
        print(f"\n  [cache] {label}: reusing existing inversion outputs")
        Ek = np.loadtxt(cached_energy, delimiter=",")
        IEk = np.loadtxt(cached_spec, delimiter=",")
        betas = np.loadtxt(cached_betas, delimiter=",")
        image_cleaned = np.loadtxt(cached_img)
        processed_img = None  # not needed for the spectrum plots
        if cached_img.with_name(f"image_GMD_{label}.txt").exists():
            p = np.loadtxt(OUTPUT_DIR / f"image_GMD_{label}.txt")
            processed_img = p
        print(f"         Ek={Ek.shape} IEk={IEk.shape} betas={betas.shape}")
        return {
            "label": label,
            "energy": Ek,
            "intensity": IEk,
            "beta": betas,
            "image": image_cleaned,
            "processed_img": processed_img,
        }

    print(f"\n  [load] {bin_path.name}")
    img_sig = np.load(bin_path)
    print(f"         shape={img_sig.shape} sum={img_sig.sum():,.0f}")

    if not NO_BKG:
        raise NotImplementedError("background subtraction not configured "
                                  "(no per-bin background file)")

    # --- hole repair -------------------------------------------------------
    img = apply_circular_bg_patch(
        image=img_sig,
        hole_xc=CENTER_OLD[1],
        hole_yc=CENTER_OLD[0],
        hole_radius=HOLE_RADIUS,
        donor_xc=CENTER_OLD[1] + DONOR_DX,
        donor_yc=CENTER_OLD[0] + DONOR_DY,
        ring_width=HOLE_RING_WIDTH,
        sigma=HOLE_SIGMA,
    )
    img = img.copy()

    # --- rotate + center ---------------------------------------------------
    imgnew = rotate(img, 90, center=(CENTER_OLD[1], CENTER_OLD[0]))
    centered = abel.tools.center.center_image(
        imgnew, method=tuple(CENTER_OLD), square=True)
    processed_img = centered.copy()

    # --- circularization (external) ---------------------------------------
    circ_in = OUTPUT_DIR / f"image_GMD_{label}.txt"
    circ_out = OUTPUT_DIR / f"image_circularized_GMD_{label}.txt"
    np.savetxt(circ_in, processed_img)
    print(f"  [circ] circularizing GMD={label} ...")
    run_circularization(circ_in, circ_out)

    IMcirc = np.loadtxt(circ_out)
    print(f"         circularized: shape={IMcirc.shape} sum={IMcirc.sum():,.0f}")

    # --- recenter + apodize ------------------------------------------------
    centered2 = abel.tools.center.center_image(
        IMcirc.astype(float), method=tuple(CENTER_NEW), square=True)
    full_processed_img = apply_soft_mcp_mask(
        centered2, np.array(centered2.shape) // 2,
        MCP_EDGE_CUTOFF, TAPER_WIDTH)
    image_cleaned = full_processed_img.copy()

    np.savetxt(OUTPUT_DIR / f"image_cleaned_GMD_{label}.txt", image_cleaned)

    # --- Abel inversion ---------------------------------------------------
    if INVERSION_BACKEND == "matlab":
        np.savetxt(MATLAB_EXAMPLES_DIR / "image.txt", image_cleaned)
        print(f"  [cPbasex] running MATLAB new_10.m for GMD={label} ...")
        run_matlab_cpbasex()

        # Copy shared MATLAB outputs before processing the next bin.
        for stem in ("energy_l10", "energy_spec_l10", "betas10", "px-py_l10"):
            src = MATLAB_EXAMPLES_DIR / f"{stem}.txt"
            if src.exists():
                shutil.copy2(src, OUTPUT_DIR / f"{stem}_GMD_{label}.txt")
        Ek, IEk, betas = load_matlab_results()
    else:
        print(f"  [PyAbel] {PYABEL_METHOD} inversion for GMD={label} ...")
        Ek, IEk, betas, inverted = run_pyabel(image_cleaned)
        for stem, values in (("energy_l10", Ek), ("energy_spec_l10", IEk),
                             ("betas10", betas), ("px-py_l10", inverted)):
            np.savetxt(OUTPUT_DIR / f"{stem}_GMD_{label}.txt", values, delimiter=",")
    print(f"         Ek={Ek.shape} IEk={IEk.shape} betas={betas.shape}")

    return {
        "label": label,
        "energy": Ek,
        "intensity": IEk,
        "beta": betas,
        "image": image_cleaned,
        "processed_img": processed_img,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    # vmi_histogram_gmd_bins.py may also emit GMD-weighted variants whose
    # filenames carry a "_w-<mode>" suffix. Select which set to process here
    # ("" = unweighted default; "_w-gmd" = raw GMD scaling; "_w-gmd_norm" =
    # linear pulse-energy normalisation).
    all_bins = sorted(BIN_DIR.glob("vmi_2d_gmd_bin*.npy"))
    if WEIGHT_SUFFIX:
        # Keep only files carrying exactly this suffix (drop plain and others)
        bins = [p for p in all_bins
                if p.stem.endswith(WEIGHT_SUFFIX)
                and "_w-" not in p.stem[:-len(WEIGHT_SUFFIX)]]
    else:
        # Keep only the unweighted set
        bins = [p for p in all_bins if "_w-" not in p.stem]

    if not bins:
        print(f"[ERROR] no bin files found in {BIN_DIR} "
              f"(weight suffix {WEIGHT_SUFFIX!r})")
        print(f"        candidates: {[p.name for p in all_bins]}")
        sys.exit(1)

    # Label from filename, KEEPING any weight suffix so weighted and unweighted
    # results never share a cache key.
    #   vmi_2d_gmd_bin0_minusinf_3.75e-08.npy          -> bin0_minusinf_3.75e-08
    #   vmi_2d_gmd_bin0_minusinf_3.75e-08_w-gmd_norm.npy -> bin0_..._w-gmd_norm
    labels = [b.stem.replace("vmi_2d_gmd_", "") for b in bins]

    print("=" * 78)
    print(f"GMD-BINNED VMI PROCESSING -> {INVERSION_BACKEND.upper()}")
    print("=" * 78)
    print(f"Bin dir     : {BIN_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    if INVERSION_BACKEND == "pyabel":
        print(f"Abel method : {PYABEL_METHOD}")
    print(f"Extracted   : {EXTRACTED_FILE.name}")
    print(f"Reassign    : {REASSIGN_FILE.name}")
    print(f"center_old  : {CENTER_OLD}")
    print(f"alpha       : {ALPHA}")
    print(f"Background  : {'SKIPPED' if NO_BKG else 'ON'}")
    print(f"Bins        : {len(bins)}")

    if not EXTRACTED_FILE.exists():
        print(f"[ERROR] extracted file missing: {EXTRACTED_FILE}")
        sys.exit(1)
    if not REASSIGN_FILE.exists():
        print(f"[ERROR] reassignment file missing: {REASSIGN_FILE}")
        sys.exit(1)

    results = {}
    for i, (bin_path, label) in enumerate(zip(bins, labels)):
        print(f"\n[{i+1}/{len(bins)}] GMD bin '{label}'")
        try:
            results[label] = process_bin(bin_path, label)
        except Exception as e:
            print(f"  [ERROR] {label} failed: {e}")
            results[label] = None

    ok = {k: v for k, v in results.items() if v is not None}
    if not ok:
        print("\n[ERROR] no bin succeeded")
        sys.exit(1)

    # ---------------------------------------------------------------- plots
    print("\n" + "=" * 78)
    print("PLOTTING")
    print("=" * 78)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ok)))

    # Energy spectra
    fig, ax = plt.subplots(figsize=(10, 6))
    for c, (label, r) in zip(colors, ok.items()):
        ax.plot(r["energy"] * ALPHA, r["intensity"],
                ls="-", marker='o', lw=1., markersize=2, c=c, label=label, alpha=0.8)
    ax.set_xlabel(r"$E_k$ (eV)" if ALPHA != 1.0 else r"$E$ (channel)")
    ax.set_ylabel("prob. (arb. unit)")
    ax.legend(fontsize=12)
    fig.tight_layout()
    f1 = OUTPUT_DIR / "gmd_bins-energy-spectrum.png"
    fig.savefig(f1, transparent=True, bbox_inches='tight', dpi=150)
    print(f"[plot] {f1}")

    # Normalized spectra
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for c, (label, r) in zip(colors, ok.items()):
        e = r["energy"] * ALPHA
        inten = np.asarray(r["intensity"]).ravel()
        e = np.asarray(e).ravel()
        denom = inten.max() if inten.max() > 0 else 1.0
        ax2.plot(e, inten / denom, ls="-", marker='o', lw=1., markersize=2,
                 c=c, label=label, alpha=0.8)
    ax2.set_xlabel(r"$E_k$ (eV)" if ALPHA != 1.0 else r"$E$ (channel)")
    ax2.set_ylabel("normalized intensity")
    ax2.legend(fontsize=12)
    fig2.tight_layout()
    f2 = OUTPUT_DIR / "gmd_bins-energy-spectrum-normalized.png"
    fig2.savefig(f2, transparent=True, bbox_inches='tight', dpi=150)
    print(f"[plot] {f2}")

    # Beta parameters.
    # The CPBASEX beta fit is ill-conditioned where the intensity -> 0 (the
    # extreme ends of the energy axis), where it produces values in the
    # hundreds/thousands. Those points are physical noise, not signal, so the
    # plot is restricted to a sensible energy window with a clipped y-axis.
    # One panel per beta parameter, all four bins overlaid.
    n_beta = max(np.asarray(r["beta"]).shape[1] for r in ok.values())
    emax_all = max(np.asarray(r["energy"] * ALPHA).ravel().max() for r in ok.values())
    e_hi = BETA_ENERGY_FRACTION * emax_all

    fig3, axes3 = plt.subplots(1, n_beta, figsize=(4 * n_beta, 5),
                               sharex=True, sharey=True)
    axes3 = np.atleast_1d(axes3)

    all_th = np.array(BETA_THRESHOLD, dtype=float)

    for j in range(n_beta):
        ax = axes3[j]
        for c, (label, r) in zip(colors, ok.items()):
            b = np.asarray(r["beta"])
            e = np.asarray(r["energy"] * ALPHA).ravel()
            if b.ndim == 1:
                b = b[:, None]
            y = b[:, j]
            m = (e >= 0) & (e <= e_hi)
            ax.plot(e[m], y[m], ls="-", lw=1.0, c=c, alpha=0.85, label=label)
        for t in all_th:
            ax.axvline(t, ls=":", c="grey", lw=0.8)
        ax.set_title(rf"$\beta_{{{2*(j+1)}}}$", fontsize=14)
        ax.set_xlabel("$E$ (channel)" if ALPHA == 1.0 else "$E_k$ (eV)")
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_ylim(BETA_YLIM)
        if j == 0:
            ax.set_ylabel(r"$\beta$")
    axes3[0].legend(fontsize=9, loc="upper right")
    fig3.tight_layout()
    f3 = OUTPUT_DIR / "gmd_bins-betas.png"
    fig3.savefig(f3, transparent=True, bbox_inches='tight', dpi=150)
    print(f"[plot] {f3}")

    # ------------------------------------------------------------- save data
    npz_path = OUTPUT_DIR / "gmd_bins_results.npz"
    np.savez(
        npz_path,
        labels=list(ok.keys()),
        alpha=ALPHA,
        inversion_backend=INVERSION_BACKEND,
        pyabel_method=PYABEL_METHOD if INVERSION_BACKEND == "pyabel" else "",
        energy=[np.asarray(ok[k]["energy"]).ravel() for k in ok],
        intensity=[np.asarray(ok[k]["intensity"]).ravel() for k in ok],
        beta=[np.asarray(ok[k]["beta"]) for k in ok],
        image=[ok[k]["image"] for k in ok],
        processed_img=[ok[k]["processed_img"] if ok[k]["processed_img"] is not None
                       else np.zeros((0, 0)) for k in ok],
    )
    print(f"[save] {npz_path}")

    # Per-bin summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for label, r in ok.items():
        b = np.asarray(r["beta"])
        bmax = np.nanmax(b) if b.size else float("nan")
        print(f"  {label:<34s}  E points={len(r['energy']):>4d}  "
              f"beta range=[{np.nanmin(b):+.3f}, {bmax:+.3f}]")

    plt.show()


if __name__ == "__main__":
    main()
