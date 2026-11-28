"""Run a Julia pBASEX uncertainty analysis from a count image or checkpoint."""

import argparse
from hashlib import sha256
from math import isclose, isfinite
import os
from pathlib import Path
from shutil import which
import subprocess
import sys
import tempfile
import tomllib

import h5py
import numpy as np

from .paths import GENERATED_DIR, validate_distinct_output

MODES = {"counts": "Integer count image", "shots": "Shot-resolved checkpoint"}
HDF5_SUFFIXES = (".h5", ".hdf5")
INPUT_SUFFIXES = {"counts": (".npy",), "shots": HDF5_SUFFIXES}
ROTATIONS = (0, 90, 180, 270)
DEFAULT_REPLICATES = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42
MAX_UNIFORM_BINS = 10_000
PBASEX_UUID = "c378d021-f22a-41a2-8b8b-18f6716cb0a9"


def _number(value, label, *, minimum=0, strict=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Enter a number for {label}.")
    if not minimum <= value <= sys.float_info.max or (strict and value == minimum):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{label} must be finite and {relation} {minimum}.")
    if integer:
        if int(value) != value or value > 2**53 - 1:
            raise ValueError(f"{label} must be an integer up to {2**53 - 1}.")
        return int(value)
    return float(value)


def uniform_windows(e_min, e_max, delta_e):
    """Build adjacent equal-width intervals with the requested endpoints."""
    low = _number(e_min, "E_min")
    high = _number(e_max, "E_max")
    width = _number(delta_e, "ΔE", strict=True)
    if high <= low:
        raise ValueError("E_max must be greater than E_min.")
    bins = (high - low) / width
    if not isfinite(bins) or bins > MAX_UNIFORM_BINS + 1e-8:
        raise ValueError(f"Use a larger ΔE for at most {MAX_UNIFORM_BINS:,} bins.")
    count = round(bins)
    if count < 1 or not isclose(bins, count, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError("E_max − E_min must be a whole multiple of ΔE.")
    edges = [low + i * width for i in range(count)] + [high]
    windows = list(zip(edges, edges[1:]))
    if any(lower >= upper for lower, upper in windows):
        raise ValueError("ΔE is too small to resolve distinct bin edges at this energy.")
    return windows


def grid_from_windows(windows):
    """Recover the three grid values from a saved uniform window list."""
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("Provide at least one energy window.")
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in windows):
        raise ValueError("Each energy window needs a lower and upper bound.")
    bounds = [tuple(_number(value, "Energy window bound") for value in pair) for pair in windows]
    low, high = bounds[0][0], bounds[-1][1]
    width = (high - low) / len(bounds)
    expected = uniform_windows(low, high, width)
    if any(not isclose(value - edge, 0, abs_tol=width * 1e-9)
           for pair, target in zip(bounds, expected) for value, edge in zip(pair, target)):
        raise ValueError("Use consecutive, equal-width windows.")
    return dict(e_min=low, e_max=high, delta_e=width)


def legacy_energy_grid(text):
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        return grid_from_windows([tuple(map(float, line.replace(",", " ").split()))
                                  for line in text.splitlines() if line.strip()])
    except ValueError:
        return {}


def analysis_options(mode, *, alpha=None, resolution=1, rotation=0, regularization=0,
                     center_x=None, center_y=None, windows=(), replicates=DEFAULT_REPLICATES,
                     confidence=DEFAULT_CONFIDENCE, block_length=1, seed=DEFAULT_SEED,
                     e_min=None, e_max=None, delta_e=None, window_batch=1):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Choose integer counts or a shot-resolved checkpoint.")
    if rotation not in ROTATIONS or isinstance(rotation, bool):
        raise ValueError("Choose a rotation of 0°, 90°, 180°, or 270°.")
    if (center_x is None) != (center_y is None):
        raise ValueError("Enter both center coordinates or leave both blank.")
    if center_x is not None:
        center_x = _number(center_x, "Center X", minimum=1, integer=True)
        center_y = _number(center_y, "Center Y", minimum=1, integer=True)
    if not isinstance(windows, (tuple, list)):
        raise ValueError("Enter energy windows as pairs of lower and upper bounds.")
    if any(value is not None for value in (e_min, e_max, delta_e)):
        if windows:
            raise ValueError("Choose either the uniform energy grid or explicit energy windows.")
        windows = uniform_windows(e_min, e_max, delta_e)
    limits = []
    for window in windows:
        if not isinstance(window, (tuple, list)) or len(window) != 2:
            raise ValueError("Enter two bounds per energy window.")
        low, high = (_number(bound, "Energy window bound") for bound in window)
        if low >= high:
            raise ValueError("Each energy window must have an upper bound above its lower bound.")
        limits.append((low, high))
    if mode == "shots" and not limits:
        raise ValueError("Enter at least one energy window for shot bootstrap.")
    options = dict(alpha=None if alpha is None else _number(alpha, "Alpha", strict=True),
                   rotation=int(rotation), regularization=_number(regularization, "Regularization"),
                   center_x=center_x, center_y=center_y, windows=limits)
    if mode == "counts":
        options["resolution"] = _number(resolution, "Bins per detector pixel", strict=True)
    else:
        confidence = _number(confidence, "Confidence level", strict=True)
        if confidence >= 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        options.update(confidence=confidence,
                       replicates=_number(replicates, "Replicates", minimum=2, integer=True),
                       block_length=_number(block_length, "Block length", minimum=1, integer=True),
                       window_batch=_number(window_batch, "Windows per batch", minimum=1, integer=True),
                       seed=_number(seed, "Seed", integer=True))
    return options


def julia_executable():
    binary = which("julia")
    if not binary:
        raise ValueError("Install Julia 1.10 or newer and add julia to the server's PATH.")
    return binary


def pbasex_repo_path(path):
    if not path:
        raise ValueError("Choose the local PBASEX-Julia checkout folder.")
    repo = Path(path).expanduser().resolve()
    try:
        project = tomllib.loads((repo / "Project.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        project = {}
    if (project.get("name") != "PBASEX" or project.get("uuid") != PBASEX_UUID
            or not (repo / "src" / "PBASEX.jl").is_file()):
        raise ValueError(f"Choose a PBASEX-Julia checkout with Project.toml and src/PBASEX.jl. Checked {repo}.")
    return repo


def _linked_repo(environment):
    try:
        project = tomllib.loads((environment / "Project.toml").read_text(encoding="utf-8"))
        manifest = tomllib.loads((environment / "Manifest.toml").read_text(encoding="utf-8"))
        entries = manifest["deps"]["PBASEX"]
        if project["deps"]["PBASEX"] == PBASEX_UUID and len(entries) == 1:
            entry = entries[0]
            if entry["uuid"] == PBASEX_UUID:
                return (environment / entry["path"]).resolve()
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
    return None


def _julia_environment():
    return {**os.environ, "JULIA_LOAD_PATH": os.pathsep.join(("@", "@stdlib"))}


def julia_runtime(repo=None):
    binary = julia_executable()
    legacy = GENERATED_DIR / "pbasex-julia"
    repo = pbasex_repo_path(repo if repo is not None else _linked_repo(legacy))
    dependency_file = repo / "Project.toml"
    dependency_declarations = dependency_file.read_bytes()

    parent = GENERATED_DIR / "pbasex-julia-checkouts"
    # Pkg resolves a path dependency from its Project.toml. Keep each resolved
    # environment separate so refreshing dependencies does not disrupt existing jobs.
    key = sha256(os.fsencode(os.path.normcase(repo)) + b"\0" + dependency_declarations).hexdigest()
    ready = parent / key
    if not ready.exists():
        parent.mkdir(parents=True, exist_ok=True)
        print(f"Preparing a Julia environment for {repo}", flush=True)
        with tempfile.TemporaryDirectory(prefix=".setup-", dir=parent) as temporary:
            staged = Path(temporary) / "project"
            staged.mkdir()
            subprocess.run([binary, "--startup-file=no", f"--project={staged}", "-e",
                            "using Pkg; Pkg.develop(path=ARGS[1]); Pkg.instantiate()",
                            str(repo)], env=_julia_environment(), check=True)
            if _linked_repo(staged) != repo:
                raise ValueError("Julia did not link the selected PBASEX-Julia checkout.")
            if dependency_file.read_bytes() != dependency_declarations:
                raise ValueError("PBASEX-Julia dependencies changed while Julia was preparing the environment. Retry the run.")
            try:
                staged.rename(ready)
            except OSError:
                # Another process can finish preparing the same checkout first.
                if _linked_repo(ready) != repo:
                    raise
    if _linked_repo(ready) != repo:
        raise ValueError(f"The cached Julia environment does not link the selected checkout. Check {ready}.")
    return binary, ready


def input_file(path, suffixes, label):
    path = Path(path).resolve()
    if path.suffix.lower() not in suffixes or not path.is_file():
        raise ValueError(f"Choose an existing {label} ({', '.join(suffixes)}).")
    return path


def run(mode, source, basis, output, *, pbasex_repo=None, spectrum_only=False, fit_record=None, **settings):
    if spectrum_only and (mode != "counts" or fit_record or settings.get("windows") or settings.get("e_min") is not None):
        raise ValueError("Spectrum-only runs require a count image and no integration windows.")
    extra_inputs = []
    if fit_record:
        from .pes_julia_fit import load_fit

        if settings.get("windows") or any(settings.get(key) is not None for key in ("e_min", "e_max", "delta_e")):
            raise ValueError("Choose a fitted peak record or explicit windows.")
        fit_record = input_file(fit_record, (".json",), "fitted peak record")
        settings["windows"] = [peak["window"] for peak in load_fit(fit_record)["peaks"]]
        extra_inputs.append(("fitted peak record", fit_record))
    options = analysis_options(mode, **settings)
    source = input_file(source, INPUT_SUFFIXES[mode], MODES[mode].lower())
    basis = input_file(basis, HDF5_SUFFIXES, "pBASEX basis")
    output = validate_distinct_output(output, [("source", source), ("basis", basis), *extra_inputs],
                                      output_label="results CSV")
    binary, environment = julia_runtime(pbasex_repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pes-julia-", dir=output.parent) as directory:
        folder = Path(directory)
        pending = folder / "result.csv"
        if mode == "counts":
            with source.open("rb") as stream:
                counts = np.load(stream, allow_pickle=False)
            if (not isinstance(counts, np.ndarray) or counts.ndim != 2 or not counts.size or counts.dtype.kind not in "iu"
                    or np.any(counts < 0) or counts.max() > 2**53):
                raise ValueError("The count image must be a nonempty 2D array of nonnegative integers representable in Float64.")
            with h5py.File(basis, "r") as basis_file:
                geometry = basis_file["geometry"].asstr()[()] if "geometry" in basis_file else "quadrant"
            source = folder / "counts.bin"
            # Pass native-endian Float64 values in Julia's column-major order.
            counts.T.astype(np.float64).tofile(source)
            estimator_arguments = [options["resolution"], *counts.shape, geometry]
        else:
            estimator_arguments = [options[key] for key in ("replicates", "confidence", "block_length", "seed")]
        arguments = [mode, source, basis, pending, options["alpha"] or 1, options["regularization"],
                     options["rotation"], options["center_x"] or "", options["center_y"] or "",
                     *estimator_arguments]
        windows_file = folder / "windows.tsv"
        windows_file.write_text("".join(f"{low!r}\t{high!r}\n" for low, high in options["windows"]), encoding="ascii")
        output_kind = ("spectrum" if spectrum_only else "peaks" if fit_record else
                       "legacy" if settings.get("e_min") is None else "uniform")
        arguments.extend((windows_file, output_kind))
        if mode == "shots":
            arguments.append(options["window_batch"])
        script = Path(__file__).with_name("pes_julia_bridge.jl")
        subprocess.run([binary, "--startup-file=no", f"--project={environment}", str(script),
                        *map(str, arguments)], env=_julia_environment(), check=True)
        pending.replace(output)
    print(f"Results saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("source")
    parser.add_argument("basis")
    parser.add_argument("--pbasex-repo", help="Path to the local PBASEX-Julia checkout")
    parser.add_argument("--spectrum-only", action="store_true", help="Run plain pBASEX on a count image")
    parser.add_argument("--fit-record", help="Use saved fitted FWHM bounds as integration windows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, help="eV per detector pixel squared")
    parser.add_argument("--resolution", type=float, default=1, help="Count-image bins per detector pixel")
    parser.add_argument("--rotation", type=int, choices=ROTATIONS, default=0)
    parser.add_argument("--regularization", type=float, default=0)
    parser.add_argument("--center-x", type=int, help="One-based column before rotation")
    parser.add_argument("--center-y", type=int, help="One-based row before rotation")
    parser.add_argument("--window", dest="windows", type=float, nargs=2, action="append", default=[])
    parser.add_argument("--e-min", type=float, help="First uniform-grid edge")
    parser.add_argument("--e-max", type=float, help="Last uniform-grid edge")
    parser.add_argument("--delta-e", type=float, help="Uniform-grid bin width")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--window-batch", type=int, default=1, help="Number of energy windows per shot-bootstrap batch")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Julia pBASEX: {exc}\n")


if __name__ == "__main__":
    main()
