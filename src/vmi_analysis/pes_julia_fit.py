"""Fit a saved pBASEX spectrum and write peak records with FWHM windows."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np

from .paths import validate_distinct_output
from .pes import BASELINE_PARAMETERS, PeakGuess, fit_peaks

FIT_SCHEMA = "vmi-pbasex-peak-fit/1"
PROFILES = ("Gaussian", "Voigt")


def load_spectrum(path):
    columns = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["kind"] == "native":
                columns.setdefault(int(row["order"]), []).append(
                    (float(row["coordinate"]), float(row["value"])))
    isotropic = np.asarray(columns.pop(0, ()), dtype=float)
    if (isotropic.ndim != 2 or len(isotropic) < 2 or not np.isfinite(isotropic).all()
            or not np.all(np.diff(isotropic[:, 0]) > 0)):
        raise ValueError("The pBASEX run has no usable native spectrum.")
    coordinate, intensity = isotropic.T
    beta = {}
    for order, values in columns.items():
        angular = np.asarray(values)
        if not np.array_equal(angular[:, 0], coordinate):
            raise ValueError("The spectrum and β coordinates do not match.")
        beta[order] = angular[:, 1]
    return coordinate, intensity, beta


def fit_settings(region, peaks, baseline, *, auto_profile="Gaussian"):
    if not isinstance(region, (tuple, list)) or len(region) != 2 or None in region:
        raise ValueError("Select a fit range or enter both endpoints.")
    if any(type(value) not in (float, int) or not np.isfinite(value) for value in region) or region[0] >= region[1]:
        raise ValueError("Enter an increasing, finite fit range.")
    if not isinstance(baseline, str) or baseline not in BASELINE_PARAMETERS:
        raise ValueError("Choose None, Constant, or Linear for the baseline.")
    if auto_profile not in PROFILES:
        raise ValueError("Choose a valid peak profile.")
    if not isinstance(peaks, (tuple, list)):
        raise ValueError("Provide one object per peak estimate.")
    low, high = region
    guesses = []
    for peak in peaks:
        if not isinstance(peak, dict) or peak.keys() - {"profile", "center", "gaussian_fwhm", "lorentzian_fwhm"}:
            raise ValueError("Enter a profile, center, and initial widths for each peak.")
        profile = peak.get("profile", "Gaussian")
        center, width = peak.get("center"), peak.get("gaussian_fwhm")
        wing = (peak.get("lorentzian_fwhm") or 0) if profile == "Voigt" else 0
        if (profile not in PROFILES or any(type(v) not in (int, float) or not np.isfinite(v) for v in (center, width, wing))
                or not low <= center <= high or not 0 < width <= high - low or not 0 <= wing <= high - low):
            raise ValueError("Peak centers must lie in the fit range. Enter valid initial widths.")
        guesses.append(PeakGuess(profile, center, width, wing))
    return [float(low), float(high)], [asdict(guess) for guess in guesses], baseline


def load_fit(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != FIT_SCHEMA
            or not isinstance(record.get("spectrum"), str)
            or not isinstance(record.get("peaks"), list) or not record["peaks"]):
        raise ValueError("Choose a saved pBASEX peak fit record.")
    for peak in record["peaks"]:
        if not isinstance(peak, dict) or not isinstance(peak.get("window"), list) or len(peak["window"]) != 2:
            raise ValueError("The fit record has an invalid FWHM window.")
        center, fwhm = peak.get("center"), peak.get("fwhm")
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in (center, fwhm)) or fwhm <= 0:
            raise ValueError("The fit record has an invalid peak center or FWHM.")
        if peak["window"] != [center - fwhm / 2, center + fwhm / 2]:
            raise ValueError("The saved window does not match its peak center and FWHM.")
    return record


def run(spectrum, output, *, region, peaks=(), baseline="Constant", auto_profile="Gaussian"):
    spectrum = Path(spectrum).resolve()
    output = validate_distinct_output(output, [("native spectrum", spectrum)], output_label="peak fit record")
    region, guesses, baseline = fit_settings(region, peaks, baseline, auto_profile=auto_profile)
    energy, intensity, _ = load_spectrum(spectrum)
    x, _, background, _, _, fitted = fit_peaks(
        energy, intensity, region, [PeakGuess(**guess) for guess in guesses], baseline, auto_profile=auto_profile)
    for peak in fitted:
        center, half = peak["center"], peak["fwhm"] / 2
        low, high = center - half, center + half
        if not energy[0] <= low < high <= energy[-1]:
            raise ValueError(f"Peak {peak['peak']}'s fitted FWHM extends outside the native spectrum.")
        peak["window"] = [low, high]
    record = dict(schema=FIT_SCHEMA, spectrum=str(spectrum), region=region, baseline=baseline,
                  auto_profile=auto_profile, guesses=guesses, peaks=fitted, fit_x=x.tolist(), background=background.tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pbasex-fit-", dir=output.parent) as directory:
        pending = Path(directory) / "fit.json"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pending.replace(output)
    print(f"Saved {len(fitted)} fitted FWHM windows to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum")
    parser.add_argument("--output", required=True)
    parser.add_argument("--region", nargs=2, type=float, required=True)
    parser.add_argument("--peaks", type=json.loads, default=[])
    parser.add_argument("--auto-profile", choices=PROFILES, default="Gaussian")
    parser.add_argument("--baseline", choices=BASELINE_PARAMETERS, default="Constant")
    try:
        run(**vars(parser.parse_args()))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"pBASEX peak fit: {exc}\n")


if __name__ == "__main__":
    main()
