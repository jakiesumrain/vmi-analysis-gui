"""Run TPX3 decoding and independent peak-yield measurements across delay positions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
from math import isfinite, sqrt
from pathlib import Path
import re

import numpy as np


POSITION_NAME = re.compile(r"^(?:Tr)?(\d+)(?:p(\d+))?(?:_\d+)?_ion$")
DELAY_IN_FILE = re.compile(r"odl_pos_([+-]?\d+(?:\.\d+)?)")
DEFAULT_PEAKS = [
    {"name": "Xe+", "low": 10150, "high": 11050, "center": 10545},
    {"name": "Xe2+", "low": 8800, "high": 9600, "center": 9155},
]
CSV_NAME = "ion-yields.csv"
FIGURE_NAME = "ion-yields.png"
RATIO_NAME = "ion-yield-ratio.csv"
COLUMNS = ("folder", "delay_mm", "peak", "frames", "selected_frames", "segments",
           "center_ns", "sigma_ns", "window_min_ns", "window_max_ns", "counts",
           "hits_per_frame", "poisson_error", "status")
GMD2_REFERENCE = "gmd2_reference_na"


@dataclass
class PositionData:
    delay_mm: float
    tof_ns: np.ndarray
    frame_index: np.ndarray
    gmd2_na: np.ndarray
    segment_lengths: list[int]


def positions(root):
    found = []
    for directory in root.iterdir():
        match = POSITION_NAME.fullmatch(directory.name)
        if match and directory.is_dir():
            whole, fraction = match.groups()
            delay = float(f"{whole}.{fraction or '0'}")
            found.append((delay, directory))
    return sorted(found, key=lambda position: (position[0], position[1].name))


def delay_from_files(paths, fallback):
    positions = [float(match[1]) for path in paths if (match := DELAY_IN_FILE.search(path.name))]
    return float(np.median(positions)) if positions else fallback


def validate_settings(options):
    if options.get("backend", "luna") not in ("luna", "fallback"):
        raise ValueError("Choose the LUNA or fallback decoder.")
    weighted = options.get("weight_by_gmd2", False)
    if type(weighted) is not bool:
        raise ValueError("Use true or false for GMD2 weighting.")
    if weighted and (not options["gate_gmd"] or options.get("backend") == "fallback"):
        raise ValueError("GMD2 weighting requires an enabled GMD2 gate in the LUNA pipeline.")
    peaks = options["peaks"]
    if not peaks:
        raise ValueError("Add at least one peak.")
    names = set()
    for peak in peaks:
        name, low, high, center = (peak[key] for key in ("name", "low", "high", "center"))
        if not name or name in names:
            raise ValueError("Give each peak a distinct name.")
        names.add(name)
        if not all(x is not None and isfinite(x) for x in (low, high, center)) or not low < center < high:
            raise ValueError(f"Enter finite TOF limits and an initial center inside the range for {name}.")
    bins, k, shift = (options[key] for key in ("fit_bins", "window_sigma", "gmd_shift"))
    if not isinstance(bins, int) or not 4 <= bins <= 20_000:
        raise ValueError("Choose a whole number of fit bins between 4 and 20,000.")
    if k is None or not isfinite(k) or k <= 0:
        raise ValueError("The count-window half-width must be a positive number of standard deviations.")
    if not isinstance(shift, int):
        raise ValueError("Use a whole-number GMD shift.")
    if options.get("fit_tof", "gated") not in ("ungated", "gated"):
        raise ValueError("Choose gated or ungated TOF for peak-window fitting.")
    workers = options.get("workers", 1)
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Choose between 1 and 8 delay positions to process at once.")
    if options["gate_gmd"]:
        low, high = options["gmd_min"], options["gmd_max"]
        if not all(x is not None and isfinite(x) for x in (low, high)) or low > high:
            raise ValueError("Enter finite GMD2 limits with minimum at or below maximum.")


def measure(position, options, selected, fit_peak, *, frame_weights=None):
    selected_count = int(np.count_nonzero(selected))
    keep = selected[position.frame_index]
    gated_tof = position.tof_ns[keep]
    arrival_weights = frame_weights[position.frame_index[keep]] if frame_weights is not None else None
    fit_gated = options.get("fit_tof", "gated") == "gated"
    fit_tof = gated_tof if fit_gated else position.tof_ns
    fit_options = {"weights": arrival_weights} if fit_gated and arrival_weights is not None else {}
    for peak in options["peaks"]:
        row = dict(peak=peak["name"], delay_mm=position.delay_mm, frames=len(selected),
                   selected_frames=selected_count, segments=len(position.segment_lengths))
        if not selected_count:
            yield dict(row, status="No frames pass the GMD2 gate.")
            continue
        fit = fit_peak(fit_tof, peak["low"], peak["high"], peak["center"], options["fit_bins"], **fit_options)
        if fit is None:
            yield dict(row, status="Gaussian fit unavailable in the requested range.")
            continue
        center, width = fit["mu"], fit["sigma"]
        half_width = options["window_sigma"] * width
        low, high = center - half_width, center + half_width
        inside = (gated_tof >= low) & (gated_tof < high)
        if arrival_weights is None:
            n = int(np.count_nonzero(inside))
            error = sqrt(n)
        else:
            weights = arrival_weights[inside]
            n = float(weights.sum())
            error = float(np.linalg.norm(weights))
        yield dict(row, center_ns=center, sigma_ns=width, window_min_ns=low, window_max_ns=high,
                   counts=n, hits_per_frame=n / selected_count, poisson_error=error / selected_count,
                   status="OK")


def yield_ratios(rows, numerator, denominator):
    """Return (folder, delay in mm, ratio) for each folder with either peak."""
    pairs = {}
    for row in rows:
        if row["peak"] in (numerator, denominator):
            pairs.setdefault(row["folder"], {})[row["peak"]] = row
    for folder, pair in sorted(pairs.items(), key=lambda item: float(next(iter(item[1].values()))["delay_mm"])):
        first, second = pair.get(numerator), pair.get(denominator)
        ratio = None
        if first and second and first["status"] == second["status"] == "OK":
            a, b = float(first["hits_per_frame"]), float(second["hits_per_frame"])
            if isfinite(a) and isfinite(b) and b > 0 and isfinite(quotient := a / b):
                ratio = quotient
        yield folder, float(next(iter(pair.values()))["delay_mm"]), ratio


def save_plot(path, rows, names, ylabel):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 4.9), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    for name in names:
        points = sorted((row for row in rows if row["peak"] == name), key=lambda row: row["delay_mm"])
        if points:
            ax.errorbar([row["delay_mm"] for row in points],
                        [row.get("hits_per_frame", np.nan) for row in points],
                        yerr=[row.get("poisson_error", np.nan) for row in points],
                        marker="o", markersize=3.5, linewidth=1, capsize=2, label=name)
    ax.set_xlabel("Delay-line position (mm)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    if any(row["status"] == "OK" for row in rows):
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No successful peak measurements", ha="center", transform=ax.transAxes)
    fig.savefig(path, dpi=180)


def run(root, output, options):
    validate_settings(options)
    weighted = options.get("weight_by_gmd2", False)
    if options.get("backend") == "fallback":
        from vmi_analysis import ion_fallback_backend as backend
    else:
        from vmi_analysis import ion_pipeline_backend as backend
    points = positions(root)
    if not points:
        raise ValueError("No delay-position folders were found.")
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json"
    settings_path.write_text(json.dumps(options, indent=2, ensure_ascii=False), encoding="utf-8")

    def process(task):
        index, (delay, folder) = task
        print(f"[{index}/{len(points)}] {folder.name}", flush=True)
        gmd2_sum = eligible_frames = 0
        try:
            data = backend.decode(folder, delay, output, options)
            weights = None
            if weighted:
                aligned = backend.gated_gmd2(data, options)
                selected = np.isfinite(aligned)
                eligible_frames = int(selected.sum())
                gmd2_sum = float(aligned[selected].sum())
                weights = np.zeros_like(aligned)
                with np.errstate(over="ignore", divide="ignore"):
                    weights[selected] = 1.0 / aligned[selected]
                if not np.isfinite(weights).all():
                    raise ValueError("GMD2 weights are not finite. Increase the GMD2 minimum.")
            else:
                selected = (backend.select_frames(data, options) if options["gate_gmd"]
                            else np.ones(len(data.gmd2_na), dtype=bool))
            measured = list(measure(data, options, selected, backend.fit_peak, frame_weights=weights))
        except (OSError, ValueError, RuntimeError) as exc:
            measured = [dict(peak=peak["name"], delay_mm=delay, status=str(exc)) for peak in options["peaks"]]
        return [dict(row, folder=folder.name) for row in measured], gmd2_sum, eligible_frames

    def write_rows(measured):
        csv_writer.writerows(measured)
        file.flush()
        for row in measured:
            summary = row["status"]
            if summary == "OK":
                counts = f"{row['counts']:,.6g} weighted counts" if weighted else f"{row['counts']:,}"
                summary = f"{counts} / {row['selected_frames']:,} frames"
            print(f"  {row['folder']} · {row['peak']}: {summary}", flush=True)

    rows = []
    total_gmd2 = selected_frames = 0
    with (output / CSV_NAME).open("w", encoding="utf-8", newline="") as file, \
            ThreadPoolExecutor(max_workers=options.get("workers", 1)) as executor:
        csv_writer = csv.DictWriter(file, (*COLUMNS, GMD2_REFERENCE) if weighted else COLUMNS)
        csv_writer.writeheader()
        for completed, (measured, gmd2_sum, eligible_frames) in enumerate(executor.map(process, enumerate(points, 1)), 1):
            rows.extend(measured)
            if weighted:
                total_gmd2 += gmd2_sum
                selected_frames += eligible_frames
            else:
                write_rows(measured)
            if options.get("backend") != "fallback":
                stage = "processed" if weighted else "written"
                print(f"Delay positions {stage}: {completed}/{len(points)}", flush=True)
        if weighted:
            if selected_frames:
                # The common numerator only scales the gated histogram's height.
                # Fit with 1/GMD2, then scale counts and uncertainties once the scan mean is known.
                reference = total_gmd2 / selected_frames
                if not isfinite(reference):
                    raise ValueError("The scan-wide mean GMD2 is not finite.")
                settings_path.write_text(json.dumps({**options, GMD2_REFERENCE: reference}, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
                print(f"Scan-wide GMD2 reference: {reference:.6g} nA from {selected_frames:,} selected frames.", flush=True)
                for row in rows:
                    row[GMD2_REFERENCE] = reference
                    if row["status"] == "OK":
                        for key in ("counts", "hits_per_frame", "poisson_error"):
                            row[key] *= reference
            write_rows(rows)
    names = [peak["name"] for peak in options["peaks"]]
    if options.get("backend") != "fallback" and len(names) > 1:
        numerator, denominator = names[:2]
        with (output / RATIO_NAME).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("folder", "delay_mm", "numerator", "denominator", "ratio"))
            writer.writerows((folder, delay, numerator, denominator, ratio)
                             for folder, delay, ratio in yield_ratios(rows, numerator, denominator))
    save_plot(output / FIGURE_NAME, rows, names, backend.y_label(options))
    print(f"{sum(row['status'] == 'OK' for row in rows)}/{len(rows)} peak measurements completed.", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=json.loads)
    args = parser.parse_args()
    try:
        run(args.input, args.output_dir, args.settings)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Ion Pipeline: {exc}\n")


if __name__ == "__main__":
    main()
