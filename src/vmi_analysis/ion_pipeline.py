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
COLUMNS = ("folder", "delay_mm", "peak", "frames", "selected_frames", "segments",
           "center_ns", "sigma_ns", "window_min_ns", "window_max_ns", "counts",
           "hits_per_frame", "poisson_error", "status")


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
    workers = options.get("workers", 1)
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Choose between 1 and 8 delay positions to process at once.")
    if options["gate_gmd"]:
        low, high = options["gmd_min"], options["gmd_max"]
        if not all(x is not None and isfinite(x) for x in (low, high)) or low > high:
            raise ValueError("Enter finite GMD2 limits with minimum at or below maximum.")


def measure(position, options, selected, fit_peak):
    selected_count = int(np.count_nonzero(selected))
    gated_tof = position.tof_ns[selected[position.frame_index]]
    for peak in options["peaks"]:
        row = dict(peak=peak["name"], delay_mm=position.delay_mm, frames=len(selected),
                   selected_frames=selected_count, segments=len(position.segment_lengths))
        if not selected_count:
            yield dict(row, status="No frames pass the GMD2 gate.")
            continue
        fit = fit_peak(position.tof_ns, peak["low"], peak["high"], peak["center"], options["fit_bins"])
        if fit is None:
            yield dict(row, status="Gaussian fit unavailable in the requested range.")
            continue
        center, width = fit["mu"], fit["sigma"]
        half_width = options["window_sigma"] * width
        low, high = center - half_width, center + half_width
        n = int(np.count_nonzero((gated_tof >= low) & (gated_tof < high)))
        yield dict(row, center_ns=center, sigma_ns=width, window_min_ns=low, window_max_ns=high,
                   counts=n, hits_per_frame=n / selected_count, poisson_error=sqrt(n) / selected_count,
                   status="OK")


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
    from vmi_analysis import ion_pipeline_backend as backend

    validate_settings(options)
    points = positions(root)
    if not points:
        raise ValueError("No delay-position folders were found.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(options, indent=2, ensure_ascii=False), encoding="utf-8")

    def process(task):
        index, (delay, folder) = task
        print(f"[{index}/{len(points)}] {folder.name}", flush=True)
        try:
            data = backend.decode(folder, delay, output, options)
            selected = backend.select_frames(data, options) if options["gate_gmd"] else np.ones(
                len(data.gmd2_na), dtype=bool
            )
            measured = list(measure(data, options, selected, backend.fit_peak))
        except (OSError, ValueError, RuntimeError) as exc:
            measured = [dict(peak=peak["name"], delay_mm=delay, status=str(exc)) for peak in options["peaks"]]
        return [dict(row, folder=folder.name) for row in measured]

    rows = []
    with (output / CSV_NAME).open("w", encoding="utf-8", newline="") as file, \
            ThreadPoolExecutor(max_workers=options.get("workers", 1)) as executor:
        csv_writer = csv.DictWriter(file, COLUMNS)
        csv_writer.writeheader()
        for measured in executor.map(process, enumerate(points, 1)):
            rows.extend(measured)
            csv_writer.writerows(measured)
            file.flush()
            for row in measured:
                summary = (f"{row['counts']:,} / {row['selected_frames']:,} frames"
                           if row["status"] == "OK" else row["status"])
                print(f"  {row['folder']} · {row['peak']}: {summary}", flush=True)
    save_plot(output / FIGURE_NAME, rows, [peak["name"] for peak in options["peaks"]], backend.y_label(options))
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
