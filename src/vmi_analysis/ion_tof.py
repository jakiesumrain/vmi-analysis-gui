"""Histogram Timepix3 arrivals relative to the selected LUNA trigger edge."""

import argparse
import json
from math import log, sqrt
from pathlib import Path

import h5py
import numpy as np

from vmi_analysis.paths import validate_distinct_output


NS_PER_TICK = 0.0001
BLOCK_SIZE = 262_144
ARRIVALS = {"pixels": ("PixelHits", "toa"), "clusters": ("Clusters", "ctoa")}
REFERENCES = {"TDC1Rising": 1, "TDC1Falling": 2, "TDC2Rising": 3, "TDC2Falling": 4}
DEFAULT_REFERENCE = "TDC1Rising"
DEFAULT_BINS = 1200
MAX_BINS = 20_000
GMD_SHIFTS = tuple(range(-5, 6))
MIN_GMD_PAIRS = 10


def _table(file, name, *fields, numeric="iu"):
    if not isinstance(file.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"Choose a LUNA HDF5 file with an embedded {name} dataset.")
    table = file[name]
    if (not isinstance(table, h5py.Dataset) or table.ndim != 1 or table.is_virtual
            or table.external or any(field not in (table.dtype.names or ())
                                     or table.dtype[field].kind not in numeric for field in fields)):
        raise ValueError(f"{name} must be a table with compatible numeric fields {', '.join(fields)}.")
    return table


def _ticks(values):
    if values.dtype.kind == "i" and np.any(values < 0):
        raise ValueError("LUNA arrival and trigger timestamps must be nonnegative.")
    return values.astype(np.uint64, copy=False)


def _trigger_times(file, reference):
    table = _table(file, "TDCEvents", "timestamp", "tdc_type")
    codes = h5py.check_enum_dtype(table.dtype["tdc_type"]) or REFERENCES
    if reference not in codes:
        raise ValueError(f"{reference} is not defined in this file's TDC types.")
    records = table.fields(("tdc_type", "timestamp"))[:]
    times = _ticks(records["timestamp"][records["tdc_type"] == codes[reference]])
    if times.size == 0:
        raise ValueError(f"No {reference} triggers found in TDCEvents.")
    if np.any(times[1:] < times[:-1]):
        times.sort()
    if np.any(times[1:] == times[:-1]):
        raise ValueError(f"Duplicate {reference} timestamps make trigger assignment ambiguous.")
    return times


def _tof_blocks(table, field, triggers):
    arrival_column = table.fields(field)
    for start in range(0, len(table), BLOCK_SIZE):
        arrivals = _ticks(arrival_column[start:start + BLOCK_SIZE])
        indices = np.searchsorted(triggers, arrivals, side="right") - 1
        matched = indices >= 0
        # Subtract on the integer detector clock to retain single-tick differences.
        matched_indices = indices[matched]
        differences = arrivals[matched] - triggers[matched_indices]
        yield differences * NS_PER_TICK, matched_indices, int(arrivals.size - differences.size)


def _gmd_metadata(path, trigger_count):
    with h5py.File(path, "r") as file:
        table = _table(file, "frames", "timestamp_ns", "gmd1", "gmd2", numeric="iuf")
        if table.dtype["timestamp_ns"].kind not in "iu":
            raise ValueError("The metadata's timestamp_ns field must contain integer nanoseconds.")
        frames = table.fields(("timestamp_ns", "gmd1", "gmd2"))[:]
    if len(frames) != trigger_count:
        issue = f"{len(frames):,} metadata rows for {trigger_count:,} selected triggers."
    elif np.any(frames["timestamp_ns"][1:] <= frames["timestamp_ns"][:-1]):
        issue = "The metadata's acquisition timestamps are not strictly increasing."
    else:
        issue = ""
    return {"gmd1_na": frames["gmd1"] * 1e9, "gmd2_na": frames["gmd2"] * 1e9,
            "gmd_alignment_issue": issue}


def gmd_correlations(gmd2, shot_counts):
    """Repeat the TPX3 diagnostic pairing GMD2[j] with counts[j + d]."""
    from scipy.stats import pearsonr, spearmanr

    if len(gmd2) != len(shot_counts):
        raise ValueError("GMD2 values and shot counts must have the same length.")
    correlations = np.full((len(GMD_SHIFTS), 2), np.nan)
    pairs = np.zeros(len(GMD_SHIFTS), dtype=np.int64)
    for index, shift in enumerate(GMD_SHIFTS):
        if abs(shift) >= len(gmd2):
            continue
        if shift < 0:
            values, counts = gmd2[-shift:], shot_counts[:shift]
        else:
            values, counts = gmd2[:len(gmd2) - shift], shot_counts[shift:]
        use = np.isfinite(values) & (values > 0) & (counts > 0)
        pairs[index] = use.sum()
        if pairs[index] < MIN_GMD_PAIRS:
            continue
        values, counts = values[use], counts[use]
        if values.min() == values.max() or counts.min() == counts.max():
            continue
        correlations[index] = pearsonr(values, counts).statistic, spearmanr(values, counts).statistic
    return correlations, pairs


def gmd_diagnostics(metadata, counts_by_source):
    """Compare GMD readings with counts from an existing TOF run."""
    trigger_count = len(next(iter(counts_by_source.values())))
    result = _gmd_metadata(metadata, trigger_count)
    if result["gmd_alignment_issue"]:
        return result
    result["gmd_shifts"] = np.asarray(GMD_SHIFTS)
    for source, shot_counts in counts_by_source.items():
        coefficients, pairs = gmd_correlations(result["gmd2_na"], shot_counts)
        result[f"{source}_gmd_correlation"] = coefficients
        result[f"{source}_gmd_pairs"] = pairs
    return result


def histogram(path, sources, reference=DEFAULT_REFERENCE, bins=DEFAULT_BINS, minimum=0.0, maximum=None,
              *, metadata=None):
    """Return a common bin grid and per-source counts and diagnostics."""
    if not 1 <= bins <= MAX_BINS:
        raise ValueError(f"Choose between 1 and {MAX_BINS:,} bins.")
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("TOF minimum must be finite and at least zero.")
    if maximum is not None and (not np.isfinite(maximum) or maximum <= minimum):
        raise ValueError("TOF maximum must be finite and greater than the minimum.")
    with h5py.File(path, "r") as file:
        triggers = _trigger_times(file, reference)
        columns = []
        for source in sources:
            name, field = ARRIVALS[source]
            columns.append((source, _table(file, name, field), field))

        if maximum is None:
            last = minimum
            for source, table, field in columns:
                print(f"Finding TOF range for {source} ({len(table):,} entries)…")
                for values, _, _ in _tof_blocks(table, field, triggers):
                    if values.size:
                        last = max(last, float(values.max()))
            maximum = last if last > minimum else minimum + 1.0
        edges = np.linspace(minimum, maximum, bins + 1)
        if not np.all(edges[1:] > edges[:-1]):
            raise ValueError("The TOF range is too narrow for this bin count.")
        result = {"edges": edges, "triggers": len(triggers), "trigger_times": triggers}
        counts_by_source = {}
        for source, table, field in columns:
            print(f"Building TOF histogram for {source}…")
            counts = np.zeros(bins, dtype=np.int64)
            shot_counts = np.zeros(len(triggers), dtype=np.int64)
            unmatched = 0
            for values, assigned, before_first in _tof_blocks(table, field, triggers):
                counts += np.histogram(values, bins=edges)[0]
                unmatched += before_first
                np.add.at(shot_counts, assigned, 1)
            outside = len(table) - unmatched - int(counts.sum())
            result[source] = counts
            result[f"{source}_unmatched"] = unmatched
            result[f"{source}_outside"] = outside
            result[f"{source}_shot_counts"] = counts_by_source[source] = shot_counts
            print(f"{source}: {int(counts.sum()):,} plotted, {unmatched:,} without a preceding trigger, "
                  f"{outside:,} outside the TOF range.")
    if metadata is not None:
        gmd = gmd_diagnostics(metadata, counts_by_source)
        issue = gmd["gmd_alignment_issue"]
        if issue:
            print(f"GMD alignment skipped: {issue}")
        result.update(gmd)
    return result


def filtered_histogram(path, tof_result, metadata, reference, low, high, shift):
    """Select triggers by GMD2 and bin their arrivals on the original TOF grid."""
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("Enter increasing, finite GMD2 limits.")
    with np.load(tof_result, allow_pickle=False) as original:
        if "trigger_times" not in original:
            raise ValueError("Plot TOF again before filtering this older result.")
        edges = original["edges"]
        original_triggers = original["trigger_times"]
        sources = [source for source in ARRIVALS if source in original]
    gmd = _gmd_metadata(metadata, len(original_triggers))
    if gmd["gmd_alignment_issue"]:
        raise ValueError(gmd["gmd_alignment_issue"])
    readings = gmd["gmd2_na"]
    if abs(shift) >= len(readings):
        raise ValueError("The alignment shift leaves no paired frames.")
    metadata_rows = np.flatnonzero(np.isfinite(readings) & (readings >= low) & (readings <= high))
    camera_rows = metadata_rows + shift
    selected = np.zeros(len(readings), dtype=bool)
    selected[camera_rows[(camera_rows >= 0) & (camera_rows < len(readings))]] = True
    selected_count = np.count_nonzero(selected)
    result = {"edges": edges, "selected_triggers": selected_count, "triggers": len(selected),
              "trigger_times": original_triggers, "selected_mask": selected}
    print(f"GMD2 {low:g} to {high:g} nA, shift {shift:+d}: {selected_count:,} selected triggers.")
    with h5py.File(path, "r") as file:
        triggers = _trigger_times(file, reference)
        if not np.array_equal(triggers, original_triggers):
            raise ValueError("The selected triggers changed. Plot TOF again before filtering.")
        for source in sources:
            name, field = ARRIVALS[source]
            table = _table(file, name, field)
            counts = np.zeros(len(edges) - 1, dtype=np.int64)
            if selected_count:
                print(f"Building GMD2-filtered TOF histogram for {source}…")
                for values, assigned, _ in _tof_blocks(table, field, triggers):
                    counts += np.histogram(values[selected[assigned]], bins=edges)[0]
            result[source] = counts
            print(f"{source}: {int(counts.sum()):,} selected counts in the TOF range.")
    return result


def fit_filtered_peaks(path, filtered_result, reference, sources, region, estimates, baseline):
    """Fit Gaussian peaks and count exact arrivals in each fitted ±3σ window."""
    from vmi_analysis.pes import PeakGuess, fit_peaks

    guesses = [PeakGuess("Gaussian", center, width) for center, width in estimates]
    results = {}
    errors = {}
    with np.load(filtered_result, allow_pickle=False) as filtered, h5py.File(path, "r") as file:
        if "selected_mask" not in filtered:
            raise ValueError("Apply the GMD2 range again before fitting this older spectrum.")
        edges = filtered["edges"]
        centers = (edges[:-1] + edges[1:]) / 2
        selected = filtered["selected_mask"]
        selected_count = int(filtered["selected_triggers"])
        triggers = _trigger_times(file, reference)
        if not np.array_equal(triggers, filtered["trigger_times"]):
            raise ValueError("The selected triggers changed. Plot TOF and apply the GMD2 range again.")
        for source in sources:
            if source not in ARRIVALS or source not in filtered:
                raise ValueError("Choose a count source from the filtered TOF result.")
            try:
                x, fitted, background, _, residual, peaks = fit_peaks(
                    centers, filtered[source], region, guesses, baseline,
                )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                errors[source] = str(exc)
                continue
            for peak in peaks:
                half_width = 3 * peak["gaussian_fwhm"] / (2 * sqrt(2 * log(2)))
                peak.update(low=peak["center"] - half_width, high=peak["center"] + half_width, counts=0)
            name, field = ARRIVALS[source]
            print(f"Counting {source} in {len(peaks)} Gaussian peak windows…")
            for values, assigned, _ in _tof_blocks(_table(file, name, field), field, triggers):
                arrivals = values[selected[assigned]]
                for peak in peaks:
                    peak["counts"] += int(np.count_nonzero((arrivals >= peak["low"]) & (arrivals < peak["high"])))
            for peak in peaks:
                peak["yield_per_trigger"] = peak["counts"] / selected_count
            results[f"{source}_curve"] = np.stack((x, fitted, background, residual))
            results[f"{source}_peaks"] = json.dumps(peaks)
    return {**results, "fit_errors": json.dumps(errors), "selected_triggers": selected_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Histogram or peak fit output (.npz)")
    parser.add_argument("--metadata", type=Path, help="Matching TPX3 Dump acquisition metadata (.h5)")
    parser.add_argument("--tof-result", type=Path, help="Existing unfiltered histogram (.npz)")
    parser.add_argument("--gmd-min", type=float, help="Inclusive GMD2 minimum in nA")
    parser.add_argument("--gmd-max", type=float, help="Inclusive GMD2 maximum in nA")
    parser.add_argument("--gmd-shift", type=int, help="Pair GMD row j with camera frame j + shift")
    parser.add_argument("--filtered-result", type=Path, help="Existing GMD2-filtered histogram (.npz)")
    parser.add_argument("--fit-spec", type=json.loads, help="Gaussian peak fit settings as JSON")
    parser.add_argument("--source", choices=(*ARRIVALS, "both"), default="pixels")
    parser.add_argument("--reference", choices=REFERENCES, default=DEFAULT_REFERENCE)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--minimum", type=float, default=0.0, help="Minimum TOF in ns")
    parser.add_argument("--maximum", type=float, help="Maximum TOF in ns (default: largest observed TOF)")
    args = parser.parse_args()
    try:
        output = validate_distinct_output(args.output, [("LUNA HDF5 input", args.input),
                                                       ("acquisition metadata", args.metadata),
                                                       ("unfiltered TOF result", args.tof_result),
                                                       ("filtered TOF result", args.filtered_result)])
        if args.filtered_result is not None or args.fit_spec is not None:
            if args.filtered_result is None or not isinstance(args.fit_spec, dict):
                raise ValueError("Peak fitting requires the filtered TOF result and fit settings.")
            try:
                result = fit_filtered_peaks(args.input, args.filtered_result, args.reference, **args.fit_spec)
            except TypeError as exc:
                raise ValueError(f"Invalid TOF peak fit settings. {exc}") from exc
        elif any(value is not None for value in (args.tof_result, args.gmd_min, args.gmd_max, args.gmd_shift)):
            if any(value is None for value in (args.tof_result, args.metadata, args.gmd_min,
                                               args.gmd_max, args.gmd_shift)):
                raise ValueError("Filtering requires the unfiltered TOF result, metadata, GMD limits, and shift.")
            result = filtered_histogram(args.input, args.tof_result, args.metadata, args.reference,
                                        args.gmd_min, args.gmd_max, args.gmd_shift)
        else:
            sources = tuple(ARRIVALS) if args.source == "both" else (args.source,)
            result = histogram(args.input, sources, args.reference, args.bins, args.minimum, args.maximum,
                               metadata=args.metadata)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            np.savez(handle, **result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"TOF: {exc}\n")


if __name__ == "__main__":
    main()
