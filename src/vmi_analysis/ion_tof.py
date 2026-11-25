"""Histogram Timepix3 arrivals relative to the selected LUNA trigger edge."""

import argparse
from contextlib import ExitStack
import json
from math import log, sqrt
from pathlib import Path

import h5py
import numpy as np

from vmi_analysis.paths import validate_distinct_output
from vmi_analysis.tof_bins import tof_bin_edges
from vmi_analysis.ion_acquisition import segment_files


NS_PER_TICK = 0.0001
BLOCK_SIZE = 262_144
ARRIVALS = {"pixels": ("PixelHits", "toa"), "clusters": ("Clusters", "ctoa")}
REFERENCES = {"TDC1Rising": 1, "TDC1Falling": 2, "TDC2Rising": 3, "TDC2Falling": 4}
DEFAULT_REFERENCE = "TDC1Rising"
DEFAULT_BINS = 1200
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


def assigned_arrival_blocks(table, field, triggers, *, coordinates=()):
    """Keep selected record fields aligned with their trigger-relative TOFs."""
    arrival_columns = table.fields((field, *coordinates))
    for start in range(0, len(table), BLOCK_SIZE):
        records = arrival_columns[start:start + BLOCK_SIZE]
        arrivals = _ticks(records[field])
        indices = np.searchsorted(triggers, arrivals, side="right") - 1
        matched = indices >= 0
        # Subtract on the integer detector clock to retain single-tick differences.
        matched_indices = indices[matched]
        differences = arrivals[matched] - triggers[matched_indices]
        yield (records[matched], differences * NS_PER_TICK, matched_indices,
               int(arrivals.size - differences.size))


def _tof_blocks(table, field, triggers):
    for _, values, indices, unmatched in assigned_arrival_blocks(table, field, triggers):
        yield values, indices, unmatched


def _gmd_metadata(path, trigger_count, segment_offsets=None):
    if Path(path).suffix.lower() == ".json":
        files = segment_files(path)
        if segment_offsets is None or len(segment_offsets) != len(files) + 1:
            raise ValueError("The TOF result and metadata collection have different acquisition segments.")
        readings = [_gmd_metadata(metadata, int(stop - start))
                    for (_, metadata), start, stop in zip(files, segment_offsets[:-1], segment_offsets[1:])]
        return {"gmd1_na": np.concatenate([r["gmd1_na"] for r in readings]),
                "gmd2_na": np.concatenate([r["gmd2_na"] for r in readings]),
                "gmd_alignment_issue": " ".join(r["gmd_alignment_issue"] for r in readings
                                                if r["gmd_alignment_issue"])}
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


def gmd_count_pairs(gmd2, shot_counts, shift, segment_offsets=None):
    """Return positive GMD2[j] and counts[j + shift] pairs within each acquisition."""
    if len(gmd2) != len(shot_counts):
        raise ValueError("GMD2 values and shot counts must have the same length.")
    boundaries = [0, len(gmd2)] if segment_offsets is None else segment_offsets
    readings, counts = [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        first = start + max(0, -shift)
        last = stop - max(0, shift)
        if first < last:
            readings.append(gmd2[first:last])
            counts.append(shot_counts[first + shift:last + shift])
    if not readings:
        return gmd2[:0], shot_counts[:0]
    values, counts = np.concatenate(readings), np.concatenate(counts)
    valid = np.isfinite(values) & (values > 0) & (counts > 0)
    return values[valid], counts[valid]


def gmd_correlations(gmd2, shot_counts, segment_offsets=None):
    """Calculate Pearson and Spearman correlations for each diagnostic shift."""
    from scipy.stats import pearsonr, spearmanr

    correlations = np.full((len(GMD_SHIFTS), 2), np.nan)
    pairs = np.zeros(len(GMD_SHIFTS), dtype=np.int64)
    for index, shift in enumerate(GMD_SHIFTS):
        values, counts = gmd_count_pairs(gmd2, shot_counts, shift, segment_offsets)
        pairs[index] = len(values)
        if pairs[index] < MIN_GMD_PAIRS:
            continue
        if values.min() == values.max() or counts.min() == counts.max():
            continue
        correlations[index] = pearsonr(values, counts).statistic, spearmanr(values, counts).statistic
    return correlations, pairs


def gmd_diagnostics(metadata, counts_by_source, segment_offsets=None):
    """Compare GMD readings with counts from an existing TOF run."""
    trigger_count = len(next(iter(counts_by_source.values())))
    result = _gmd_metadata(metadata, trigger_count, segment_offsets)
    if result["gmd_alignment_issue"]:
        return result
    result["gmd_shifts"] = np.asarray(GMD_SHIFTS)
    for source, shot_counts in counts_by_source.items():
        coefficients, pairs = gmd_correlations(result["gmd2_na"], shot_counts, segment_offsets)
        result[f"{source}_gmd_correlation"] = coefficients
        result[f"{source}_gmd_pairs"] = pairs
    return result


def histogram(path, sources, reference=DEFAULT_REFERENCE, bins=DEFAULT_BINS, minimum=0.0, maximum=None,
              *, metadata=None, bin_size_ns=None):
    """Return a common bin grid and per-source counts and diagnostics."""
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("TOF minimum must be finite and at least zero.")
    if maximum is not None and (not np.isfinite(maximum) or maximum <= minimum):
        raise ValueError("TOF maximum must be finite and greater than the minimum.")
    paths = segment_files(path)
    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(data, "r")) for data, _ in paths]
        trigger_sets = [_trigger_times(file, reference) for file in files]
        boundaries = np.r_[0, np.cumsum([len(times) for times in trigger_sets])]
        if maximum is None:
            last = minimum
            for file, triggers in zip(files, trigger_sets):
                for source in sources:
                    name, field = ARRIVALS[source]
                    for values, _, _ in _tof_blocks(_table(file, name, field), field, triggers):
                        if values.size:
                            last = max(last, float(values.max()))
            maximum = last if last > minimum else minimum + 1.0
        edges = tof_bin_edges(minimum, maximum, bins, bin_size_ns=bin_size_ns)
        result = {"edges": edges, "triggers": int(boundaries[-1]),
                  "trigger_times": np.concatenate(trigger_sets), "segment_offsets": boundaries}
        counts_by_source = {}
        for source in sources:
            counts = np.zeros(len(edges) - 1, dtype=np.int64)
            shot_counts = np.zeros(int(boundaries[-1]), dtype=np.int64)
            unmatched = total = 0
            name, field = ARRIVALS[source]
            for file, triggers, start in zip(files, trigger_sets, boundaries[:-1]):
                table = _table(file, name, field)
                total += len(table)
                for values, assigned, before_first in _tof_blocks(table, field, triggers):
                    counts += np.histogram(values, bins=edges)[0]
                    unmatched += before_first
                    np.add.at(shot_counts, assigned + start, 1)
            outside = total - unmatched - int(counts.sum())
            result[source] = counts
            result[f"{source}_unmatched"] = unmatched
            result[f"{source}_outside"] = outside
            result[f"{source}_shot_counts"] = counts_by_source[source] = shot_counts
            print(f"{source}: {int(counts.sum()):,} plotted, {unmatched:,} without a preceding trigger, "
                  f"{outside:,} outside the TOF range.")
    if metadata is not None:
        result.update(gmd_diagnostics(metadata, counts_by_source, boundaries))
    return result


def gmd_limits(readings, low, high):
    finite = readings[np.isfinite(readings)]
    if not finite.size:
        raise ValueError("The GMD2 data contain no finite values.")
    low = float(finite.min()) if low is None else low
    high = float(finite.max()) if high is None else high
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Enter finite GMD2 limits with minimum at or below maximum.")
    return low, high


def gmd2_by_frame(readings, segment_offsets, shift):
    """Align GMD2 row j to frame j + shift without crossing segments."""
    aligned = np.full(len(readings), np.nan)
    for start, stop in zip(segment_offsets[:-1], segment_offsets[1:]):
        if abs(shift) >= stop - start:
            continue
        if shift >= 0:
            aligned[start + shift:stop] = readings[start:stop - shift]
        else:
            aligned[start:stop + shift] = readings[start - shift:stop]
    return aligned


def _saved_blocks(path, reference, source, expected, boundaries):
    paths = segment_files(path)
    if len(paths) + 1 != len(boundaries):
        raise ValueError("The input acquisition segments changed. Plot TOF again.")
    for (data, _), start, stop in zip(paths, boundaries[:-1], boundaries[1:]):
        with h5py.File(data, "r") as file:
            triggers = _trigger_times(file, reference)
            if not np.array_equal(triggers, expected[start:stop]):
                raise ValueError("The selected triggers changed. Plot TOF again before filtering or fitting.")
            name, field = ARRIVALS[source]
            for values, indices, unmatched in _tof_blocks(_table(file, name, field), field, triggers):
                yield values, indices + start, unmatched


def filtered_histogram(path, tof_result, metadata, reference, low, high, shift, *, weight_by_gmd2=False):
    """Gate within each acquisition and use the saved combined TOF grid."""
    with np.load(tof_result, allow_pickle=False) as original:
        if "trigger_times" not in original:
            raise ValueError("Plot TOF again before filtering this older result.")
        edges = original["edges"]
        expected = original["trigger_times"]
        boundaries = original["segment_offsets"] if "segment_offsets" in original else np.array([0, len(expected)])
        sources = [source for source in ARRIVALS if source in original]
    gmd = _gmd_metadata(metadata, len(expected), boundaries)
    if gmd["gmd_alignment_issue"]:
        raise ValueError(gmd["gmd_alignment_issue"])
    readings = gmd["gmd2_na"]
    low, high = gmd_limits(readings, low, high)
    aligned = gmd2_by_frame(readings, boundaries, shift)
    selected = np.isfinite(aligned) & (aligned >= low) & (aligned <= high)
    weights = np.zeros(len(expected), dtype=float) if weight_by_gmd2 else None
    if weights is not None:
        selected &= aligned > 0
    count = int(selected.sum())
    result = {"edges": edges, "selected_triggers": count, "triggers": len(expected),
              "trigger_times": expected, "selected_mask": selected, "segment_offsets": boundaries,
              "gmd_min": low, "gmd_max": high}
    print(f"GMD2 {low:g} to {high:g} nA, shift {shift:+d}: {count:,} selected triggers.")
    if weights is not None:
        if not count:
            raise ValueError("The GMD2 gate contains no aligned triggers with positive GMD2.")
        reference_na = float(aligned[selected].mean())
        with np.errstate(over="ignore", divide="ignore"):
            weights[selected] = reference_na / aligned[selected]
        if not np.isfinite(weights).all():
            raise ValueError("GMD2 weights are not finite. Increase the GMD2 minimum.")
        result.update(frame_weights=weights, gmd2_reference_na=reference_na)
        print(f"GMD2 reference: {reference_na:.6g} nA (mean of selected triggers).")
    for source in sources:
        counts = np.zeros(len(edges) - 1, dtype=np.int64 if weights is None else float)
        for values, indices, _ in _saved_blocks(path, reference, source, expected, boundaries):
            keep = selected[indices]
            counts += np.histogram(values[keep], bins=edges,
                                   weights=None if weights is None else weights[indices[keep]])[0]
        result[source] = counts
        if weights is None:
            print(f"{source}: {int(counts.sum()):,} selected counts in the TOF range.")
        else:
            print(f"{source}: {counts.sum():,.6g} weighted counts in the TOF range.")
    return result

def fit_filtered_peaks(path, filtered_result, reference, sources, region, estimates, baseline, tof_result=None):
    """Fit either histogram and count selected arrivals in each fitted ±3σ window."""
    from vmi_analysis.pes import PeakGuess, fit_peaks

    guesses = [PeakGuess("Gaussian", center, width) for center, width in estimates]
    results = {}
    errors = {}
    with np.load(filtered_result, allow_pickle=False) as filtered:
        if "selected_mask" not in filtered:
            raise ValueError("Apply the GMD2 range again before fitting this older spectrum.")
        edges = filtered["edges"]
        centers = (edges[:-1] + edges[1:]) / 2
        selected = filtered["selected_mask"]
        weights = filtered["frame_weights"] if "frame_weights" in filtered else None
        if weights is not None:
            results["gmd2_reference_na"] = float(filtered["gmd2_reference_na"])
        selected_count = int(filtered["selected_triggers"])
        if selected_count < 1:
            raise ValueError("Select at least one GMD2-gated trigger before fitting peak yields.")
        triggers = filtered["trigger_times"]
        boundaries = filtered["segment_offsets"] if "segment_offsets" in filtered else np.array([0, len(triggers)])
        fit_counts = {source: filtered[source] for source in sources if source in filtered}
        if tof_result is not None:
            with np.load(tof_result, allow_pickle=False) as unfiltered:
                for key in ("edges", "trigger_times"):
                    if not np.array_equal(unfiltered[key], filtered[key]):
                        raise ValueError("The unfiltered and GMD2-filtered TOF results do not match. Plot TOF again.")
                original_boundaries = (unfiltered["segment_offsets"] if "segment_offsets" in unfiltered
                                       else np.array([0, len(triggers)]))
                if not np.array_equal(original_boundaries, boundaries):
                    raise ValueError("The unfiltered and GMD2-filtered acquisition segments do not match.")
                fit_counts = {source: unfiltered[source] for source in sources if source in unfiltered}
        for source in sources:
            if source not in ARRIVALS or source not in filtered or source not in fit_counts:
                raise ValueError("Choose a count source from the filtered TOF result.")
            try:
                x, fitted, background, _, residual, peaks = fit_peaks(
                    centers, fit_counts[source], region, guesses, baseline,
                )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                errors[source] = str(exc)
                continue
            for peak in peaks:
                half_width = 3 * peak["gaussian_fwhm"] / (2 * sqrt(2 * log(2)))
                peak.update(low=peak["center"] - half_width, high=peak["center"] + half_width, counts=0)
            print(f"Counting {source} in {len(peaks)} Gaussian peak windows…")
            for values, assigned, _ in _saved_blocks(path, reference, source, triggers, boundaries):
                keep = selected[assigned]
                arrivals = values[keep]
                arrival_weights = weights[assigned[keep]] if weights is not None else None
                for peak in peaks:
                    inside = (arrivals >= peak["low"]) & (arrivals < peak["high"])
                    peak["counts"] += (int(np.count_nonzero(inside)) if arrival_weights is None
                                       else float(arrival_weights[inside].sum()))
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
    parser.add_argument("--weight-by-gmd2", action="store_true",
                        help="Weight gated frames by the selected mean GMD2 divided by frame GMD2")
    parser.add_argument("--filtered-result", type=Path, help="Existing GMD2-filtered histogram (.npz)")
    parser.add_argument("--fit-spec", type=json.loads, help="Gaussian peak fit settings as JSON")
    parser.add_argument("--source", choices=(*ARRIVALS, "both"), default="pixels")
    parser.add_argument("--reference", choices=REFERENCES, default=DEFAULT_REFERENCE)
    binning = parser.add_mutually_exclusive_group()
    binning.add_argument("--bins", type=int, default=DEFAULT_BINS)
    binning.add_argument("--bin-size", type=float, help="TOF bin width in ns")
    parser.add_argument("--minimum", type=float, default=0.0, help="Minimum TOF in ns")
    parser.add_argument("--maximum", type=float, help="Maximum TOF in ns (default: largest observed TOF)")
    args = parser.parse_args()
    try:
        output = validate_distinct_output(args.output, [("LUNA HDF5 input", args.input),
                                                       ("acquisition metadata", args.metadata),
                                                       ("unfiltered TOF result", args.tof_result),
                                                       ("filtered TOF result", args.filtered_result)])
        if args.filtered_result is not None or args.fit_spec is not None:
            if args.weight_by_gmd2:
                raise ValueError("Set GMD2 weighting when filtering TOF. Fitting uses the saved gate's weights.")
            if args.filtered_result is None or not isinstance(args.fit_spec, dict):
                raise ValueError("Peak fitting requires the filtered TOF result and fit settings.")
            try:
                result = fit_filtered_peaks(args.input, args.filtered_result, args.reference,
                                            **args.fit_spec, tof_result=args.tof_result)
            except TypeError as exc:
                raise ValueError(f"Invalid TOF peak fit settings. {exc}") from exc
        elif args.weight_by_gmd2 or any(value is not None for value in (args.tof_result, args.gmd_min, args.gmd_max, args.gmd_shift)):
            if any(value is None for value in (args.tof_result, args.metadata, args.gmd_shift)):
                raise ValueError("Filtering requires the unfiltered TOF result, metadata, and shift.")
            result = filtered_histogram(args.input, args.tof_result, args.metadata, args.reference,
                                        args.gmd_min, args.gmd_max, args.gmd_shift, weight_by_gmd2=args.weight_by_gmd2)
        else:
            sources = tuple(ARRIVALS) if args.source == "both" else (args.source,)
            result = histogram(args.input, sources, args.reference, args.bins, args.minimum, args.maximum,
                               metadata=args.metadata, bin_size_ns=args.bin_size)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            np.savez(handle, **result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"TOF: {exc}\n")


if __name__ == "__main__":
    main()
