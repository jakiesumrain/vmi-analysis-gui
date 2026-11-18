"""Histogram Timepix3 arrivals relative to the selected LUNA trigger edge."""

import argparse
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


def _table(file, name, *fields):
    if not isinstance(file.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"Choose a LUNA HDF5 file with an embedded {name} dataset.")
    table = file[name]
    if (not isinstance(table, h5py.Dataset) or table.ndim != 1 or table.is_virtual
            or table.external or any(field not in (table.dtype.names or ())
                                     or table.dtype[field].kind not in "iu" for field in fields)):
        raise ValueError(f"{name} must be a table containing integer fields {', '.join(fields)}.")
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
        differences = arrivals[matched] - triggers[indices[matched]]
        yield differences * NS_PER_TICK, int(arrivals.size - differences.size)


def histogram(path, sources, reference=DEFAULT_REFERENCE, bins=DEFAULT_BINS, minimum=0.0, maximum=None):
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
                for values, _ in _tof_blocks(table, field, triggers):
                    if values.size:
                        last = max(last, float(values.max()))
            maximum = last if last > minimum else minimum + 1.0
        edges = np.linspace(minimum, maximum, bins + 1)
        if not np.all(edges[1:] > edges[:-1]):
            raise ValueError("The TOF range is too narrow for this bin count.")
        result = {"edges": edges, "triggers": len(triggers)}
        for source, table, field in columns:
            print(f"Building TOF histogram for {source}…")
            counts = np.zeros(bins, dtype=np.int64)
            unmatched = 0
            for values, before_first in _tof_blocks(table, field, triggers):
                counts += np.histogram(values, bins=edges)[0]
                unmatched += before_first
            outside = len(table) - unmatched - int(counts.sum())
            result[source] = counts
            result[f"{source}_unmatched"] = unmatched
            result[f"{source}_outside"] = outside
            print(f"{source}: {int(counts.sum()):,} plotted, {unmatched:,} without a preceding trigger, "
                  f"{outside:,} outside the TOF range.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Histogram output (.npz)")
    parser.add_argument("--source", choices=(*ARRIVALS, "both"), default="pixels")
    parser.add_argument("--reference", choices=REFERENCES, default=DEFAULT_REFERENCE)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--minimum", type=float, default=0.0, help="Minimum TOF in ns")
    parser.add_argument("--maximum", type=float, help="Maximum TOF in ns (default: largest observed TOF)")
    args = parser.parse_args()
    try:
        output = validate_distinct_output(args.output, [("LUNA HDF5 input", args.input)])
        sources = tuple(ARRIVALS) if args.source == "both" else (args.source,)
        result = histogram(args.input, sources, args.reference, args.bins, args.minimum, args.maximum)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            np.savez(handle, **result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"TOF: {exc}\n")


if __name__ == "__main__":
    main()
