"""Read an HDF5 file's structure and summarize known LUNA datasets."""

from collections import Counter

import h5py
import numpy as np


ROWS_PER_BLOCK = 65_536
FIELDS = {
    "Clusters": {"cx": ("Centroid X", "px"), "cy": ("Centroid Y", "px"),
                 "size": ("Cluster size", "pixels"), "ctot": ("Cluster ToT", "ns"),
                 "sum_tot": ("Sum of pixel ToT", "ns"), "ctoa": None},
    "PixelHits": {"tot": ("Pixel ToT", "ns"), "toa": None},
    "TDCEvents": {"tdc_type": None, "timestamp": None},
    "frames": {"gmd1": ("GMD 1", ""), "gmd2": ("GMD 2", ""), "timestamp_ns": None},
}
TIME_FIELDS = {"Clusters": "ctoa", "PixelHits": "toa", "TDCEvents": "timestamp", "frames": "timestamp_ns"}


def check_cancelled(stopping):
    if stopping():
        raise ValueError("HDF5 inspection was stopped.")


def update_stats(stats, key, values):
    values = values[np.isfinite(values)]
    if values.size:
        low, high, total, count = stats.get(key, (float("inf"), float("-inf"), 0.0, 0))
        stats[key] = (min(low, values.min().item()), max(high, values.max().item()),
                      total + values.sum(dtype=np.float64), count + values.size)


def describe(stats, units):
    low, high, total, count = stats
    suffix = f" {units}" if units else ""
    return f"Mean {total / count:,.6g}{suffix} · range {low:,.6g} to {high:,.6g}{suffix}"


def event_summary(name, dataset, stopping):
    fields = [key for key in FIELDS[name] if key in (dataset.dtype.names or ())
              and dataset.dtype[key].kind in "uif"]
    if dataset.ndim != 1 or not fields:
        return []
    stats, types = {}, Counter()
    reader = dataset.fields(fields)
    for start in range(0, len(dataset), ROWS_PER_BLOCK):
        check_cancelled(stopping)
        block = reader[start:start + ROWS_PER_BLOCK]
        for key in fields:
            if key == "tdc_type":
                kinds, counts = np.unique(block[key], return_counts=True)
                types.update(dict(zip(kinds.tolist(), counts.tolist())))
            else:
                update_stats(stats, key, block[key])
    summary = []
    for key, spec in FIELDS[name].items():
        if spec is not None and key in stats:
            label, units = spec
            summary.append((label, describe(stats[key], units)))
    if time := stats.get(TIME_FIELDS[name]):
        low, high, _, count = time
        seconds = (high - low) * (1e-9 if name == "frames" else 1e-13)
        summary.append(("Time span", f"{seconds:,.6g} s"))
        if seconds > 0 and name != "frames":
            summary.append(("Mean rate", f"{count / seconds:,.6g} Hz"))
    if types:
        summary.append(("TDC types", " · ".join(f"{key}: {count:,}" for key, count in sorted(types.items()))))
    return summary


def timewalk_summary(dataset, stopping):
    if dataset.ndim != 2 or dataset.dtype.kind not in "uif":
        return []
    cols = np.asarray(dataset.attrs.get("col_names", []))
    if cols.dtype.names:
        cols = cols[cols.dtype.names[0]]
    names = [item.decode(errors="replace") if isinstance(item, bytes) else str(item) for item in cols.flat]
    if len(names) != dataset.shape[1] or "Count" not in names:
        return []
    count_col = names.index("Count")
    correction_col = names.index("AverageDToA") if "AverageDToA" in names else None
    stats = {}
    populated = high_count = samples = 0
    for start in range(0, len(dataset), ROWS_PER_BLOCK):
        check_cancelled(stopping)
        block = dataset[start:start + ROWS_PER_BLOCK]
        counts = block[:, count_col]
        valid = np.isfinite(counts) & (counts > 0)
        populated += np.count_nonzero(valid)
        high_count += np.count_nonzero(counts[valid] > 100)
        samples += int(counts[valid].sum(dtype=np.float64))
        if correction_col is not None:
            update_stats(stats, "correction", block[valid, correction_col].astype(np.float64) * 1e-4)
    summary = [("Populated bins", f"{populated:,}"), ("Bins with over 100 samples", f"{high_count:,}"),
               ("Samples", f"{samples:,}")]
    if "correction" in stats:
        summary.append(("Average ΔToA", describe(stats["correction"], "ns")))
    return summary


def inspect_hdf5(path, stopping):
    """Return display-ready summary rows and a text tree without retaining datasets."""
    summary, structure = [], []
    with h5py.File(path, "r") as file:
        def visit(name, item):
            check_cancelled(stopping)
            indent = "  " * (name.count("/") + 1) if name else ""
            path_label = f"/{name}"
            if isinstance(item, h5py.Dataset):
                structure.append(f"{indent}{path_label}  dataset  shape={item.shape}  dtype={item.dtype}")
                if datatype := h5py.check_string_dtype(item.dtype):
                    structure.append(f"{indent}  String encoding: {datatype.encoding}")
                for field in item.dtype.names or ():
                    field_type = item.dtype[field]
                    strings = h5py.check_string_dtype(field_type)
                    representation = f"{strings.encoding} string" if strings else str(field_type)
                    structure.append(f"{indent}  {field}: {representation}")
            else:
                structure.append(f"{indent}{path_label}  {type(item).__name__.lower()}")
            for key in item.attrs:
                value = item.attrs[key] if item.attrs.get_id(key).get_storage_size() <= 65_536 else "[large attribute]"
                preview = np.array2string(value, threshold=12) if isinstance(value, np.ndarray) else repr(value)
                structure.append(f"{indent}  @{key} = {preview}")

        visit("", file)
        file.visititems(visit)
        for name in (*FIELDS, "TimewalkMatrix", "TimewalkLookupTable", "ExposureTimeBoundaries", "RawData"):
            if not isinstance(dataset := file.get(name), h5py.Dataset) or dataset.shape is None:
                continue
            rows = [("Rows", f"{dataset.shape[0]:,}")] if dataset.ndim else [("Elements", "1")]
            try:
                if name in FIELDS:
                    rows += event_summary(name, dataset, stopping)
                elif name in ("TimewalkMatrix", "TimewalkLookupTable"):
                    rows += timewalk_summary(dataset, stopping)
            except OSError as exc:
                rows.append(("Statistics unavailable", str(exc)))
            summary.extend({"dataset": f"/{name}", "metric": metric, "value": value}
                           for metric, value in rows)
    return summary, "\n".join(structure)
