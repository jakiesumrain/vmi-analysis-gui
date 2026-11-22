"""Build matched detector-position and TOF views for ion momentum analysis."""

import argparse
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .ion_acquisition import segment_files
from .ion_tof import (ARRIVALS, DEFAULT_REFERENCE, REFERENCES, _table, _trigger_times,
                      assigned_arrival_blocks, histogram)
from .paths import validate_distinct_output
from .tof_bins import DEFAULT_BIN_SIZE_NS


COORDINATES = {"pixels": ("x", "y"), "clusters": ("cx", "cy")}
MAX_IMAGE_BINS = 1_048_576


def position_histograms(path, sources, reference=DEFAULT_REFERENCE, *, minimum=0, maximum=None,
                        bin_size_ns=DEFAULT_BIN_SIZE_NS):
    """Histogram positions of the same matched arrivals inside the TOF interval."""
    result = histogram(path, sources, reference, minimum=minimum, maximum=maximum,
                       bin_size_ns=bin_size_ns)
    low, high = result["edges"][[0, -1]]
    for source in sources:
        image = np.zeros((256, 256), dtype=np.int64)
        invalid = 0
        name, field = ARRIVALS[source]
        x_field, y_field = COORDINATES[source]
        for data, _ in segment_files(path):
            with h5py.File(data, "r") as file:
                triggers = _trigger_times(file, reference)
                table = _table(file, name, field, x_field, y_field, numeric="iuf")
                if table.dtype[field].kind not in "iu":
                    raise ValueError(f"{name}.{field} must contain integer detector-clock ticks.")
                for records, tof, _, _ in assigned_arrival_blocks(
                        table, field, triggers, coordinates=(x_field, y_field)):
                    selected = (tof >= low) & (tof <= high)
                    x, y = records[x_field][selected], records[y_field][selected]
                    valid = np.isfinite(x) & np.isfinite(y) & (x >= -0.5) & (y >= -0.5)
                    invalid += int(np.count_nonzero(~valid))
                    x, y = x[valid].astype(float), y[valid].astype(float)
                    if not x.size:
                        continue
                    width = max(image.shape[1], int(np.floor(x.max() + 0.5)) + 1)
                    height = max(image.shape[0], int(np.floor(y.max() + 0.5)) + 1)
                    if width * height > MAX_IMAGE_BINS:
                        raise ValueError(f"{name} coordinates require more than {MAX_IMAGE_BINS:,} image bins. "
                                         "Check that positions are in detector pixels.")
                    image = np.pad(image, ((0, height - image.shape[0]), (0, width - image.shape[1])))
                    columns = np.floor(x + 0.5).astype(np.intp)
                    rows = np.floor(y + 0.5).astype(np.intp)
                    np.add.at(image, (rows, columns), 1)
        result[f"{source}_image"] = image
        result[f"{source}_invalid_positions"] = invalid
        print(f"{source}: {int(image.sum()):,} mapped positions, {invalid:,} invalid positions.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=(*ARRIVALS, "both"), default="pixels")
    parser.add_argument("--reference", choices=REFERENCES, default=DEFAULT_REFERENCE)
    parser.add_argument("--bin-size", type=float, default=DEFAULT_BIN_SIZE_NS, help="TOF bin width in ns")
    parser.add_argument("--minimum", type=float, default=0, help="TOF minimum in ns")
    parser.add_argument("--maximum", type=float, help="TOF maximum in ns, omitted for automatic")
    args = parser.parse_args(argv)
    sources = tuple(ARRIVALS) if args.source == "both" else (args.source,)
    temporary = None
    try:
        if args.output.suffix.lower() != ".npz":
            raise ValueError("Output must end in .npz.")
        inputs = [item for pair in segment_files(args.input) for item in pair if item is not None]
        output = validate_distinct_output(args.output, [("input", path) for path in (args.input, *inputs)])
        result = position_histograms(args.input, sources, args.reference, minimum=args.minimum,
                                     maximum=args.maximum, bin_size_ns=args.bin_size)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **result)
        os.replace(temporary, output)
        print(f"Detector positions and TOF ready: {output}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"[Ion momenta] {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
