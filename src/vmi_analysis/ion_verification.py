"""Write TPX3 verification dumps and analyze them with the standalone routines."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import h5py
import numpy as np

from vmi_analysis import tpx3_analysis as reference
from vmi_analysis.paths import validate_distinct_output


SCHEMA = "vmi-tpx3-verification-1"
FRAME_DTYPE = np.dtype([
    ("timestamp_s", "<f8"), ("gmd1_na", "<f8"), ("gmd2_na", "<f8"),
    ("pixel_packets", "<i8"), ("size_bytes", "<i8"), ("delay_mm", "<f8"),
])


def unpack(source: Path, output: Path, unwrap=reference.UNWRAP_MODE):
    if source.is_dir():
        records = reference.scan_folder(source)
        folder = source
    elif source.is_file() and source.suffix.lower() == ".tpx3":
        records = reference.scan_files([source])
        folder = source.parent
    else:
        raise ValueError("Choose a .tpx3 file or a directory of .tpx3 files at one delay position.")
    if not records:
        raise ValueError("No valid TPX3 frame headers were found.")
    output = validate_distinct_output(output, [("TPX3 input", source)], output_label="verification dump")
    delays = [r.header_pos for r in records if np.isfinite(r.header_pos)]
    delay = float(np.median(delays)) if delays else reference.parse_delay_from_name(folder.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Decoding {len(records):,} frames in acquisition timestamp order ({unwrap}).", flush=True)
    with TemporaryDirectory(prefix=".tpx3-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dump.h5"
        with h5py.File(staged, "w") as file:
            file.attrs.update(schema=SCHEMA, dump_id=uuid4().hex, position=folder.name, unwrap=unwrap,
                              delay_mm=np.nan if delay is None else delay)
            file.create_dataset("frames", data=np.array([
                (r.t, r.gmd1, r.gmd2, r.npix, r.size, r.header_pos) for r in records
            ], dtype=FRAME_DTYPE))
            file.create_dataset("filenames", data=[r.name for r in records], dtype=h5py.string_dtype("utf-8"))
            tof = file.create_dataset("tof_ns", shape=(0,), maxshape=(None,), dtype="<f8", chunks=(65536,))
            offsets = np.zeros(len(records) + 1, dtype=np.int64)
            for start in range(0, len(records), 256):
                batch = [reference.read_record_tof(r, unwrap) for r in records[start:start + 256]]
                stop = start + len(batch)
                offsets[start + 1:stop + 1] = offsets[start] + np.cumsum([hits.size for hits in batch])
                if offsets[stop] != offsets[start]:
                    tof.resize(int(offsets[stop]), axis=0)
                    tof[offsets[start]:offsets[stop]] = np.concatenate(batch)
            file.create_dataset("frame_offsets", data=offsets)
        staged.replace(output)
    print(f"Saved {int(offsets[-1]):,} decoded TOFs from {len(records):,} frames to {output}", flush=True)


def _embedded_table(file, name):
    if not isinstance(file.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"The verification dump requires an embedded {name} dataset.")
    table = file[name]
    if not isinstance(table, h5py.Dataset) or table.ndim != 1 or table.is_virtual or table.external:
        raise ValueError(f"{name} must be an embedded one-dimensional dataset.")
    return table


def read_dump(path: Path, *, expected_id=None):
    with h5py.File(path, "r") as file:
        if file.attrs.get("schema") != SCHEMA:
            raise ValueError("Choose an HDF5 file produced by the verification TPX3 Dump panel.")
        dump_id = str(file.attrs["dump_id"])
        if expected_id is not None and expected_id != dump_id:
            raise ValueError("The input was replaced. Plot TOF again before applying the GMD2 range.")
        frames = _embedded_table(file, "frames")
        names = _embedded_table(file, "filenames")
        offsets = _embedded_table(file, "frame_offsets")
        tof = _embedded_table(file, "tof_ns")
        if (frames.dtype != FRAME_DTYPE or h5py.check_string_dtype(names.dtype) is None
                or offsets.dtype.kind != "i" or tof.dtype != np.dtype("<f8")
                or not len(frames) or len(names) != len(frames) or len(offsets) != len(frames) + 1):
            raise ValueError("Invalid verification dump table layout.")
        indices = offsets[:]
        if indices[0] != 0 or indices[-1] != len(tof) or np.any(indices[1:] < indices[:-1]):
            raise ValueError("Invalid verification dump frame offsets.")
        name = str(file.attrs["position"])
        records = [
            reference.FrameRecord(Path(filename), filename, float(frame["timestamp_s"]), float(frame["gmd1_na"]),
                                  float(frame["gmd2_na"]), int(frame["pixel_packets"]), int(frame["size_bytes"]),
                                  float(frame["delay_mm"]))
            for filename, frame in zip(names.asstr()[:], frames[:])
        ]
        tofs = np.split(tof[:], indices[1:-1])
        return reference.Position(path.parent, name, float(file.attrs["delay_mm"]), records), tofs, dump_id


def analyze(path: Path, *, bins=1200, gmd_window=None, shift=reference.FRAME_SHIFT, tracked=True,
            diagnostics=True, expected_id=None):
    if not 1 <= bins <= 20_000:
        raise ValueError("Choose between 1 and 20,000 TOF bins.")
    if gmd_window is not None and (not np.isfinite(gmd_window).all() or gmd_window[0] > gmd_window[1]):
        raise ValueError("Enter finite GMD2 limits with minimum at or below maximum.")
    position, tofs, dump_id = read_dump(path, expected_id=expected_id)
    gmd1, gmd2, pixels = position.arrays()
    # The standalone yield routine requires enough decoded hits to identify Xe peaks.
    if any(tof.size for tof in tofs):
        yields = reference.compute_yield(position, gmd_window, shift, tracked, tofs=tofs)
    else:
        yields = {"valid": False, "reason": "No decoded TOF hits."}
    all_tof = np.concatenate(tofs)
    edges = np.linspace(*reference.TOF_PLOT_RANGE, bins + 1)
    keep = (np.ones(position.n_frames, dtype=bool) if gmd_window is None
            else reference.select_tof_frames(gmd2, gmd_window, shift))
    selected = all_tof
    if gmd_window is not None:
        selected = np.concatenate([tof for tof, use in zip(tofs, keep) if use]) if keep.any() else np.empty(0)
    unfiltered = np.histogram(all_tof, bins=edges)[0]
    result = {
        "dump_id": dump_id, "edges": edges, "unfiltered": unfiltered,
        "filtered": unfiltered if selected is all_tof else np.histogram(selected, bins=edges)[0],
        "selected_frames": int(keep.sum()), "total_frames": position.n_frames,
        "yield_summary": json.dumps(yields),
    }
    if diagnostics:
        shifts = reference.frame_offset_scan(gmd2, pixels, min(reference.MAX_SHIFT_SCAN, position.n_frames - 1))
        result.update(gmd1=gmd1, gmd2=gmd2, correlations=np.array([
            (row["d"], row["r"], row["rho"], row["n"]) for row in shifts
        ]))
    if yields["valid"]:
        print(f"{yields['n_window']:,} selected frames, Xe+ {yields['xe1']:,}, Xe2+ {yields['xe2']:,}, "
              f"ratio {yields['ratio']:.8g}")
    else:
        print(yields.get("reason", "No frames passed the GMD2 selection."))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    dump = commands.add_parser("dump")
    spectrum = commands.add_parser("analyze")
    for command in (dump, spectrum):
        command.add_argument("--input", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    dump.add_argument("--unwrap", default=reference.UNWRAP_MODE, choices=("signed", "go4"))
    spectrum.add_argument("--bins", type=int, default=1200)
    spectrum.add_argument("--gmd-min", type=float)
    spectrum.add_argument("--gmd-max", type=float)
    spectrum.add_argument("--shift", type=int, default=reference.FRAME_SHIFT)
    spectrum.add_argument("--fixed-window", action="store_true")
    spectrum.add_argument("--expected-dump-id")
    args = parser.parse_args()
    try:
        if args.action == "dump":
            unpack(args.input, args.output, args.unwrap)
            return
        if (args.gmd_min is None) != (args.gmd_max is None):
            raise ValueError("Enter both GMD2 limits.")
        window = None if args.gmd_min is None else (args.gmd_min, args.gmd_max)
        output = validate_distinct_output(args.output, [("verification dump", args.input)], output_label="TOF result")
        data = analyze(args.input, bins=args.bins, gmd_window=window, shift=args.shift,
                       tracked=not args.fixed_window, diagnostics=window is None, expected_id=args.expected_dump_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            np.savez(file, **data)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"TPX3 verification: {exc}\n")


if __name__ == "__main__":
    main()
