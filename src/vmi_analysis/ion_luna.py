"""Run ASI LUNA's TPX3 processor and publish a completed HDF5 result."""

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath
import re
import shlex
from shutil import copyfile, which
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import h5py

from .paths import validate_distinct_output

ACQUISITION_PREFIX = struct.Struct("<qqdd")
SERVAL_HEADER = struct.Struct("<4sBBH")
MAX_CHUNK_BYTES = 8191 * 8
FRAME_SUFFIX = re.compile(r"(?:^|_)frame_(\d+)$", re.IGNORECASE)
LUNA_COMMAND_PREFIX = "[LUNA] $ "


@dataclass(frozen=True)
class Option:
    flag: str
    label: str
    default: object = None
    choices: tuple = ()


PROCESS_OPTIONS = (
    Option("layout", "Chip layout", "single", ("single", "quad")),
    Option("clustering-algorithm", "Clustering algorithm", "DFSCluster", ("DFSCluster", "Kuttruff2024")),
    Option("eps-t", "Clustering time gap", "100ns"),
    Option("eps-s", "Clustering space gap (pixels)", 1),
    Option("kuttruff-buffer-size", "Kuttruff buffer size", 10),
    Option("disable-clustering", "Disable clustering", False),
    Option("raw-only", "Raw unpacking only", False),
    Option("calculate-timewalk-statistics", "Calculate timewalk statistics", False),
    Option("ctot-cut", "Cluster ToT cut (ns)", 450),
    Option("tof-tdc-reference", "Time-of-flight reference", None,
           ("TDC1Rising", "TDC1Falling", "TDC2Rising", "TDC2Falling")),
    Option("time-bin-interval", "Time bin interval", "500ms"),
    Option("num-time-bins", "Number of time bins", 3),
    Option("n-phases", "PLL phases", 1, (1, 2, 4, 8, 16)),
    Option("discard-pixel-data", "Discard pixel hits", False),
    Option("hdf5-chunk-size", "HDF5 chunk size (bytes)", 15000),
    Option("compression-algorithm", "Compression algorithm", None,
           ("BloscLz4hc", "BloscBlosclz", "BloscSnappy", "BloscZlib", "BloscZstd", "Deflate")),
    Option("compression-level", "Compression level", "5", ("off", *map(str, range(1, 10)))),
    Option("compression-shuffle", "Compression shuffle", "Byte", ("None", "Bit", "Byte")),
    Option("compression-nthreads", "Compression threads", 8),
    Option("log-level", "Log level", "info", ("off", "trace", "debug", "info", "warn", "error")),
)


def process_arguments(values):
    """Serialize the selected LUNA process options without using a shell."""
    args = []
    for option in PROCESS_OPTIONS:
        value = values.get(option.flag)
        if isinstance(option.default, bool):
            if value:
                args.append(f"--{option.flag}")
        elif value not in (None, ""):
            args.extend((f"--{option.flag}", str(value)))
    return args


def input_files(source):
    if source.is_dir():
        paths = sorted(path for path in source.glob("*.tpx3") if path.is_file())
        if not paths:
            raise ValueError(f"No .tpx3 files in {source}.")
        return paths
    if not source.is_file() or source.suffix.lower() != ".tpx3":
        raise ValueError("Choose an existing TPX3 file or a directory containing .tpx3 files.")
    return [source]


def check_headers(paths):
    for path in paths:
        with path.open("rb") as source:
            header = source.read(8)
        if len(header) < 8 or header[:4] != b"TPX3":
            raise ValueError(f"{path.name} lacks an ASI Serval TPX3 chunk header at byte 0. "
                             "Enable 32-byte acquisition preprocessing for files with a timestamp and GMD prefix.")


def frame_order(paths):
    numbered = [(FRAME_SUFFIX.search(path.stem), path) for path in paths]
    if not any(match for match, _ in numbered):
        return None
    if missing := next((path for match, path in numbered if match is None), None):
        raise ValueError(f"{missing.name} has no frame number. Use files with a consistent naming pattern.")
    return [path for _, path in sorted(numbered, key=lambda item: (int(item[0][1]), item[1].name))]


def _write_chunks(source, framed, body, path):
    while body:
        if len(body) % 8:
            raise ValueError(f"{path.name} ends with an incomplete 8-byte Timepix3 packet.")
        framed.write(SERVAL_HEADER.pack(b"TPX3", 0, 0, len(body)))
        framed.write(body)
        body = source.read(MAX_CHUNK_BYTES)


def preprocess_acquisition(paths, destination, metadata, *, numbered, combined=False):
    """Preserve each file's acquisition metadata and frame its single-chip packets."""
    destination.mkdir()
    records = []
    prepared = {}
    combined_path = destination / "acquisition.tpx3"
    with combined_path.open("wb") if combined else nullcontext() as combined_stream:
        for index, path in enumerate(paths):
            with path.open("rb") as source:
                prefix = source.read(ACQUISITION_PREFIX.size)
                if prefix.startswith(b"TPX3"):
                    raise ValueError(f"{path.name} already has a Serval header. Select ASI Serval input.")
                if len(prefix) != ACQUISITION_PREFIX.size:
                    raise ValueError(f"{path.name} has an incomplete 32-byte acquisition prefix.")
                seconds, nanoseconds, gmd1, gmd2 = ACQUISITION_PREFIX.unpack(prefix)
                if not 0 <= nanoseconds < 1_000_000_000:
                    raise ValueError(f"{path.name} has an invalid acquisition timestamp nanosecond field.")
                timestamp_ns = seconds * 1_000_000_000 + nanoseconds
                if not -(2**63) <= timestamp_ns < 2**63:
                    raise ValueError(f"{path.name} has an acquisition timestamp outside the 64-bit nanosecond range.")
                records.append((path.name, timestamp_ns, gmd1, gmd2))
                body = source.read(MAX_CHUNK_BYTES)
                if not body:
                    continue
                if combined_stream is not None:
                    _write_chunks(source, combined_stream, body, path)
                    continue
                # Fixed-width names preserve the chosen order in LUNA's filename sort.
                order = index if numbered else timestamp_ns + 2**63
                target = destination / f"{order:020d}_{index:09d}.tpx3"
                with target.open("wb") as framed:
                    _write_chunks(source, framed, body, path)
                prepared[target] = path
    with h5py.File(metadata, "w") as result:
        frames = result.create_dataset("frames", data=records if numbered else sorted(records, key=lambda row: row[1]), dtype=[
            ("filename", h5py.string_dtype("utf-8")), ("timestamp_ns", "<i8"),
            ("gmd1", "<f8"), ("gmd2", "<f8"),
        ])
        frames.attrs["timestamp_ns_unit"] = "nanoseconds since Unix epoch"
    return prepared


# WSL's Bash startup can add LUNA to the interactive PATH only. Resolve it
# there if necessary, then run a noninteractive subprocess with a pipe that
# closes when the Windows job is stopped. No user values are shell code.
_WSL_PROCESS = r"""
set -e
binary=$1; mode=$2; src=$3; dst=$4; matrix=$5; shift 5
if [ -z "$binary" ]; then
    binary=$(command -v tpx3dump || bash -ic 'command -v tpx3dump' || true)
fi
if [ -z "$binary" ] || [ ! -x "$binary" ]; then
    printf '%s\n' 'LUNA not found. Set the LUNA executable path or add tpx3dump to the WSL PATH.' >&2
    exit 127
fi
src=$(wslpath -a "$src")
dst=$(wslpath -a "$dst")
if [ -n "$matrix" ]; then
    set -- "$@" --timewalk-file "$(wslpath -a "$matrix")"
fi
if [ "$mode" = process ]; then
    set -- process -i "$src" -o "$dst" "$@"
else
    set -- apply-timewalk-correction --timewalk-file "$src" --output-file "$dst" "$@"
fi
printf '[LUNA] $'
printf ' %q' "$binary" "$@"
printf '\n'
exec 3<&0
"$binary" "$@" </dev/null 3<&- &
worker=$!
( IFS= read -r unused || kill "$worker" 2>/dev/null ) <&3 &
watcher=$!
trap 'kill "$worker" "$watcher" 2>/dev/null || true' EXIT
wait "$worker"
"""


def luna_command(source, output, matrix, args, distro="", executable="", *, windows, correction=False):
    mode = "apply-timewalk-correction" if correction else "process"
    if windows:
        distribution = ["--distribution", distro] if distro else []
        return ["wsl.exe", *distribution, "--cd", "/", "--exec", "bash", "-c", _WSL_PROCESS,
                "luna", executable, mode, str(source), str(output), str(matrix) if matrix else "", *args]
    if correction:
        return ["tpx3dump", mode, "--timewalk-file", str(source), "--output-file", str(output), *args]
    args = [*args, "--timewalk-file", str(matrix)] if matrix else args
    return ["tpx3dump", mode, "-i", str(source), "-o", str(output), *args]


def display_input_path(line, original_paths):
    """Use original paths in LUNA's printed input configuration."""
    if original_paths and (match := re.fullmatch(r"(\s*- path: )(.*?)(\r?\n)?", line)):
        prefix, staged, newline = match.groups()
        name = PureWindowsPath(staged.strip("'\"")).name
        if name in original_paths:
            return f"{prefix}{original_paths[name]}{newline or ''}"
    return line


def _run_luna(command, *, windows, original_paths=None, log=None):
    if not windows:
        command = [which(command[0]) or command[0], *command[1:]]
        print(f"{LUNA_COMMAND_PREFIX}{shlex.join(command)}", file=log, flush=True)
    with subprocess.Popen(command, stdin=subprocess.PIPE if windows else subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace", env={**os.environ, "WSL_UTF8": "1"},
                          creationflags=subprocess.CREATE_NO_WINDOW if windows else 0) as process:
        for line in process.stdout:
            print(display_input_path(line, original_paths), end="", file=log, flush=True)
            if "panicked at" in line:
                if windows:
                    process.stdin.close()
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise ValueError("LUNA reported a processing error. See the console above.")
        if code := process.wait():
            raise ValueError(f"LUNA exited with status {code}. See the console above.")


def publish_result(command, pending, output, *, windows, staged_paths=None, log=None):
    originals = {staged.name: original for staged, original in (staged_paths or {}).items()}
    _run_luna(command, windows=windows, original_paths=originals, log=log)
    with h5py.File(pending, "r") as result:
        for name in ("PixelHits", "Clusters", "TDCEvents", "TimewalkMatrix", "TimewalkLookupTable"):
            if name in result:
                print(f"{name}: {len(result[name]):,}", file=log, flush=True)
    pending.replace(output)
    print(f"HDF5 ready: {output}", file=log, flush=True)


def apply_timewalk(source, output, options, *, windows, log=None):
    with h5py.File(source, "r") as data:
        for name in ("TimewalkMatrix", "PixelHits", "Clusters"):
            if name not in data or len(data[name]) == 0:
                raise ValueError(f"The correction input needs a nonempty /{name} dataset. "
                                 "Run TPX3 Dump with timewalk statistics, clustering, and pixel hits enabled.")
    output.parent.mkdir(parents=True, exist_ok=True)
    args = process_arguments({"ctot-cut": options.ctot_cut, "log-level": options.log_level})
    with TemporaryDirectory(dir=output.parent, prefix=".luna-") as staging:
        pending = Path(staging) / output.name
        command = luna_command(source, pending, None, args, options.wsl_distribution,
                               options.wsl_executable, windows=windows, correction=True)
        publish_result(command, pending, output, windows=windows, log=log)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply-timewalk-correction", action="store_true",
                        help="Apply a timewalk correction to the input HDF5 and write a separate HDF5 file.")
    parser.add_argument("--metadata-output", type=Path,
                        help="Read 32-byte acquisition prefixes and save them to this separate HDF5 file.")
    parser.add_argument("--timewalk-file", type=Path)
    parser.add_argument("--wsl-distribution", default="")
    parser.add_argument("--wsl-executable", default="")
    for option in PROCESS_OPTIONS:
        if isinstance(option.default, bool):
            parser.add_argument(f"--{option.flag}", action="store_true")
        else:
            parser.add_argument(f"--{option.flag}", choices=option.choices or None,
                                type=int if type(option.default) is int else str)
    options = parser.parse_args(argv)
    values = {key.replace("_", "-"): value for key, value in vars(options).items()}
    source = options.input.expanduser().resolve()
    matrix = options.timewalk_file.expanduser().resolve() if options.timewalk_file else None
    windows = sys.platform == "win32"
    try:
        if options.apply_timewalk_correction:
            output = validate_distinct_output(options.output, [("timewalk input", source)], output_label="corrected HDF5 output")
            if output.suffix.lower() not in (".h5", ".hdf5"):
                raise ValueError("Choose an output ending in .h5 or .hdf5.")
            apply_timewalk(source, output, options, windows=windows)
            return
        if matrix and (not options.calculate_timewalk_statistics or not matrix.is_file()):
            raise ValueError("An existing timewalk file requires Calculate timewalk statistics.")
        inputs = [("TPX3 input", source), ("timewalk file", matrix)]
        output = validate_distinct_output(options.output, inputs, output_label="HDF5 output")
        if output.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError("Choose an output ending in .h5 or .hdf5.")
        metadata = None
        if options.metadata_output:
            metadata = validate_distinct_output(options.metadata_output, [*inputs, ("LUNA output", output)],
                                                output_label="metadata HDF5 output")
            if metadata.suffix.lower() not in (".h5", ".hdf5"):
                raise ValueError("Choose a metadata output ending in .h5 or .hdf5.")
            if options.layout not in (None, "single"):
                raise ValueError("The 32-byte acquisition format requires a single-chip layout.")
        paths = input_files(source)
        numbered_paths = frame_order(paths)
        if metadata is None:
            check_headers(paths)
        print(f"Processing {len(paths)} TPX3 file(s).", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            staging = Path(stack.enter_context(TemporaryDirectory(dir=output.parent, prefix=".luna-")))
            pending = staging / output.name
            luna_input = source
            staged_paths = {}
            if metadata is not None:
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata_stage = Path(stack.enter_context(TemporaryDirectory(dir=metadata.parent, prefix=".luna-")))
                pending_metadata = metadata_stage / metadata.name
                framed_dir = staging / "framed"
                staged_paths = preprocess_acquisition(numbered_paths or paths, framed_dir, pending_metadata,
                                                      numbered=numbered_paths is not None)
                print(f"Saved metadata for {len(paths)} file(s) and framed {len(staged_paths)} packet stream(s).", flush=True)
                pending_metadata.replace(metadata)
                print(f"Acquisition metadata ready: {metadata}", flush=True)
                if not staged_paths:
                    raise ValueError("The acquisition contains no Timepix3 packets.")
                luna_input = framed_dir if source.is_dir() else next(iter(staged_paths))
            elif numbered_paths is not None and numbered_paths != paths:
                luna_input = staging / "ordered"
                luna_input.mkdir()
                for index, path in enumerate(numbered_paths):
                    target = luna_input / f"{index:09d}.tpx3"
                    try:
                        os.link(path, target)
                    except OSError:
                        copyfile(path, target)
                    staged_paths[target] = path
            command = luna_command(luna_input, pending, matrix, process_arguments(values),
                                   options.wsl_distribution, options.wsl_executable, windows=windows)
            publish_result(command, pending, output, windows=windows, staged_paths=staged_paths)
    except FileNotFoundError as exc:
        raise SystemExit(f"[LUNA] Could not find {exc.filename}. Check LUNA, WSL and the selected paths.") from exc
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[LUNA] {exc}") from exc


if __name__ == "__main__":
    main()
