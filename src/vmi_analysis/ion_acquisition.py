"""Acquisition restart detection and saved LUNA segment collections."""

import json
from pathlib import Path
import struct

import numpy as np


PREFIX = struct.Struct("<qqdd")
TDC_PERIOD_TICKS = 2**32 * 250_000
SCHEMA = "vmi-luna-segments-1"


def read_timing(path):
    raw = path.read_bytes()
    if len(raw) < PREFIX.size:
        raise ValueError(f"{path.name} has an incomplete acquisition header.")
    sec, ns, _, _ = PREFIX.unpack_from(raw)
    if not 0 <= ns < 10**9 or (len(raw) - PREFIX.size) % 8:
        raise ValueError(f"Invalid TPX3 acquisition header or packet length in {path.name}.")
    words = np.frombuffer(raw, dtype="<u8", offset=PREFIX.size)
    references = words[(words >> 56) == 0x6F]
    if references.size != 1:
        raise ValueError(f"{path.name} contains {references.size} TDC1 rising edges. Expected one per frame.")
    word = int(references[0])
    coarse = (word >> 12) & 0xFFFFFFFF
    fine = ((((word >> 5) & 0xF) - 1) << 9) // 12
    tdc = coarse * 250_000 + ((word & 0xE00) | (fine & 0x1FF)) * (250_000 / 4096)
    return sec * 10**9 + ns, tdc


def acquisition_splits(wall_ns, tdc_ticks):
    """Split backward TDC jumps whose rollover interval disagrees with wall time by >1 s."""
    wall_step = np.diff(wall_ns)
    if np.any(wall_step <= 0):
        raise ValueError("Frame numbers and wall-clock timestamps must increase in the same order.")
    step = np.diff(tdc_ticks)
    expected_ns = (step % TDC_PERIOD_TICKS) / 10_000
    return np.flatnonzero((step < 0) & (np.abs(expected_ns - wall_step) > 1e9)) + 1


def manifest_path(output):
    return Path(output).with_suffix(".segments.json")


def segment_files(path):
    """Return data and metadata pairs from one HDF5 or a contained segment collection."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        return [(path, None)]
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(doc, dict) or doc.get("schema") != SCHEMA
            or not isinstance(doc.get("segments"), list) or not doc["segments"]):
        raise ValueError("Choose a TPX3 Dump segment collection or a LUNA HDF5 file.")
    pairs = []
    used = set()
    for segment in doc["segments"]:
        if not isinstance(segment, dict):
            raise ValueError("Each acquisition segment must list its data and metadata files.")
        files = []
        for key in ("data", "metadata"):
            value = segment.get(key)
            if not isinstance(value, str):
                raise ValueError("Each acquisition segment needs data and metadata paths.")
            candidate = (path.parent / value).resolve()
            if Path(value).is_absolute() or not candidate.is_relative_to(path.parent) or candidate.suffix.lower() not in (".h5", ".hdf5"):
                raise ValueError("Segment files must be HDF5 files within the collection directory.")
            if candidate in used:
                raise ValueError("A segment file is listed more than once.")
            used.add(candidate)
            files.append(candidate)
        pairs.append(tuple(files))
    return pairs


def collection_reads(path):
    pairs = segment_files(path)
    return [Path(path), *(item for pair in pairs for item in pair if item is not None)]
