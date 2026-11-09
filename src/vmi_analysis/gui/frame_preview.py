"""Read one raw frame for an interactive pixel-value histogram.

The Accumulate panel previews a single frame from the sorted ``RAW-*.h5``
files without launching the accumulation job: enumerating frames (the same
ordering the Movie tool uses) and reading one 1024×1024 frame is fast enough
to keep in the web process. Analysis scripts remain subprocess-only; this
module is a read-only GUI helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import h5py
import numpy as np

RAW_DATASET = "raw/data"


@dataclass(frozen=True)
class FrameRef:
    """One frame in acquisition order across the sorted files."""

    path: Path
    frame_index: int   # index inside the file
    global_index: int  # index across all files, sorted by file name

    @property
    def file_name(self) -> str:
        return self.path.name


# (directory, dataset) -> (file signature, refs). The signature is rebuilt
# whenever a file is added, removed, renamed, or changed.
_FRAME_CACHE: dict[tuple[str, str], tuple[tuple, tuple[FrameRef, ...]]] = {}


def list_raw_frames(h5_dir: str | Path, dataset: str = RAW_DATASET) -> tuple[FrameRef, ...]:
    """Return every frame across the ``*.h5`` files, sorted by file name."""
    folder = Path(h5_dir).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"HDF5 directory not found: {folder}")
    files = sorted((p for p in folder.iterdir() if p.suffix == ".h5"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")

    signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    key = (str(folder.resolve()), dataset)
    cached = _FRAME_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    refs: list[FrameRef] = []
    for path in files:
        try:
            with h5py.File(path, "r") as handle:
                if dataset not in handle:
                    raise ValueError(f"{path.name} has no {dataset!r} dataset")
                count = int(handle[dataset].shape[0])
        except OSError as exc:
            raise OSError(f"Cannot read {path.name}: {exc}") from exc
        base = len(refs)
        refs.extend(FrameRef(path=path, frame_index=i, global_index=base + i)
                    for i in range(count))
    _FRAME_CACHE[key] = (signature, tuple(refs))
    return tuple(refs)


def load_raw_frame(ref: FrameRef, dataset: str = RAW_DATASET) -> np.ndarray:
    """Read the pixel data of one frame listed by :func:`list_raw_frames`."""
    with h5py.File(ref.path, "r") as handle:
        return np.asarray(handle[dataset][ref.frame_index])


def pick_frame(refs, index: int | None = None, rng: random.Random | None = None) -> FrameRef:
    """Return the requested global frame, or a uniform random frame if index is None."""
    if not refs:
        raise ValueError("No frames available")
    if index is None:
        return refs[(rng or random).randrange(len(refs))]
    if index < 0 or index >= len(refs):
        raise IndexError(f"Frame index {index} is out of range (0–{len(refs) - 1})")
    return refs[index]
