"""Read frames for the Accumulate panel's interactive histograms."""

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


def _raw_files(h5_dir: str | Path) -> list[Path]:
    folder = Path(h5_dir).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"HDF5 directory not found: {folder}")
    files = sorted((p for p in folder.iterdir() if p.suffix == ".h5"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")
    return files


def list_raw_frames(h5_dir: str | Path, dataset: str = RAW_DATASET) -> tuple[FrameRef, ...]:
    """Return every frame across the ``*.h5`` files, sorted by file name."""
    files = _raw_files(h5_dir)
    signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    key = (str(files[0].parent.resolve()), dataset)
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


def average_raw_frames(h5_dir: str | Path, file_count: int,
                       rng: random.Random | None = None, *,
                       stopping=lambda: False) -> tuple[np.ndarray, int]:
    """Average every frame from a sample of distinct HDF5 files."""
    files = _raw_files(h5_dir)
    if not 1 <= file_count <= len(files):
        raise ValueError(f"Files to average must be between 1 and {len(files)}.")
    total = None
    frame_count = 0
    for path in (rng or random).sample(files, file_count):
        with h5py.File(path, "r") as handle:
            frames = handle[RAW_DATASET]
            if frames.ndim != 3 or not frames.size:
                raise ValueError(f"{path.name} has no valid image frames.")
            if total is None:
                total = np.zeros(frames.shape[1:], dtype=np.float64)
            elif frames.shape[1:] != total.shape:
                raise ValueError(f"{path.name} has a different frame shape.")
            for frame in frames:
                if stopping():
                    raise InterruptedError("Histogram preview stopped during shutdown.")
                total += frame
            frame_count += frames.shape[0]
    total /= frame_count
    return total, frame_count


def subtract_dark(values: np.ndarray, dark_path: Path) -> np.ndarray:
    dark = np.load(dark_path, allow_pickle=False)
    if not isinstance(dark, np.ndarray):
        dark.close()
        raise ValueError("Dark frame must be a numeric .npy array.")
    if dark.dtype.kind not in "fiu":
        raise ValueError("Dark frame must be a numeric .npy array.")
    if dark.shape != values.shape:
        raise ValueError(f"Dark frame shape {dark.shape} does not match frame shape {values.shape}.")
    return np.subtract(values, dark, dtype=np.float64)


def pick_frame(refs, index: int | None = None, rng: random.Random | None = None) -> FrameRef:
    """Return the requested global frame, or a uniform random frame if index is None."""
    if not refs:
        raise ValueError("No frames available")
    if index is None:
        return refs[(rng or random).randrange(len(refs))]
    if index < 0 or index >= len(refs):
        raise IndexError(f"Frame index {index} is out of range (0–{len(refs) - 1})")
    return refs[index]
