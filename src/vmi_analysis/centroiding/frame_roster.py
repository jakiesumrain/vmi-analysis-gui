"""Acquisition-order processing outcomes in a raw Centroiding checkpoint."""

import h5py
import numpy as np


ROSTER = "frame_roster"
SKIPPED, PROCESSED, FAILED = 0, 1, 2


def write_frame_roster(checkpoint, frames, outcomes):
    roster = checkpoint.create_group(ROSTER)
    roster.attrs.update(skipped=SKIPPED, processed=PROCESSED, failed=FAILED)
    roster.create_dataset("id", data=np.arange(len(frames), dtype=np.int64))
    roster.create_dataset("source_file", data=[str(path) for path, _ in frames],
                          dtype=h5py.string_dtype("utf-8"))
    roster.create_dataset("frame_in_file", data=[index for _, index in frames], dtype="i8")
    roster.create_dataset("status", data=outcomes, dtype="u1")


def successful_shots(checkpoint):
    message = "The raw checkpoint needs a valid shot roster. Rerun Centroiding with a new checkpoint filename."
    roster = checkpoint.get(ROSTER)
    if not isinstance(roster, h5py.Group):
        raise ValueError(message)
    ids, files, indices, status = (roster.get(name) for name in ("id", "source_file", "frame_in_file", "status"))
    if not all(isinstance(column, h5py.Dataset) for column in (ids, files, indices, status)):
        raise ValueError(message)
    if (ids.ndim != 1 or ids.dtype.kind != "i" or ids.dtype.itemsize != 8
            or any(column.shape != ids.shape for column in (files, indices, status))
            or h5py.check_string_dtype(files.dtype) is None or indices.dtype.kind not in "iu"
            or status.dtype.kind not in "iu"):
        raise ValueError(message)
    shot_ids, outcomes = ids[:], status[:]
    if (not len(shot_ids) or shot_ids[0] < 0 or np.any(shot_ids[1:] <= shot_ids[:-1])
            or np.any(indices[:] < 0) or not np.isin(outcomes, (SKIPPED, PROCESSED, FAILED)).all()):
        raise ValueError(message)
    return shot_ids[outcomes == PROCESSED]
