"""Numerical sidecars for interactive GUI plots; PNG exports stay standalone."""

import json
from pathlib import Path

import numpy as np

TUNING_DATA = "01_tuning_check_picasso.npz"
SUMMARY_DATA = "04_final_summary.npz"


def save_plot_data(path, metadata, **arrays):
    """Use numeric arrays and JSON metadata only (never pickle objects)."""
    np.savez_compressed(Path(path), metadata=np.asarray(json.dumps(metadata)), **arrays)


def load_plot_data(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        arrays = {key: saved[key] for key in saved.files if key != "metadata"}
    return metadata, arrays
