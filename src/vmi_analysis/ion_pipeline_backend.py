"""Standalone decoder input for the Ion Pipeline verification branch."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from vmi_analysis import tpx3_analysis as reference
from vmi_analysis.ion_pipeline import PositionData
from vmi_analysis.ion_verification import read_dump, unpack


NAME = "Standalone decoder"
DEFAULT_SHIFT = reference.FRAME_SHIFT
SHIFT_HELP = "The standalone convention pairs GMD2 row j with camera frame j − shift."
fit_peak = reference.fit_gaussian_peak


def decode(folder, delay, output, options):
    with TemporaryDirectory(dir=output, prefix=".ion-position-") as temporary:
        cache = Path(temporary)
        if options["keep_dumps"]:
            cache = output / "dumps"
            cache.mkdir(exist_ok=True)
        target = cache / (folder.name + ".h5")
        unpack(folder, target, options["unwrap"])
        position, tofs, _ = read_dump(target)
    gmd2 = position.arrays()[1]
    indices = np.repeat(np.arange(position.n_frames), [tof.size for tof in tofs])
    return PositionData(position.delay, np.concatenate(tofs), indices, gmd2, [position.n_frames])


def select_frames(data, options):
    return reference.select_tof_frames(data.gmd2_na, (options["gmd_min"], options["gmd_max"]), options["gmd_shift"])


def y_label(_options):
    return "Pixel hits per selected frame"
