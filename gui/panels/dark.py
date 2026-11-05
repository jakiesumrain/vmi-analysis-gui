"""Panel: compute_dark.py — mean/median dark frame from dark HDF5 files."""

import sys
from pathlib import Path

import numpy as np
from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     path_input, render_image, zoomable_image)

SCRIPT = REPO_ROOT / "centroiding" / "compute_dark.py"


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Dark Frame",
        "Compute a mean or median dark frame from dark HDF5 files "
        "(centroiding/compute_dark.py). Point the Centroiding panel's "
        "'Pre-computed dark .npy' at the result to skip recomputing it "
        "on every run.")

    view = JobView(jobs, results_builder=_results)

    with theme.panel_card():
        dark_dir = path_input("Dark HDF5 directory", "", browse="dir",
                              placeholder="folder containing dark RAW-*.h5 files")
        with ui.row().classes("gap-4 flex-wrap"):
            pattern = ui.input("HDF5 pattern", value="RAW-*.h5") \
                .props("outlined dense")
            dataset = ui.input("Frame dataset", value="raw/data") \
                .props("outlined dense")
        with ui.row().classes("gap-4 flex-wrap"):
            method = ui.select(["median", "mean"], value="median",
                               label="Method").props("outlined dense")
            frames = num_input("Dark frame count (median)", 2000, step=100)
        out_dir = path_input("Output directory",
                             str(REPO_ROOT / "centroiding"), browse="dir")

        def run():
            npy = str(Path(out_dir.value) / f"dark_{method.value}.npy")
            info = str(Path(out_dir.value)
                       / f"dark_{method.value}_info.txt")
            cmd = [sys.executable, str(SCRIPT),
                   "--config", str(REPO_ROOT / "centroiding" / "config.toml"),
                   "--dark-dir", dark_dir.value,
                   "--pattern", pattern.value,
                   "--dataset", dataset.value,
                   "--method", method.value,
                   "--frames", str(int(frames.value or 2000)),
                   "--output-dir", out_dir.value]
            job = jobs.submit(Job(name="Dark frame", cmd=cmd,
                                  cwd=str(REPO_ROOT),
                                  extra={"dark_npy": npy, "info": info}))
            view.attach(job)

        theme.run_button(on_click=run)

    view.place_console()


def _results(job: Job):
    if job.status != "done":
        return
    npy = Path(job.extra["dark_npy"])
    if not npy.exists():
        return
    data = np.load(npy)
    ui.label(f"Dark frame: {npy}  —  mean={data.mean():.1f}, "
             f"std={data.std():.1f}, min={data.min():.0f}, "
             f"max={data.max():.0f}").classes("vmi-subtitle")
    png = render_image(data, GENERATED / "dark_frame.png",
                       cmap="gray", log=False, vmin_pct=0, vmax_pct=99.9)
    zoomable_image(file_url(png)).classes("w-full max-w-2xl")
    info = Path(job.extra["info"])
    if info.exists():
        ui.label(info.read_text(encoding="utf-8").strip()) \
            .classes("vmi-caption")
