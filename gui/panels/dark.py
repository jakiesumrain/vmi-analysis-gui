"""Panel: compute_dark.py — mean/median dark frame from dark HDF5 files."""

import sys
from pathlib import Path

import numpy as np
from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     open_folder_button, path_input, persist, render_image,
                     zoomable_image)

SCRIPT = REPO_ROOT / "centroiding" / "compute_dark.py"


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Dark Frame",
        "Compute a mean or median dark frame from dark HDF5 files "
        "(centroiding/compute_dark.py). Point the Centroiding panel's "
        "'Pre-computed dark .npy' at the result to skip recomputing it "
        "on every run.")

    view = JobView(jobs, job_name='Dark frame', results_builder=_results)

    with theme.panel_card():
        dark_dir = persist(path_input(
            "Dark HDF5 directory", "", browse="dir",
            placeholder="folder containing dark RAW-*.h5 files"),
            "dark.dark_dir")
        with ui.row().classes("gap-4 flex-wrap"):
            pattern = persist(ui.input("HDF5 pattern", value="RAW-*.h5")
                              .props("outlined dense"), "dark.pattern")
            dataset = persist(ui.input("Frame dataset", value="raw/data")
                              .props("outlined dense"), "dark.dataset")
        with ui.row().classes("gap-4 flex-wrap"):
            method = persist(ui.select(["median", "mean"], value="median",
                                       label="Method")
                             .props("outlined dense"), "dark.method")
            frames = persist(num_input("Dark frame count (0 = all)", 2000,
                                       step=100), "dark.frames")
        out_dir = persist(path_input("Output directory",
                                     str(REPO_ROOT / "centroiding"),
                                     browse="dir"), "dark.out_dir")

        def run():
            npy = str(Path(out_dir.value) / f"dark_{method.value}.npy")
            info = str(Path(out_dir.value)
                       / f"dark_{method.value}_info.txt")
            cmd = [sys.executable, str(SCRIPT),
                   "--dark-dir", dark_dir.value,
                   "--pattern", pattern.value,
                   "--dataset", dataset.value,
                   "--method", method.value,
                   "--frames", str(int(frames.value if frames.value is not None else 2000)),
                   "--output-dir", out_dir.value]
            job = jobs.submit(Job(name="Dark frame", cmd=cmd,
                                  cwd=str(REPO_ROOT),
                                  extra={"dark_npy": npy, "info": info}))
            view.attach(job)

        with ui.row().classes("gap-2"):
            view.run_button(on_click=run)
            ui.button("Clear output", icon="delete",
                      on_click=lambda: _clear_output(out_dir, method)) \
                .props("flat color=negative")

    view.place_console()


def _clear_output(out_dir, method) -> None:
    """Delete dark_{method}.npy (+ info) so the next run recomputes."""
    folder = Path((out_dir.value or "").strip())
    if not folder.is_dir():
        ui.notify("Set an output directory first.", type="warning")
        return
    targets = [folder / f"dark_{method.value}.npy",
               folder / f"dark_{method.value}_info.txt"]
    existing = [p for p in targets if p.exists()]
    if not existing:
        ui.notify(f"Nothing to delete in {folder}", type="info")
        return

    with ui.dialog() as dlg, ui.card():
        ui.label("Delete dark frame outputs?").classes("vmi-caption")
        for p in existing:
            ui.label(str(p)).style("font-family: monospace")
        with ui.row().classes("gap-2"):
            ui.button("Delete", color="negative",
                      on_click=lambda: do_delete())
            ui.button("Cancel", on_click=dlg.close).props("flat")
    dlg.open()

    def do_delete():
        failed = []
        for p in existing:
            try:
                p.unlink()
            except OSError as e:
                failed.append(f"{p.name}: {e}")
        dlg.close()
        if failed:
            ui.notify("Could not delete: " + "; ".join(failed),
                      type="negative")
        else:
            ui.notify(f"Deleted {len(existing)} file(s) from {folder}",
                      type="positive")


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
    open_folder_button(npy)
    png = render_image(data, GENERATED / "dark_frame.png",
                       cmap="gray", log=False, vmin_pct=0, vmax_pct=99.9)
    zoomable_image(file_url(png)).classes("w-full max-w-2xl")
    info = Path(job.extra["info"])
    if info.exists():
        ui.label(info.read_text(encoding="utf-8").strip()) \
            .classes("vmi-caption")
