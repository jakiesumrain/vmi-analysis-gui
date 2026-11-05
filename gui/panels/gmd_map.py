"""Panel: build_gmd_map.py — frame → GMD correspondence map."""

import os
import sys
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import (JobView, REPO_ROOT, num_input, open_folder_button,
                     path_input, persist)


def build(jobs: JobManager) -> None:
    theme.section_title(
        "GMD Map",
        "Match each camera frame to its GMD pulse-energy record by timestamp "
        "(build_gmd_map.py). Required before GMD weighting or binning.")

    view = JobView(jobs, results_builder=_results)

    with theme.panel_card():
        h5_dir = persist(path_input(
            "HDF5 directory", "", browse="dir",
            placeholder="folder containing RAW-*.h5 files"), "gmd_map.h5_dir")
        gmd_path = persist(path_input(
            "GMD timestamp file (current-time-both.txt)",
            "", browse="file", filters=[".txt"],
            placeholder="current-time-both.txt"), "gmd_map.gmd_path")
        output_path = persist(path_input(
            "Output map file", str(REPO_ROOT / "gmd_map_all.txt"),
            browse="save", filters=[".txt"],
            default_name="gmd_map_all.txt"), "gmd_map.output_path")
        with ui.row().classes("gap-4"):
            n_files = persist(num_input("First N files (0 = all)", 0, step=1),
                              "gmd_map.n_files")
        with ui.row().classes("gap-2"):
            theme.run_button(on_click=lambda: _run(jobs, view, h5_dir, gmd_path,
                                                   output_path, n_files))
            ui.button("Clear output", icon="delete",
                      on_click=lambda: _clear_output(output_path)) \
                .props("flat color=negative")

    view.place_console()


def _run(jobs, view, h5_dir, gmd_path, output_path, n_files):
    cmd = [sys.executable, str(REPO_ROOT / "build_gmd_map.py"),
           "--h5-dir", h5_dir.value,
           "--gmd-path", gmd_path.value,
           "--output-path", output_path.value,
           "--n-files", str(int(n_files.value or 0))]
    job = jobs.submit(Job(name="GMD map", cmd=cmd, cwd=str(REPO_ROOT),
                          extra={"output_path": output_path.value}))
    view.attach(job)


def _clear_output(output_path) -> None:
    """Delete the output map file so the next run rewrites it from scratch."""
    txt = (output_path.value or "").strip()
    if not txt:
        ui.notify("Set an output path first.", type="warning")
        return
    target = Path(txt)
    if not target.exists():
        ui.notify(f"Nothing to delete at {txt}", type="info")
        return

    with ui.dialog() as dlg, ui.card():
        ui.label("Delete GMD map file?").classes("vmi-caption")
        ui.label(str(target)).style("font-family: monospace")
        with ui.row().classes("gap-2"):
            ui.button("Delete", color="negative", on_click=lambda: do_delete())
            ui.button("Cancel", on_click=dlg.close).props("flat")
    dlg.open()

    def do_delete():
        try:
            target.unlink()
        except OSError as e:
            dlg.close()
            ui.notify(f"Could not delete {target}: {e}", type="negative")
            return
        dlg.close()
        ui.notify(f"Deleted {target}", type="positive")


def _results(job: Job):
    out = job.extra.get("output_path", "")
    if job.status == "done" and out:
        size = os.path.getsize(out) / 1e6
        with open(out, "rb") as fh:
            n_lines = sum(1 for _ in fh)
        ui.label(f"Map written: {out}  ({n_lines:,} lines, {size:.1f} MB)") \
            .classes("vmi-subtitle")
        open_folder_button(out)
