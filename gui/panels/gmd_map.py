"""Panel: build_gmd_map.py — frame → GMD correspondence map."""

import sys

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import JobView, REPO_ROOT, num_input, path_input, serve_dir


def build(jobs: JobManager) -> None:
    theme.section_title(
        "GMD Map",
        "Match each camera frame to its GMD pulse-energy record by timestamp "
        "(build_gmd_map.py). Required before GMD weighting or binning.")

    view = JobView(jobs, results_builder=_results)

    with theme.panel_card():
        h5_dir = path_input("HDF5 directory", "", browse="dir",
                            placeholder="folder containing RAW-*.h5 files")
        gmd_path = path_input("GMD timestamp file (current-time-both.txt)",
                              "", browse="file", filters=[".txt"],
                              placeholder="current-time-both.txt")
        output_path = path_input("Output map file",
                                 str(REPO_ROOT / "gmd_map_all.txt"),
                                 browse="save", filters=[".txt"],
                                 default_name="gmd_map_all.txt")
        with ui.row().classes("gap-4"):
            n_files = num_input("First N files (0 = all)", 0, step=1)
        theme.run_button(on_click=lambda: _run(jobs, view, h5_dir, gmd_path,
                                               output_path, n_files))

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


def _results(job: Job):
    out = job.extra.get("output_path", "")
    if job.status == "done" and out:
        import os
        size = os.path.getsize(out) / 1e6
        n_lines = sum(1 for _ in open(out, "rb"))
        ui.label(f"Map written: {out}  ({n_lines:,} lines, {size:.1f} MB)") \
            .classes("vmi-subtitle")
        ui.link("open containing folder", serve_dir(out.rsplit("\\", 1)[0]
                                                    if "\\" in out else "."),
                new_tab=True).classes("vmi-caption")
