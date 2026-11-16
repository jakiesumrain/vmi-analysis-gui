"""Panel: build_gmd_map.py — frame → GMD correspondence map."""

import os
import sys
from pathlib import Path

from nicegui import ui
from vmi_analysis.paths import validate_distinct_output

from ..jobs import Job, JobManager
from .. import theme
from .common import (JobView, REPO_ROOT, num_input, open_folder_button,
                     path_input, persist, confirm_clear_outputs)


def build(jobs: JobManager) -> None:
    theme.section_title(
        "GMD Map",
        "Match camera frames to GMD pulse-energy records by timestamp.")

    view = JobView(jobs, job_name='GMD map', results_builder=_results)

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
        with ui.row().classes("gap-4 flex-wrap"):
            pattern = persist(ui.input("HDF5 pattern", value="RAW-*.h5")
                              .props("outlined dense"), "gmd_map.h5_pattern")
            pattern.tooltip("Glob inside the HDF5 directory. Use *RAW-*.h5 "
                            "when the filenames have a prefix before RAW-.")
            n_files = persist(num_input("First N files (0 = all)", 0, step=1),
                              "gmd_map.n_files")
        with ui.row().classes("gap-2"):
            view.run_button(on_click=lambda: _run(jobs, view, h5_dir, gmd_path,
                                                   output_path, n_files, pattern))
            ui.button("Clear output", icon="delete",
                      on_click=lambda: _clear_output(output_path, h5_dir, gmd_path, view)) \
                .props("flat color=negative")

    view.place_console()


def _run(jobs, view, h5_dir, gmd_path, output_path, n_files, pattern):
    cmd = [sys.executable, "-m", "vmi_analysis.gmd.build_gmd_map",
           "--h5-dir", h5_dir.value,
           "--h5-pattern", (pattern.value or "RAW-*.h5").strip(),
           "--gmd-path", gmd_path.value,
           "--output-path", output_path.value,
           "--n-files", str(int(n_files.value or 0))]
    job = jobs.submit(Job(name="GMD map", cmd=cmd, cwd=str(REPO_ROOT),
                          extra={"output_path": output_path.value}))
    view.attach(job)


def _clear_output(output_path, h5_dir, gmd_path, view) -> None:
    """Delete the output map file so the next run rewrites it from scratch."""
    txt = (output_path.value or "").strip()
    if not txt:
        ui.notify("Set an output path first.", type="warning")
        return
    try:
        target = validate_distinct_output(txt, (("timestamp file", gmd_path.value),
                                               ("HDF5 directory", h5_dir.value)),
                                          cwd=REPO_ROOT, output_label="GMD map output")
    except (OSError, ValueError) as exc:
        ui.notify(str(exc), type="warning")
        return
    confirm_clear_outputs([target], "Delete GMD map file?", on_cleared=view.results.clear)


def _results(job: Job):
    out = job.extra.get("output_path", "")
    if job.status == "done" and out:
        out = Path(job.cwd or REPO_ROOT) / out
        if not out.is_file():
            return
        size = os.path.getsize(out) / 1e6
        with open(out, "rb") as fh:
            n_lines = sum(1 for _ in fh)
        ui.label(f"Map written: {out}  ({n_lines:,} lines, {size:.1f} MB)") \
            .classes("vmi-subtitle")
        open_folder_button(out)
